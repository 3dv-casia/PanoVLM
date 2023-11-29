/*
 * @Author: Diantao Tu
 * @Date: 2022-04-24 13:42:34
 */
#include "CameraLidarTrackAssociate.h"

using namespace std;

void Visualize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs_all,
                    const std::vector<Frame>& frames, const std::vector<Velodyne>& lidars,
                    const std::string path, const Eigen::Matrix4d& T_cl_init, 
                    int line_width, int point_size )
{
    if(!boost::filesystem::exists(path))
        boost::filesystem::create_directories(path);
    LOG(INFO) << "save joint virtualization result in " << path;
    #pragma omp parallel
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
    {
        #pragma omp single nowait
        {
            const size_t image_idx = it->first.first;
            const size_t lidar_idx = it->first.second;
            
            Eigen::Matrix4d T_cl = T_cl_init;
            if(frames[image_idx].IsPoseValid() && lidars[image_idx].IsPoseValid())
                T_cl = frames[image_idx].GetPose().inverse() * lidars[lidar_idx].GetPose();

            cv::Mat img_line = DrawLinePairsOnImage(frames[image_idx].GetImageGray(), it->second, T_cl, line_width);                
            cv::imwrite(path + "/line_pair_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_line);        
        }
    }
}

eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> AssociateLinesInTrack( 
                                                const std::vector<LineTrack>& image_tracks, const std::vector<LineTrack>& lidar_tracks,
                                                const std::vector<PanoramaLine>& image_lines_all, const std::vector<Frame>& frames,
                                                const std::vector<Velodyne>& lidars, const std::vector<std::vector<int>>& each_frame_neighbor,
                                                const Eigen::Matrix4d& T_cl_init)
                                                
{
    // 用一个mask来表示哪些直线（图像直线、雷达直线）是被包含于track中的，哪些是没被包含的
    // 在进行特征匹配的时候，只选择被包含于track中的直线进行特征匹配
    vector<vector<bool>> image_mask_all, lidar_mask_all;
    for(size_t i = 0; i < image_lines_all.size(); i++)
        image_mask_all.push_back(vector<bool>(image_lines_all[i].GetLines().size(), false));
    for(size_t i = 0; i < lidars.size(); i++)
        lidar_mask_all.push_back(vector<bool>(lidars[i].edge_segmented.size(), false));
    for(const LineTrack& track : image_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            image_mask_all[feature.first][feature.second] = true;
    }
    for(const LineTrack& track : lidar_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            lidar_mask_all[feature.first][feature.second] = true;
    }
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all;
    // #pragma omp parallel for schedule(dynamic)
    for(size_t frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        for(const int& lidar_id : each_frame_neighbor[frame_id])
        {
            const Velodyne& lidar = lidars[lidar_id];
            Eigen::Matrix4d T_cl = T_cl_init;
            if(frames[frame_id].IsPoseValid() && lidar.IsPoseValid())
            {
                Eigen::Matrix4d T_wc = frames[frame_id].GetPose();
                Eigen::Matrix4d T_wl = lidar.GetPose();
                T_cl = T_wc.inverse() * T_wl;
            }
            CameraLidarLineAssociate associate(frames[frame_id].GetImageRows(), frames[frame_id].GetImageCols());
            // 判断LiDAR是否已经进行过直线拟合了，如果是的话就以此为先验信息，得到更准确的匹配效果
            if(!lidar.edge_segmented.empty())
                associate.AssociateByAngle(image_lines_all[frame_id].GetLines(), lidar.edge_segmented, lidar.segment_coeffs,
                             lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points, T_cl, true,
                             image_mask_all[frame_id], lidar_mask_all[lidar_id]);
            else 
                associate.Associate(image_lines_all[frame_id].GetLines(), lidar.cornerLessSharp, T_cl);
            vector<CameraLidarLinePair> pairs = associate.GetAssociatedPairs();
            for(CameraLidarLinePair& p : pairs)
            {
                p.image_id = frame_id;
                p.lidar_id = lidar_id;
            }
            #pragma omp critical
            {
                line_pairs_all[{frame_id, lidar_id}] = pairs;
            }
        }
    }
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs in init track association";

    return line_pairs_all;
}

eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> AssociateTrack(
                                                const std::vector<LineTrack>& image_tracks, const std::vector<LineTrack>& lidar_tracks, 
                                                const std::vector<PanoramaLine>& image_lines_all, const std::vector<Frame>& frames,
                                                const std::vector<Velodyne>& lidars, const std::vector<std::vector<int>>& each_frame_neighbor,
                                                const Eigen::Matrix4d& T_cl_init
                                                )
{
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_track 
                = AssociateLinesInTrack(image_tracks, lidar_tracks, image_lines_all, frames, lidars, each_frame_neighbor, T_cl_init);
    Visualize(line_pairs_track, frames, lidars, "./track_match/", T_cl_init, 7, 7);
    // 从特征到track的关联，key = {image id, line id}  value=该特征对应的trackid的集合
    map<pair<uint32_t, uint32_t>, set<uint32_t>> feature_to_image_trackid, feature_to_lidar_trackid;
    for(const LineTrack& track : image_tracks)
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            feature_to_image_trackid[feature].insert(track.id);
    for(const LineTrack& track : lidar_tracks)
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            feature_to_lidar_trackid[feature].insert(track.id);
    // 邻接矩阵，代表lidar track和 image track之间是否存在匹配对，比如这个矩阵的第i行第j列的值为k，就代表第
    // i个image track 和第j个lidar track中有k条直线是相互匹配的
    // 这里用了稀疏矩阵，因为本身track数量还是比较多的，但是绝大多数track之间没有交集，稀疏矩阵比较适合
    Eigen::SparseMatrix<uint16_t> adjacency_matrix(image_tracks.size(), lidar_tracks.size());
    // 先给个初始值，大概每个image track能和4个lidar track相匹配上
    adjacency_matrix.reserve(image_tracks.size() * 4);  
    // 遍历所有的图像-LiDAR直线匹配对，找到图像track和LiDAR track之间的关联
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_track.begin();
        it != line_pairs_track.end(); it++)
    {
        for(const CameraLidarLinePair& line_pair : it->second)
        {
            // 找到当前的图像直线特征对应的track id的集合
            const set<uint32_t>& image_trackids = feature_to_image_trackid.find({it->first.first, line_pair.image_line_id}) -> second;
            // 找到当前的LiDAR特征对应的track id的集合
            const set<uint32_t>& lidar_trackids = feature_to_lidar_trackid.find({it->first.second, line_pair.lidar_line_id}) -> second;
            for(const uint32_t& image_track_id : image_trackids)
                for(const uint32_t& lidar_track_id : lidar_trackids)
                    adjacency_matrix.coeffRef(image_track_id, lidar_track_id) += 1;
        }
    }
    line_pairs_track.clear();
    // 根据track id 和 image id 找到当前对应的直线id
    // key = {track id, image id}  value = 当前图像上属于当前track的直线id的集合
    map<pair<uint32_t, uint32_t>, vector<uint32_t>>  track_image_to_feature, track_lidar_to_feature;
    vector<set<uint32_t>> image_in_each_track(image_tracks.size()), lidar_in_each_track(image_tracks.size());
    for(const LineTrack& track : image_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
        {
            track_image_to_feature[{track.id, feature.first}].push_back(feature.second);
            image_in_each_track[track.id].insert(feature.first);
        }
    }
    for(const LineTrack& track : lidar_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
        {
            track_lidar_to_feature[{track.id, feature.first}].push_back(feature.second);
            lidar_in_each_track[track.id].insert(feature.first);
        }
    }
    
	for(int col = 0; col < adjacency_matrix.outerSize(); col++)
	{
        for (Eigen::SparseMatrix<uint16_t>::InnerIterator it(adjacency_matrix, col); it; ++it)
		{
            int match_threshold = min(lidar_tracks[it.col()].feature_pairs.size(), image_tracks[it.row()].feature_pairs.size());
            if(it.value() < match_threshold - 2)
                continue;
            // if(it.value() < lidar_tracks[it.col()].feature_pairs.size() / 2 || 
            //     it.value() < image_tracks[it.row()].feature_pairs.size() / 2)
            //     continue;
            
            for(const uint32_t& image_id : image_in_each_track[it.row()])
            {
                for(const uint32_t& lidar_id : each_frame_neighbor[image_id])
                {
                    if(lidar_in_each_track[it.col()].count(lidar_id) == 0)
                        continue;
                    for(const uint32_t& image_line_id : track_image_to_feature[{it.row(), image_id}])
                    {
                        for(const uint32_t& lidar_line_id : track_lidar_to_feature[{it.col(), lidar_id}])
                        {
                            CameraLidarLinePair association;
                            association.image_line_id = image_line_id;
                            association.lidar_line_id = lidar_line_id;
                            association.image_line = image_lines_all[image_id].GetLines()[image_line_id];
                            association.lidar_line_start = lidars[lidar_id].end_points[lidar_line_id * 2];
                            association.lidar_line_end = lidars[lidar_id].end_points[lidar_line_id * 2 + 1];
                            line_pairs_track[{image_id, lidar_id}].push_back(association);
                        }
                    }
                }
            }
		}
    }
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_track.begin();
        it != line_pairs_track.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs in track association";

    return line_pairs_track;
}