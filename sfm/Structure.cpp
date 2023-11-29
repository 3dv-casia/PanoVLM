/*
 * @Author: Diantao Tu
 * @Date: 2022-07-18 15:50:39
 */

#include "Structure.h"

std::vector<PointTrack> TriangulateTracks(const std::vector<Frame>& frames, const std::vector<MatchPair>& image_pairs)
{
    vector<PointTrack> structure;
    // 找到所有的匹配的图像对以及他们之间的特征匹配关系，然后用这些关系建立一个track
    vector<pair<size_t, size_t>> pairs;
    vector<vector<cv::DMatch>> pair_matches;
    for(const MatchPair& p : image_pairs)
    {
        pairs.push_back(p.image_pair);
        pair_matches.push_back(p.matches);
    }
    TrackBuilder tracks_builder;
    tracks_builder.Build(pairs, pair_matches);
    tracks_builder.Filter(3);
    // {track_id, {image_id, feature_id},{image_id, feature_id},{image_id, feature_id}... }
    map<uint32_t, set<pair<uint32_t, uint32_t>>> tracks;
    tracks_builder.ExportTracks(tracks);
    LOG(INFO) << "Prepare to triangulate " << tracks.size() << " tracks";
    size_t max_track_id = tracks_builder.GetMaxID();
    if(tracks.empty())
    {
        LOG(ERROR) << "Fail to estimate initial structure";
        return structure;
    }
    // 释放内存
    pairs.clear();
    pair_matches.clear();

    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    #pragma omp parallel for
    for(size_t track_idx = 0; track_idx < max_track_id; track_idx++)
    {
        if(!tracks.count(track_idx))
            continue;
        map<uint32_t, set<pair<uint32_t, uint32_t>>>::iterator it_track = tracks.find(track_idx);
        eigen_vector<Eigen::Matrix3d> R_cw_list;
        eigen_vector<Eigen::Vector3d> t_cw_list;
        vector<cv::Point3f> points;
        for(const pair<uint32_t, uint32_t>& feature_pair : (*it_track).second)
        {
            const uint32_t image_idx = feature_pair.first;
            const Eigen::Matrix4d T_cw = frames[image_idx].GetPose().inverse();
            // const Eigen::Matrix4d T_cw = frames[image_idx].GetPose();
            R_cw_list.push_back(T_cw.block<3,3>(0,0));
            t_cw_list.push_back(T_cw.block<3,1>(0,3));
            points.push_back(eq.ImageToCam(frames[image_idx].GetKeyPoints()[feature_pair.second].pt));
        }
        Eigen::Vector3d point_world = TriangulateNView(R_cw_list, t_cw_list, points);
        if(isinf(point_world.x()) || isinf(point_world.y()) || isinf(point_world.z()))
            continue;
        #pragma omp critical
        {
            structure.push_back(PointTrack(it_track->first, it_track->second, point_world));
        }
    }
    // 误差超过25度的三角化的点直接过滤掉
    FilterTracksAngleResidual(frames, structure, 25);
    // if(!RemoveFarPoints(5))
    //     return false;
    LOG(INFO) << "Successfully triangulate " << structure.size() << " tracks";
    
    return structure;

    // 用于debug，显示每个track在各个图像上匹配的特征点，以及三角化后的点投影到图像上的位置
    // set<size_t> frame_ids = {162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172};
    // #pragma omp parallel for
    // for(size_t i = 0; i < structure.size(); i++)
    // {
    //     for(auto pair : structure[i].feature_pairs)
    //     {
    //         if(frame_ids.count(pair.first) > 0)
    //         {
    //             VisualizeTrack(structure[i], config.sfm_result_path);
    //             break;
    //         }
    //     }
    // }
}

size_t FilterTracksToFar(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    vector<PointTrack> valid_tracks;
    #pragma omp parallel for
    for(size_t i = 0; i < tracks.size(); i++)
    {
        // 找到当前三维点所对应的全部图像，然后找出其中相距最远的两个图像，把它们
        // 之间的距离当做 baseline （基线）
        set<size_t> frame_ids;
        for(const auto& pair : tracks[i].feature_pairs)
            frame_ids.insert(pair.first);
        eigen_vector<Eigen::Vector3d> frame_center;
        for(const size_t& id : frame_ids)
            if(frames[id].IsPoseValid())
                frame_center.push_back(frames[id].GetPose().block<3,1>(0,3));
        double baseline_distance = 0;
        FurthestPoints(frame_center, *(new int), *(new int), baseline_distance);
        // 计算当前三维点到各个图像之间的距离的平均值，如果这个距离超过了基线的 threshold 倍，就认为
        // 当前三维点不可靠，过滤掉
        double average_distance = 0;
        for(const Eigen::Vector3d& center : frame_center)
            average_distance += (center - tracks[i].point_3d).norm();
        average_distance /= frame_center.size();
        if(threshold * baseline_distance < average_distance)
            continue;
        #pragma omp critical
        {
            valid_tracks.push_back(tracks[i]);
        }
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}

size_t FilterTracksPixelResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    if(threshold < 0)
        return 0;
    double sq_threshold = Square(threshold);
    vector<PointTrack> valid_tracks;
    eigen_vector<Eigen::Matrix4d> T_cw_list;
    for(const Frame& frame : frames)
    {
        if(frame.IsPoseValid())
            T_cw_list.push_back(frame.GetPose().inverse());
        else 
            T_cw_list.push_back(Eigen::Matrix4d::Zero());
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(size_t i = 0; i < tracks.size(); i++)
    {
        bool valid = true;
        const Eigen::Vector3d pt_world = tracks[i].point_3d;
        for(const auto& feature : tracks[i].feature_pairs)
        {
            Eigen::Vector3d pt_cam = (T_cw_list[feature.first] * pt_world.homogeneous()).head(3);
            Eigen::Vector2d pt_proj = eq.CamToImage(pt_cam);
            const cv::Point2f& pt = frames[feature.first].GetKeyPoints()[feature.second].pt;
            double sq_dist = Square(pt.x - pt_proj.x()) + Square(pt.y - pt_proj.y());
            if(sq_dist > sq_threshold)
            {
                valid = false;
                break;
            }
        }
        if(valid)
            valid_tracks.push_back(tracks[i]);
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}

size_t FilterTracksAngleResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold)
{
    // 计算夹角的阈值对应的余弦值，减少后面的计算量
    double cos_threshold = cos(threshold * M_PI / 180.0);
    vector<PointTrack> valid_tracks;
    eigen_vector<Eigen::Matrix4d> T_cw_list;
    for(const Frame& frame : frames)
    {
        if(frame.IsPoseValid())
            T_cw_list.push_back(frame.GetPose().inverse());
        else 
            T_cw_list.push_back(Eigen::Matrix4d::Zero());
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(size_t i = 0; i < tracks.size(); i++)
    {
        bool valid = true;
        const Eigen::Vector3d pt_world = tracks[i].point_3d;
        for(const auto& feature : tracks[i].feature_pairs)
        {
            Eigen::Vector3d pt_cam = (T_cw_list[feature.first] * pt_world.homogeneous()).head(3);
            cv::Point3f view_ray = eq.ImageToCam(frames[feature.first].GetKeyPoints()[feature.second].pt);
            double cos_angle = (pt_cam.x() * view_ray.x + pt_cam.y() * view_ray.y +  pt_cam.z() * view_ray.z)
                            / pt_cam.norm() / cv::norm(view_ray);
            if(cos_angle < cos_threshold)
            {
                valid = false;
                break;
            }
        }
        if(valid)
            valid_tracks.push_back(tracks[i]);
    }
    valid_tracks.swap(tracks);
    return valid_tracks.size() - tracks.size();
}