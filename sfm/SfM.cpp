/*
 * @Author: Diantao Tu
 * @Date: 2021-11-22 16:44:10
 */

#include "SfM.h"
#include "../base/Geometry.hpp"

using namespace std;

SfM::SfM(const Config& _config):config(_config)
{
    track_triangulated = false;
}
bool SfM::ReadImages(const vector<string>& image_names, const cv::Mat& mask)
{
    LOG(INFO) << "=============== Read image and extract features begin ==============" << endl;
    omp_set_num_threads(config.num_threads);
    frames.clear();
    // 目前假设所有图像的分辨率都是相同的
    cv::Mat img = cv::imread(image_names[0]);
    ProcessBar bar(image_names.size(), 0.1);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < image_names.size(); i++)
    {
        Frame frame(img.rows, img.cols, i, image_names[i]);
        frame.LoadImageGray(frame.name);
        frame.ExtractKeyPoints(config.num_sift, mask);
        frame.ComputeDescriptor(config.root_sift);
        frame.ReleaseImageGray();
        #pragma omp critical 
        {
            frames.push_back(frame);
            bar.Add();
        }
    }
    // 对frame按id大小排列，因为经过openmp乱序执行
    sort(frames.begin(), frames.end(), [this](Frame& a, Frame& b){return a.id < b.id;});
    LOG(INFO) << "Read " << frames.size() << " images";
    LOG(INFO) << "=============== Read image and extract features end ==============" << endl;
    return frames.size() > 0;
}

bool SfM::LoadFrameBinary(const std::string& image_path, const std::string& frame_path, const bool skip_descriptor)
{
    return ReadFrames(frame_path, image_path, frames, config.num_threads, skip_descriptor);
}

bool SfM::InitImagePairs(const int frame_match_type)
{
    image_pairs.clear();
    if(frame_match_type & FrameMatchMethod::EXHAUSTIVE)
    {
        LOG(INFO) << "init match pairs with exhausive match";
        for(int i = 0; i < frames.size(); i++)
            for(int j = i + 1; j < frames.size(); j++)
                image_pairs.push_back(MatchPair(i,j));
        return image_pairs.size() > 0;
    }
    // 图像对的初始生成方式是可以组合的，也就是说可以同时使用VLAD和连续匹配，这种情况下也会有很多重复
    // 为了避免出现重复的匹配对，就需要用一个set来保存已有的匹配对的图像id
    set<pair<size_t,size_t>> pairs;
    if(frame_match_type & FrameMatchMethod::CONTIGUOUS)
    {
        int neighbor_size = 20;
        LOG(INFO) << "init match pairs with contiguous match, neighbor size = " << neighbor_size;
        for(int i = 0; i < frames.size(); i++)
            for(int j = i + 1; j < i + neighbor_size && j < frames.size(); j++)
            {
                image_pairs.push_back(MatchPair(i,j));
                pairs.insert({i, j});
            }
    }
    if(frame_match_type & FrameMatchMethod::VLAD)
    {
        int neighbor_size = max(int(frames.size() / 40), 15);
        LOG(INFO) << "init match pairs with VLAD, neighbor size = " << neighbor_size;
        VLADMatcher vlad(frames, config, RESIDUAL_NORMALIZATION_PWR_LAW);
        vlad.GenerateCodeBook(0.5);
        vlad.ComputeVLADEmbedding();
        std::vector<std::vector<size_t>> neighbors_all = vlad.FindNeighbors(neighbor_size);
        for(size_t i = 0; i < frames.size(); i++)
        {
            for(const size_t& neighbor : neighbors_all[i])
            {
                if(neighbor == i)
                    continue;
                size_t min_id = min(i, neighbor);
                size_t max_id = max(i, neighbor);
                if(pairs.count({min_id, max_id}) == 0)
                {
                    image_pairs.push_back(MatchPair(min_id, max_id));
                    pairs.insert({min_id, max_id});
                }
            }
        }
    }
    if(frame_match_type & FrameMatchMethod::GPS)
    {
        int neighbor_size = 15;
        float distance_threshold = 7;
        LOG(INFO) << "init match pairs with GPS, neighbor size = " << neighbor_size << ", distance threshold = " << distance_threshold;
        if(LoadGPS(config.gps_path))
        {
            // 记录下每个图像的位置, 为了之后进行近邻搜索
            pcl::PointCloud<pcl::PointXYZI> frame_center;
            for(size_t i = 0; i < frames.size(); i++)
            {
                pcl::PointXYZI pt(i);
                EigenVec2PclPoint(frames[i].GetGPS(), pt);
                frame_center.push_back(pt);
            }
            pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_center(new pcl::KdTreeFLANN<pcl::PointXYZI>());
            kd_center->setInputCloud(frame_center.makeShared());
            for(const pcl::PointXYZI& pt : frame_center)
            {
                vector<float> sq_neighbor_dist;
                vector<int> neighbor_ids;
                kd_center->nearestKSearch(pt, neighbor_size + 1, neighbor_ids, sq_neighbor_dist);
                for(size_t i = 1; i < neighbor_ids.size(); i++)
                {
                    if(sq_neighbor_dist[i] > Square(distance_threshold))
                        break;
                    pair<size_t,size_t> curr_pair(min(pt.intensity, frame_center[neighbor_ids[i]].intensity), max(pt.intensity, frame_center[neighbor_ids[i]].intensity));
                    if(pairs.count(curr_pair) > 0)
                        continue;
                    pairs.insert(curr_pair);
                    image_pairs.push_back(MatchPair((size_t)pt.intensity, (size_t)frame_center[neighbor_ids[i]].intensity));
                }
            }
        }
        else 
            LOG(ERROR) << "Unable to init match pairs with GPS";
    }
    if(frame_match_type & FrameMatchMethod::GPS_VLAD)
    {
        int neighbor_size = frames.size() / 40;
        float distance_threshold = 20;
        LOG(INFO) << "init match pairs with VLAD and filter with GPS, neighbor size = " << neighbor_size << ", distance threshold = " << distance_threshold;;
        if(LoadGPS(config.gps_path))
        {
            VLADMatcher vlad(frames, config, RESIDUAL_NORMALIZATION_PWR_LAW);
            vlad.GenerateCodeBook(0.5);
            vlad.ComputeVLADEmbedding();
            std::vector<std::vector<size_t>> neighbors_all = vlad.FindNeighbors(neighbor_size);
            for(size_t i = 0; i < frames.size(); i++)
            {
                for(const size_t& neighbor : neighbors_all[i])
                {
                    if(neighbor == i)
                        continue;
                    size_t min_id = min(i, neighbor);
                    size_t max_id = max(i, neighbor);
                    double gps_distance = (frames[min_id].GetGPS() - frames[max_id].GetGPS()).norm();
                    // GPS 距离大于阈值，过滤掉
                    if(gps_distance > distance_threshold)
                        continue;
                    if(pairs.count({min_id, max_id}) == 0)
                    {
                        image_pairs.push_back(MatchPair(min_id, max_id));
                        pairs.insert({min_id, max_id});
                    }
                }
            }
        }
    }
    return image_pairs.size() > 0;
}

bool SfM::ComputeDepthImage(const Eigen::Matrix4d T_cl)
{
    LOG(INFO) << "=================== Compute Depth Image begin =====================";
    // 目前假设图像和雷达的数目是一样的，且初始的外参基本正确
    if(lidars.size() != frames.size())
    {
        LOG(ERROR) << "warning: lidar size != frame size" << endl;
        return false;
    }
    bool save_depth = !config.depth_path.empty();
    if(save_depth && !boost::filesystem::exists(config.depth_path))
    {
        boost::filesystem::create_directory(config.depth_path);
        LOG(INFO) << "save depth image in " << config.depth_path;
    }

    string visualize_path = config.sfm_result_path + "/depth_visualize";
    if(!boost::filesystem::exists(visualize_path))
        boost::filesystem::create_directory(visualize_path);

    bool half_size = true;
    omp_set_num_threads(config.num_threads);
    ProcessBar bar(lidars.size(), 0.1);
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 从所有雷达数据中复制一份，这样读取完点云后直接就释放了，不会占内存
        Velodyne lidar(lidars[i]);
        lidar.LoadLidar(lidar.name);
        cv::Mat depth;
        // 深度图只用一半的分辨率即可，可以大大节约内存
        if(half_size)
            depth = ProjectLidar2PanoramaDepth(lidar.cloud, (frames[i].GetImageRows() + 1) / 2, 
                            (frames[i].GetImageCols() + 1) / 2, T_cl, 4);
        else 
            depth = ProjectLidar2PanoramaDepth(lidar.cloud, frames[i].GetImageRows(), 
                            frames[i].GetImageCols(), T_cl, 4);
        depth = DepthCompletion(depth, config.max_depth);       // 深度补全后是CV_32F的真实深度
        if(save_depth)
        {
            // 把深度图和彩色图结合起来，得到更好的可视化效果
            cv::Mat depth_with_color = frames[i].GetImageColor();
            if(half_size)
                cv::pyrDown(depth_with_color, depth_with_color);
            depth_with_color = CombineDepthWithRGB(depth, depth_with_color, config.max_depth_visual);
            cv::imwrite(visualize_path + "/depth_" + num2str(i) + ".jpg", depth_with_color);
            // 变成CV_16U的形式，虽然会损失一点精度，但是大大节约了内存
            depth *= 256.f;
            depth.convertTo(depth, CV_16U);                         
            // 把深度图保存到本地
            ExportOpenCVMat(config.depth_path + num2str(i) + ".bin", depth);
        }
        bar.Add();
    }
    LOG(INFO) << "=================== Compute Depth Image end ===================";
    return true;
}


bool SfM::MatchImagePairs(const int matches_threshold)
{
    LOG(INFO) << "========= Match Image Pairs begin ===========" << endl;
    vector<MatchPair> good_pair;
    set<size_t> covered_frames;     // 记录一下被image pair覆盖的frame个数，没有实际作用，只是用来输出一下
    omp_set_num_threads(config.num_threads);
#ifdef USE_CUDA
    bool use_cuda = config.use_cuda && (cv::cuda::getCudaEnabledDeviceCount() > 0);
    vector<cv::cuda::GpuMat> d_descriptors(frames.size());
    if(use_cuda)
    {
        #pragma omp parallel for
        for(size_t i = 0; i < frames.size(); i++)
            d_descriptors[i].upload(frames[i].GetDescriptor());
    }
#else 
    bool use_cuda = false;
#endif
    // 第一步：通过SIFT特征匹配，过滤掉不合适的匹配关系
    LOG(INFO) << "match image pair with SIFT, " << (use_cuda ? "use cuda" : "use cpu"); 
    LOG(INFO) << "init pair : " << image_pairs.size() << endl;
    
    ProcessBar bar1(image_pairs.size(), 0.1);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < image_pairs.size(); i++)
    {
        bar1.Add(1);
        size_t source = image_pairs[i].image_pair.first;
        size_t target = image_pairs[i].image_pair.second;
        vector<cv::DMatch> matches;
#ifdef USE_CUDA
        if(use_cuda)
            matches = MatchSIFT(d_descriptors[source], d_descriptors[target], config.sift_match_dist_threshold);
        else 
#endif
            matches = MatchSIFT(frames[source].GetDescriptor(), frames[target].GetDescriptor(), config.sift_match_dist_threshold);
                                            
        if(matches.size() < matches_threshold)
            continue;
        vector<cv::DMatch>::iterator it_max = max_element(matches.begin(), matches.end());
        vector<cv::DMatch> good_matches;
        // 用匹配距离再去筛选一遍，得到更准确的匹配效果
        for(vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end(); it++)
            if(it->distance < 0.8 * it_max->distance)
                good_matches.push_back(*it);
        if(int(good_matches.size()) < matches_threshold)
            continue;
        #pragma omp critical
        {
            good_pair.push_back(MatchPair(source, target, good_matches));
            covered_frames.insert(source);
            covered_frames.insert(target);
            // 画出匹配关系
            // cv::Mat matches_vertical = DrawMatchesVertical(frames[source].GetImageColor(), frames[source].GetKeyPoints(),
            //                         frames[target].GetImageColor(), frames[target].GetKeyPoints(), good_matches);
            // cv::imwrite(config.sfm_result_path + "/sift_" + num2str(source) + "_" + num2str(target) + ".jpg", matches_vertical);
        }
    }
    good_pair.swap(image_pairs);
    LOG(INFO) << "after filter with SIFT : " << image_pairs.size() << " image pairs, with " <<
             covered_frames.size() << " frames" << endl;
    ExportMatchPairTXT(config.sfm_result_path + "/after_sift_match.txt");
    for(size_t i = 0; i < frames.size(); i++)
        frames[i].ReleaseDescriptor();
    LOG(INFO) << "========= Match Image Pairs end ===========" << endl;
    return true;
}


bool SfM::FilterImagePairs(const int triangulation_num_threshold , const float triangulation_angle_threshold, const bool keep_no_scale )
{
    LOG(INFO) << "=========== Filter Image Pairs begin ================";
    omp_set_num_threads(config.num_threads);
    set<size_t> covered_frames;
    size_t num_pairs_init = image_pairs.size(); 
    // 这几个数是用来统计匹配的图像对分别是由于什么缘故被过滤掉的，可以辅助分析一下SfM表现不好的原因
    size_t estimate_essential_fail = 0 ,            // 无法估计本质矩阵
            decompose_essential_fail = 0,           // 无法分解本质矩阵得到好的相对位姿
            refine_pose_fail = 0,                   // 优化相对位姿时失败
            no_scale = 0,                           // 无法计算尺度
            no_connection = 0;                      // 不满足边双联通 bi-edge-connection

    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    // 把所有二维特征点投影到单位球上
    vector<vector<cv::Point3f>> key_points_sphere(frames.size());
    for(size_t i = 0; i < frames.size(); i++)
    {
        const vector<cv::KeyPoint>& key_points = frames[i].GetKeyPoints();
        for(auto& kp : key_points)
            key_points_sphere[i].push_back(eq.ImageToCam(kp.pt));
    }
    vector<MatchPair> good_pair;
    #pragma omp parallel for schedule(dynamic)
    for(const MatchPair& p : image_pairs)
    {
        size_t idx1 = p.image_pair.first;
        size_t idx2 = p.image_pair.second;
        const vector<cv::Point3f>& keypoints1 = key_points_sphere[idx1];
        const vector<cv::Point3f>& keypoints2 = key_points_sphere[idx2];
        vector<MatchPair> pair_each_iter;
        vector<pair<int,int>> inlier_each_pair;
        for(int iter = 0; iter < 40; iter++)
        {
            // 计算本质矩阵E
            Eigen::Matrix3d essential;
            vector<size_t> inlier_idx;
            // 使用AC-RANSAC计算
            double error_upper_bound = 5.0 * M_PI / 180.0;      // 设置误差的上限为5度
            ACRansac_NFA nfa(p.matches.size(), 8, false);
            essential = FindEssentialACRANSAC(p.matches, keypoints1, keypoints2, 300, 
                        error_upper_bound, nfa, inlier_idx);
            if(essential.isZero())
                continue;
            
            // 本质矩阵分解得到4组R t
            eigen_vector<Eigen::Matrix3d> rotations;
            eigen_vector<Eigen::Vector3d> trans;
            DecomposeEssential(essential, rotations, trans);

            // 找到4组R t 中最正确的那个
            vector<double> parallax(rotations.size());
            vector<int> num_pts(rotations.size(), 0);
            vector<eigen_vector<Eigen::Vector3d>> triangulated_points(rotations.size());
            vector<vector<size_t>> inliers(rotations.size());
            // 只使用计算本质矩阵的内点去进行三角化来检测R t的正确性
            vector<bool> inlier(p.matches.size(), false);
            for(const size_t& idx : inlier_idx)
                inlier[idx] = true;
            for(size_t j = 0; j < rotations.size(); j++)
                num_pts[j] = CheckRT(rotations[j], trans[j], inlier, p.matches, 
                            keypoints1, keypoints2, parallax[j], triangulated_points[j], inliers[j]);
            
            vector<int>::iterator max_points_idx = max_element(num_pts.begin(), num_pts.end());
            int best_idx = max_points_idx - num_pts.begin();
            // 三角化的点数量太少，过滤掉
            if(*max_points_idx < triangulation_num_threshold)
            {
                continue;
            }
            // 视差太小，过滤掉
            // if(parallax[best_idx] < 1)
            // {
            //     little_parallax++;
            //     continue;
            // }
            int num_similar = count_if(num_pts.begin(), num_pts.end(), 
                            [max_points_idx](unsigned int num_points){return num_points > 0.8 * (*max_points_idx);});
            // 有多个位姿都能三角化出大量三维点，过滤掉
            if(num_similar > 1)
            {
                continue;
            }

            // 注意这个里面没有保存匹配的特征点对，因为对于每次迭代的结果来说，匹配的特征点对都是相同的
            MatchPair match_pair(p.image_pair.first, p.image_pair.second);
            match_pair.R_21 = rotations[best_idx];
            match_pair.t_21 = trans[best_idx];
            match_pair.triangulated = triangulated_points[best_idx];
            match_pair.inlier_idx = inliers[best_idx];

            pair_each_iter.push_back(match_pair);
            inlier_each_pair.push_back({match_pair.inlier_idx.size(), inlier_each_pair.size()});            
        }
        if(pair_each_iter.empty())
        {
            estimate_essential_fail++;
            continue;
        }

        #if 0
        // 输出一下每次迭代的结果，用于debug
        ofstream f(config.sfm_result_path +  num2str(p.image_pair.first) + "_" + num2str(p.image_pair.second) + ".txt");
        for(const MatchPair& pair : pair_each_iter)
        {
            Eigen::Vector3d t_12 = -pair.R_21.transpose() * pair.t_21;
            f << "inliner : " <<  pair.inlier_idx.size() << endl;
            f << t_12.x() << " " << t_12.y() << " " << t_12.z() << endl;
            Eigen::AngleAxisd angleAxis(pair.R_21.transpose());
            f << angleAxis.axis().x() << " " << angleAxis.axis().y() << " " << angleAxis.axis().z() << " " << angleAxis.angle() * 180.0 / M_PI << endl;
        }
        #endif 

        // 对每次的结果排序，按照内点数量从大到小排列
        // first - 内点数量， second - 索引
        sort(inlier_each_pair.begin(), inlier_each_pair.end(), 
            [](const pair<int,int>& a, const pair<int,int>& b){return a.first > b.first;});

        int best_idx = inlier_each_pair[0].second;
        #if 0
        // 限制相邻帧之间旋转的角度，最大角度 = 帧数差异 * 1.5度，也就是说限制连续两帧之间旋转角度不能超过1.5度
        // 这个比较适用于车载的场景
        double rotation_angle_threshold = (p.image_pair.second - p.image_pair.first) * 1.5 / 180.0 * M_PI;
        for(const auto& idx : inlier_each_pair)
        {
            best_idx = idx.second;
            Eigen::AngleAxisd angleAxis(pair_each_iter[best_idx].R_21);
            if(angleAxis.angle() <= rotation_angle_threshold)
                break;
            best_idx = -1;
        }
        if(best_idx < 0)
        {
            decompose_essential_fail ++;
            continue;
        }
        #endif 
        pair_each_iter[best_idx].matches = p.matches;
        if(!RefineRelativePose(pair_each_iter[best_idx]))
        {
            refine_pose_fail ++;
            // continue;
        }
        #pragma omp critical
        {
            good_pair.push_back(pair_each_iter[best_idx]);
        }
    }

    image_pairs = good_pair;
    
    /* 计算相对平移的尺度 */
    LOG(INFO) << "start to set relative translation scale";
    SetTranslationScaleDepthMap(keep_no_scale);
    no_scale = good_pair.size() - image_pairs.size();
    size_t tmp = image_pairs.size();
    // 根据边双连通过滤
    image_pairs = LargestBiconnectedGraph(image_pairs, covered_frames);
    no_connection = tmp - image_pairs.size();
    LOG(INFO) << "count of image pairs with valid relative motion : " << image_pairs.size() << " image pairs, with " <<
             covered_frames.size() << " frames" << endl;
    LOG(INFO) << "filter " << num_pairs_init - image_pairs.size() << " image pairs" << 
                "\r\n\t\t filter by compute essential fail : " << estimate_essential_fail << 
                "\r\n\t\t filter by decompose essential fail : " << decompose_essential_fail << 
                "\r\n\t\t filter by refine pose fail : " << refine_pose_fail <<
                "\r\n\t\t filter by no scale : " << no_scale << 
                "\r\n\t\t filter by not bi-connected : " << no_connection; 
    // 经过上面openmp的并行操作后，image_pair的顺序就被打乱了，重新按图像的索引排序，排列成
    // 0-1  0-2  0-3  0-4 ... 1-2  1-3  1-4 ... 2-3  2-4  ... 3-4 这样的顺序
    // 这里不排序也是可以的，只是为了后面debug方便才排序的
    sort(image_pairs.begin(), image_pairs.end(), 
        [this](const MatchPair& mp1,const MatchPair& mp2)
        {
            if(mp1.image_pair.first < mp2.image_pair.first)
                return true;
            else 
                return mp1.image_pair.second < mp2.image_pair.second;
        }
        );

    LOG(INFO) << "=========== Filter Image Pairs end ================";
    return true;
}

bool SfM::RefineRelativePose(MatchPair& image_pair)
{
    return SfMLocalBA(frames[image_pair.image_pair.first], frames[image_pair.image_pair.second], PIXEL_RESIDUAL, image_pair);
}

bool SfM::SetTranslationScaleDepthMap(const Equirectangular& eq, MatchPair& pair)
{
    size_t idx1 = pair.image_pair.first;
    size_t idx2 = pair.image_pair.second;
    const cv::Mat& depth_image1 = frames[idx1].depth_map;
    const cv::Mat& depth_image2 = frames[idx2].depth_map;
    if(depth_image1.empty() || depth_image2.empty())
        return false;
    // 判断一下深度图是否为半尺寸的，如果是半尺寸的后面投影的结果都要除以2
    bool half_size = (depth_image1.rows == int((frames[idx1].GetImageRows() + 1 ) / 2));
    pair.points_with_depth = 0;
    vector<double> scale;
    for(const Eigen::Vector3d& p : pair.triangulated)
    {
        // 把当前点投影到图像1下，计算投影深度和真实深度的比值
        Eigen::Vector2d point_project1 = eq.CamToImage(p) / (1.0 + half_size);
        int row = round(point_project1.y()), col = round(point_project1.x());
        if(!eq.IsInside(cv::Point2i(col, row)))
            continue;
        double depth1 = p.norm();
        const float depth1_real = depth_image1.at<uint16_t>(row, col) / 256.0;
        if(depth1_real <= 0)
            continue;
        double scale1 = depth1_real / depth1;
        // 把当前点投影到图像2下，计算投影深度和真实深度的比值
        const Eigen::Vector3d point_in_frame2 = pair.R_21 * p + pair.t_21;
        Eigen::Vector2d point_project2 = eq.CamToImage(point_in_frame2)  / (1.0 + half_size);
        row = round(point_project2.y());
        col = round(point_project2.x());
        if(!eq.IsInside(cv::Point2i(col, row)))
            continue;

        double depth2 = point_in_frame2.norm();
        const float depth2_real = depth_image2.at<uint16_t>(row, col) / 256.0;
        if(depth2_real <= 0)
            continue;
        double scale2 = depth2_real / depth2;
        // 如果算出来的两个尺度差异太大，就认为不可靠，除去
        if(abs(scale1 - scale2) / min(scale1, scale2) > 0.2)
            continue;
        scale.push_back(scale1);
        scale.push_back(scale2);
    }
    if(scale.size() < 10)
        return false;
    // 对计算得到的scale进行一定的过滤，去掉不准确的scale，保留稳定的scale，并用这些稳定的scale算出最终的scale
    // 具体方法就是把scale分成直方图，只保留直方图中占比较高的几个bin，因为直方图的上限和下限都是根据所有scale中的
    // 最大值和最小值来计算的，所以如果有某些点的scale特别大或特别小，就会导致绝大部分其他点的scale都集中在某几个bin中，
    // 那么这些特别“离谱”的scale就很容易通过直方图剔除掉。这整体是一个迭代的过程，迭代的次数越多，那么scale的分布也就越
    // 集中，算出的scale也就越准确。但是相应的，保留下来的scale也会很少
    // 这种方法确实能得到比较稳定的尺度，但是会导致很多图像对没有尺度信息，因为过滤的太狠了，所以我又选了一个更简单的方法，
    // 也就是把当前图像对的尺度排序，选择中间的那个值作为尺度，但这种方法肯定没有上一种好。因此做了一个判断，如果没法用
    // 好方法得到尺度，那么就用这个差方法
    bool scale_is_good = true;
    vector<double> scale_preserve(scale);
    const size_t num_bins = 10;
    for(size_t iter = 0; iter < 2; iter++)
    {
        size_t num_scale = scale.size();
        if(num_scale < 10)
        {
            // LOG(INFO) << "Not enough scale factor for image pair " << pair.image_pair.first 
            //             << " - " << pair.image_pair.second << endl;
            scale_is_good = false;
            break;
        }
        double max_scale = *(max_element(scale.begin(), scale.end()));
        double min_scale = *(min_element(scale.begin(), scale.end()));
        if(max_scale / min_scale < 1.2)
            break;
        double interval = (max_scale - min_scale) / num_bins;
        // 在计算直方图的时候，最大的那个scale会超出直方图的范围，
        // 解决方法是在所有的scale上进行一点小小的偏移，
        // 比如把所有的数都减去0.0000001，这样就不会造成这种问题了。
        vector<vector<double>> histo(num_bins);
        for(const double& s : scale)
        {
            // 在很少的情况下，如果真的scale都特别小，那么可能就会出现bin_idx越界问题，为了避免这个问题，
            // 就用min 和 max来限制bin_idx的范围
            int bin_idx = int((s - min_scale - 1e-8) / interval);
            bin_idx = min(bin_idx, int(num_bins - 1));
            bin_idx = max(0, bin_idx); 
            histo[bin_idx].push_back(s);
        }
        scale.clear();
        for(const vector<double>& bin: histo)
        {
            if(bin.size() > 0.1 * num_scale)
                scale.insert(scale.end(), bin.begin(), bin.end());
        }
    }
    
    double final_scale = 0;
    if(scale_is_good)
    {
        for(const double& s : scale)
            final_scale += s;
        final_scale /= scale.size();
        pair.t_21 *= final_scale;
        pair.points_with_depth = scale.size() / 2;
        pair.upper_scale = *(max_element(scale.begin(), scale.end()));
        pair.lower_scale = *(min_element(scale.begin(), scale.end()));
    }
    else 
    {
        nth_element(scale_preserve.begin(), scale_preserve.begin() + scale_preserve.size() / 2, scale_preserve.end());
        final_scale = scale_preserve[scale_preserve.size() / 2];
        pair.t_21 *= final_scale;
        pair.upper_scale = 0;
        pair.lower_scale = 0;
        pair.points_with_depth = scale_preserve.size() / 2;
    }
    // 设置了尺度后，三角化的点也要乘以相应的尺度
    for(size_t i = 0; i < pair.triangulated.size(); i++)
        pair.triangulated[i] *= final_scale;
    return true;
}

bool SfM::SetTranslationScaleDepthMap(const bool keep_no_scale)
{
    // 记录对每张图像的深度图的引用数量，一旦数量降到零就代表可以释放当前图像的深度图了
    vector<size_t> depth_ref_count(frames.size(), 0);
    for(const MatchPair& p : image_pairs)
    {
        depth_ref_count[p.image_pair.first] ++;
        depth_ref_count[p.image_pair.second] ++;
    }

    // 互斥量，用于读取深度图和释放深度图
    vector<mutex> depth_mutex(frames.size());
    vector<string> depth_image_names = IterateFiles(config.depth_path, ".bin");
    // 找到引用数量最少的那个图像，也就是说从这张图像开始处理
    size_t start_idx = min_element(depth_ref_count.begin(), depth_ref_count.end()) - depth_ref_count.begin();
    vector<size_t> process_order;
    for(int idx = start_idx; idx < frames.size(); idx++)
        process_order.push_back(idx);
    for(int idx = 0; idx < start_idx; idx++)
        process_order.push_back(idx);
    vector<MatchPair> good_pair;
    set<pair<size_t,size_t>> pairs_processed;
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(const size_t& idx1 : process_order)
    {
        #pragma omp parallel for
        for(MatchPair& p : image_pairs)
        {
            if(pairs_processed.count(p.image_pair) > 0)
                continue;
            size_t idx2;
            if(p.image_pair.first == idx1)
                idx2 = p.image_pair.second;
            else if(p.image_pair.second == idx1)
                idx2 = p.image_pair.first;
            else 
                continue;
            
            // 读取两个frame对应的深度图
            {
                lock_guard<mutex> guard(depth_mutex[idx1]);
                if(frames[idx1].depth_map.empty())
                    ReadOpenCVMat(depth_image_names[idx1], frames[idx1].depth_map);
            }
            
            {
                lock_guard<mutex> guard(depth_mutex[idx2]);
                if(frames[idx2].depth_map.empty())
                    ReadOpenCVMat(depth_image_names[idx2], frames[idx2].depth_map);
            }
            bool valid = true;
            // 设置相对位姿的尺度,如果keep_no_scale=true那么即使当前图像对没有尺度也会保留下来
            if(!SetTranslationScaleDepthMap(eq, p) && !keep_no_scale)
            {
                valid = false;
            }
            #pragma omp critical
            {
                depth_ref_count[idx1]--;
                depth_ref_count[idx2]--;
                if(depth_ref_count[idx1] == 0)
                    frames[idx1].depth_map.release();
                if(depth_ref_count[idx2] == 0)
                    frames[idx2].depth_map.release();
                pairs_processed.insert(p.image_pair);
                if(valid)
                    good_pair.push_back(p);
            }
        }
    }
    good_pair.swap(image_pairs);
    for(Frame& f : frames)
        f.depth_map.release();
    return image_pairs.size() > 0;
}

bool SfM::SetTranslationScaleGPS(const std::string& gps_file, bool overwrite)
{
    if(!LoadGPS(gps_file)) 
        return false;
    for(MatchPair& p : image_pairs)
    {
        if(!overwrite && p.lower_scale >= 0 && p.upper_scale >= 0)
            continue;
        if(!frames[p.image_pair.first].IsGPSValid() || !frames[p.image_pair.second].IsGPSValid())
            continue;
        double scale_gps = (frames[p.image_pair.first].GetGPS() - frames[p.image_pair.second].GetGPS()).norm();
        double scale_pair = p.t_21.norm();
        double ratio = scale_gps / scale_pair;
        p.t_21 *= ratio;
        for(Eigen::Vector3d& point : p.triangulated)
            point *= ratio;
        p.lower_scale = (p.lower_scale > 0 ? p.lower_scale * ratio : 0);
        p.upper_scale = (p.upper_scale > 0 ? p.upper_scale * ratio : 0);
    }
    LOG(INFO) << "Set translation scale using GPS";
    return true;
}


std::vector<MatchPair> SfM::FilterByTriplet(const std::vector<MatchPair>& init_pairs, const double angle_threshold, std::set<size_t>& covered_frames)
{
    LOG(INFO) << "angle threshold for triplet filter: " << angle_threshold;
    covered_frames.clear();
    // 检测Triplet
    set<pair<size_t, size_t>> pairs;
    for(const MatchPair& p : init_pairs)
    {
        pairs.insert(p.image_pair);
        covered_frames.insert(p.image_pair.first);
        covered_frames.insert(p.image_pair.second);
    }
    int num_frames_before_filter = static_cast<int>(covered_frames.size());
    vector<Triplet> triplets = PoseGraph::FindTriplet(pairs);
    // 建立映射关系，可以方便的从图像对找到对应的旋转
    map<pair<size_t, size_t>, Eigen::Matrix3d> map_rotations;
    for(const MatchPair& p : init_pairs)
    {
        assert(p.image_pair.first < p.image_pair.second);
        map_rotations[p.image_pair] = p.R_21;
    }
    // 对初始生成的triplet进行过滤，角度误差超过一定阈值的都过滤掉
    vector<Triplet> valid_triplets;
    Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    for(size_t i = 0; i < triplets.size(); i++)
    {
        const Triplet& triplet = triplets[i];
        // 这里的 idx1 idx2 idx3 是依次增大的
        uint32_t idx1 = triplet.i;
        uint32_t idx2 = triplet.j;
        uint32_t idx3 = triplet.k;
        Eigen::Matrix3d R_21 = map_rotations[pair<size_t, size_t>(idx1, idx2)];
        Eigen::Matrix3d R_32 = map_rotations[pair<size_t, size_t>(idx2, idx3)];
        Eigen::Matrix3d R_13 = map_rotations[pair<size_t, size_t>(idx1, idx3)].transpose();
        Eigen::Matrix3d rot = R_13 * R_32 * R_21;
        double cos_theta = (identity.array() * rot.array()).sum() / 3.0;
        cos_theta = min(1.0, max(cos_theta, -1.0));
        double angle_error = acos(cos_theta) * 180.0 / M_PI;
        if(angle_error < angle_threshold)
            valid_triplets.push_back(triplet);
    }
    LOG(INFO) << "Triplets before filter : " << triplets.size() <<  ", after filter : " << valid_triplets.size();
    covered_frames.clear();
    pairs.clear();
    for(const Triplet& t : valid_triplets)
    {
        pairs.insert(pair<size_t, size_t>(t.i, t.j));
        pairs.insert(pair<size_t, size_t>(t.j, t.k));
        pairs.insert(pair<size_t, size_t>(t.i, t.k));
        covered_frames.insert(t.i);
        covered_frames.insert(t.j);
        covered_frames.insert(t.k);
    }
    LOG(INFO) << "Valid frames before triplet filter : " << num_frames_before_filter << ", after triplet filter: " << covered_frames.size();

    {
        vector<size_t> invalid_frames;
        for(size_t i = 0; i < frames.size(); i++)
            if(covered_frames.count(i) <= 0)
                invalid_frames.push_back(i);
        LOG(INFO) << "invalid frames : " << Join(invalid_frames);
    }

    vector<MatchPair> pairs_after_triplet_filter;
    for(const MatchPair& m : init_pairs)
    {
        if(pairs.count(m.image_pair) > 0)
            pairs_after_triplet_filter.push_back(m);
    }
    // 得到好的Triplet后还要重新建立一次pose graph并过滤
    pairs_after_triplet_filter = LargestBiconnectedGraph(pairs_after_triplet_filter, covered_frames);
    LOG(INFO) << "relative motion after pose graph filter : " << pairs_after_triplet_filter.size() << ", with " << covered_frames.size() << " frames";
    return pairs_after_triplet_filter;
}

std::vector<MatchPair> SfM::LargestBiconnectedGraph(const std::vector<MatchPair>& pairs, std::set<size_t>& nodes)
{
    set<pair<size_t, size_t>> edges;
    for(const MatchPair& p : pairs)
        edges.emplace(p.image_pair);
    PoseGraph graph(edges);
    nodes.clear();
    nodes = graph.KeepLargestEdgeBiconnected();
    if(nodes.empty())
        return vector<MatchPair>();
    
    vector<MatchPair> good_pair;
    for(const MatchPair& p : pairs)
    {
        if(nodes.count(p.image_pair.first) > 0 &&
            nodes.count(p.image_pair.second) > 0)
            good_pair.push_back(p);
    }
    return good_pair;
}

bool SfM::RemoveFarPoints(double scale)
{
    if(structure.empty())
        return false;
    size_t num_filter = FilterTracksToFar(frames, structure, scale);
    LOG(INFO) << "Filter " << num_filter << " tracks with base-line, " << structure.size() << " points left";
    return true;
}


bool SfM::EstimateGlobalRotation(const int method)
{
    LOG(INFO) << "================ Estimate Global Rotation begin ================";
    // if(!config.gps_path.empty())
    //     SetTranslationScaleGPS(config.gps_path, true);
    
    // FilterByStraightMotion(100);
    
    // 如果只使用有尺度的图像对进行旋转平均，那么就只保留有尺度的图像对
    if(!config.use_all_pairs_ra)
    {
        vector<MatchPair> pairs_with_scale;
        for(const MatchPair& p : image_pairs)
        {
            if(p.upper_scale >= 0 && p.lower_scale >= 0)
                pairs_with_scale.push_back(p);
        }
        pairs_with_scale.swap(image_pairs);
        LOG(INFO) << "Only use pairs with scale to estimate global rotation, " << image_pairs.size() << " pairs";
    }
    
    // 输出相对位姿到文件中，用于debug
    // PrintRelativePose(config.sfm_result_path + "relpose-RA-before-filter.txt");

    set<size_t> covered_frames;
    image_pairs = LargestBiconnectedGraph(image_pairs, covered_frames);
    LOG(INFO) << "after filter with graph connection: " << image_pairs.size() << " pairs, " << covered_frames.size() << " frames";

    image_pairs = FilterByTriplet(image_pairs, 0.1, covered_frames);

    // PrintRelativePose(config.sfm_result_path + "relpose-RA.txt");


    // 从以下就是正式开始旋转平均，之前的都是进行一些前期准备
    LOG(INFO) << "Global rotation estimation:\n" << "\t\tprepare to estimate " << covered_frames.size() << 
            " global rotations, with " << image_pairs.size() << " relative motion";

    // 图像对的id进行重新映射
    map<size_t, size_t> old_to_new, new_to_old;
    ReIndex(image_pairs ,old_to_new, new_to_old);
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = old_to_new[pair.image_pair.first];
        pair.image_pair.second = old_to_new[pair.image_pair.second];
    }
    // 计算全局的旋转
    eigen_vector<Eigen::Matrix3d> global_rotations(old_to_new.size());
    bool success;
    if(method == ROTATION_AVERAGING_L2)
    {
        LOG(INFO) << "Rotation averaging L2 begin";
        if(!RotationAveragingLeastSquare(image_pairs, global_rotations))
        {
            LOG(ERROR) << "Rotation averaging L2 failed";
            return false;
        }
        if(!RotationAveragingL2(config.num_threads, image_pairs, global_rotations))
            LOG(ERROR) << "Rotation averaging refine L2 failed";
    }
    else if (method == ROTATION_AVERAGING_L1)
    {
        LOG(INFO) << "Rotation averaging L1 begin";
        if(!RotationAveragingL1(image_pairs, global_rotations, 0, -1))
        {
            LOG(ERROR) << "Rotation averaging L1 failed";
            return false;
        }
        
        #if 0
        // global rotation 算出来的是 R_cw, frame里保存的是 R_wc
        for(size_t i = 0; i < global_rotations.size(); i++)
            frames[new_to_old[i]].SetRotation(global_rotations[i].transpose());
        PrintGlobalPose(config.sfm_result_path + "frame_pose-after-L1.txt");
        #endif

        if(!RotationAveragingL2(config.num_threads, image_pairs, global_rotations))
            LOG(ERROR) << "Rotation averaging refine L2 failed";
    }
    // 用全局旋转更新图像对之间的相对旋转，因为后面的平移平均可能会用到相对旋转
    for(MatchPair& pair : image_pairs)
    {
        const Eigen::Matrix3d& R_1w = global_rotations[pair.image_pair.first];
        const Eigen::Matrix3d& R_2w = global_rotations[pair.image_pair.second];
        pair.R_21 = R_2w * R_1w.transpose();
        pair.image_pair.first = new_to_old[pair.image_pair.first];
        pair.image_pair.second = new_to_old[pair.image_pair.second];
    }
    for(size_t i = 0; i < global_rotations.size(); i++)
    {
        // global rotation 算出来的是 R_cw, frame里保存的是 R_wc
        frames[new_to_old[i]].SetRotation(global_rotations[i].transpose());
    }
    LOG(INFO) << "===================== Estimate Global Rotation end ===============";
    return true;
}


void SfM::ReIndex(const std::vector<MatchPair>& pairs, std::map<size_t, size_t>& forward, std::map<size_t, size_t>& backward)
{
    forward.clear();
    backward.clear();
    set<size_t> new_id;
    set<size_t> old_id;
    for(const MatchPair& pair : pairs)
    {
        old_id.insert(pair.image_pair.first);
        old_id.insert(pair.image_pair.second);
    }
    for(const MatchPair& pair : pairs)
    {
        if(forward.find(pair.image_pair.first) == forward.end())
        {
            const size_t dist = distance(old_id.begin(), old_id.find(pair.image_pair.first));
            forward[pair.image_pair.first] = dist;
            backward[dist] = pair.image_pair.first;
        }
        if(forward.find(pair.image_pair.second) == forward.end())
        {
            const size_t dist = distance(old_id.begin(), old_id.find(pair.image_pair.second));
            forward[pair.image_pair.second] = dist;
            backward[dist] = pair.image_pair.second;
        }
    }
}

// 尚未完成
bool SfM::EstimateRelativeTwithRotation()
{
    // 找到所有已经计算了全局旋转的frame的id，其实不用id，用当前frame在frames里的索引是一样的
    // 因为这两个数是相同的
    set<uint32_t> frame_with_rotation;
    for(const Frame& f : frames)
    {
        if(f.GetPose().block<3,3>(0,0).isZero())
            continue;
        frame_with_rotation.insert(f.id);
    }
    vector<pair<size_t, size_t>> pair_with_rotation;
    for(const MatchPair& pair : image_pairs)
    {
        const size_t idx1 = pair.image_pair.first;
        const size_t idx2 = pair.image_pair.second;
        if(frame_with_rotation.count(idx1) > 0 && frame_with_rotation.count(idx2) > 0)
            pair_with_rotation.emplace_back(pair.image_pair);
    }
    vector<Triplet> triplets = PoseGraph::FindTriplet(pair_with_rotation);
    LOG(INFO) << "number of triplet with global rotation: " << triplets.size()  << endl;

    // 记录每个图像对（每条边）在估算位姿时用到的triplet，只记录triplet的索引就行
    // 也就是每条边被哪些triplet所包含
    // key = edge  value = triplet id
    map<pair<size_t, size_t>, vector<size_t>> triplet_per_edge;
    for(size_t i = 0; i < triplets.size(); i++)
    {
        const Triplet& trip = triplets[i];
        // (i,j) (i,k) (j,k) 这三条边在计算相对平移的时候都用到了第i个triplet
        // 注意ijk是有顺序的，i < j < k
        triplet_per_edge[pair<size_t, size_t>(trip.i, trip.j)].push_back(i);
        triplet_per_edge[pair<size_t, size_t>(trip.i, trip.k)].push_back(i);
        triplet_per_edge[pair<size_t, size_t>(trip.j, trip.k)].push_back(i);
    }
    // 记录每个triplet能看到的三维点的数量，也就是当前triplet里三个边能三角化的点的数量之和
    // key = triplet id   value = 三维点数量
    map<size_t, size_t> tracks_per_triplet;
    // 记录被triplet包含的edge，其实就是所有的triplet的edge的集合
    vector<pair<size_t, size_t>> valid_edges;
    for(const MatchPair& pair : image_pairs)
    {
        if(triplet_per_edge.count(pair.image_pair) == 0)
            continue;
        valid_edges.push_back(pair.image_pair);
        const vector<size_t>& triplet_id = triplet_per_edge.at(pair.image_pair);
        for(const size_t& id : triplet_id)
        {
            if(tracks_per_triplet.count(id) == 0)
                tracks_per_triplet[id] = pair.triangulated.size();
            else 
                tracks_per_triplet[id] += pair.triangulated.size();
        }
    }
    // 用一个map来记录image_pair 和 它的索引之间的关系，这样就可以快速的通过匹配的图像id找到它在
    // image_pairs 里的索引
    map<pair<size_t, size_t>, size_t> image_pair_to_idx;
    for(size_t i = 0; i < image_pairs.size(); i++)
        image_pair_to_idx[image_pairs[i].image_pair] = i;

    set<pair<size_t, size_t>> processed_edges;
    for(size_t i = 0; i < valid_edges.size(); i++)
    {
        if(processed_edges.count(valid_edges[i]) > 0)
            continue;
        const vector<size_t>& triplet_id = triplet_per_edge[valid_edges[i]];
        // 找到所有包含当前的edge的triplet，然后把这些triplet按照他们能看到的三维点降序排列
        // first - triplet id    second - triplet包含的三维点数量
        vector<pair<size_t, size_t>> triplet_tracks_sorted;
        for(const size_t& id : triplet_id)
            triplet_tracks_sorted.push_back(pair<size_t, size_t>(id, tracks_per_triplet[id]));
        sort(triplet_tracks_sorted.begin(), triplet_tracks_sorted.end(), 
            [this](pair<size_t, size_t> a, pair<size_t,size_t> b){return a.second > b.second;});
        for(const pair<size_t,size_t>& t : triplet_tracks_sorted)
        {
            size_t id = t.first;
            const Triplet& triplet = triplets[id];
            vector<pair<size_t, size_t>> pairs = {pair<size_t, size_t>(triplet.i, triplet.j),
                                                pair<size_t, size_t>(triplet.i, triplet.k),
                                                pair<size_t, size_t>(triplet.j, triplet.k)};
            // 如果这个triplet所包含的三条边都已经处理过了，那就跳过
            if(processed_edges.count(pairs[0]) > 0 && 
               processed_edges.count(pairs[1]) > 0 &&
               processed_edges.count(pairs[2]) > 0 )
               continue;
            // 根据每个triplet计算相对平移
            // 1.找到这三个边对应的matches
            vector<vector<cv::DMatch>> pair_matches = {image_pairs[image_pair_to_idx[pairs[0]]].matches,
                                                    image_pairs[image_pair_to_idx[pairs[1]]].matches,
                                                    image_pairs[image_pair_to_idx[pairs[2]]].matches};
            // 2. 根据特征点之间的match生成track，并且只保留长度为3的track，也就是三张图像上都有关联
            TrackBuilder tracks_builder;
            tracks_builder.Build(pairs, pair_matches);
            tracks_builder.Filter(3);
            map<uint32_t, set<pair<uint32_t, uint32_t>>> tracks;
            tracks_builder.ExportTracks(tracks);
            if(tracks.size() < 30)
                continue;
            // openmp 并行时要单独操作
            {
                processed_edges.insert(pairs[0]);
                processed_edges.insert(pairs[1]);
                processed_edges.insert(pairs[2]);
            }
        }
    }
    return false;
}


bool SfM::EstimateGlobalTranslation(const int method)
{
    LOG(INFO) << "==================== Estimate Global Translation start =================";

    if(!config.gps_path.empty())
        SetTranslationScaleGPS(config.gps_path, true);

    // 先根据全局旋转进行相对平移的估计，得到更准确的相对平移
    // 尚未完成
    // EstimateRelativeTwithRotation();


    PrintRelativePose(config.sfm_result_path + "relpose-TA-raw.txt");
    PrintGlobalPose(config.sfm_result_path + "global-TA-raw.txt");
    // FilterByIndexDifference(10, 100, 8450);
    // FilterByStraightMotion(30);
    
    /*************************************************************************************************/
    // 找到所有已经计算了全局旋转的frame的id，其实不用id，用当前frame在frames里的索引是一样的
    // 因为这两个数是相同的
    set<uint32_t> frame_with_rotation;
    for(const Frame& f : frames)
    {
        if(f.GetPose().block<3,3>(0,0).isZero())
            continue;
        frame_with_rotation.insert(f.id);
    }
    vector<MatchPair> pair_with_rotation;
    for(const MatchPair& pair: image_pairs)
    {
        if(frame_with_rotation.count(pair.image_pair.first) > 0 && 
            frame_with_rotation.count(pair.image_pair.second) > 0)
            pair_with_rotation.emplace_back(pair);
    }
    LOG(INFO) << "image pairs with global rotaion: " << pair_with_rotation.size();
    // 把匹配的图像对分成有尺度和无尺度两种
    vector<MatchPair> pair_with_scale, pair_without_scale;
    for(const MatchPair& p : pair_with_rotation)
    {
        if(p.upper_scale >= 0 && p.lower_scale >= 0)
            pair_with_scale.push_back(p);
        else 
            pair_without_scale.push_back(p);
    }
    LOG(INFO) << "image pairs with scale: " << pair_with_scale.size() << ", without scale: " << pair_without_scale.size(); 
    assert(pair_with_rotation.size() == pair_with_scale.size() + pair_without_scale.size());
    
    // 同样，仅保留最大的边双连通子图
    set<size_t> largest_component_nodes;
    if(config.use_all_pairs_ta)
    {
        image_pairs = LargestBiconnectedGraph(pair_with_rotation, largest_component_nodes);
        LOG(INFO) << "Estimate global translation with all image pairs";
    }
    else 
    {
        image_pairs = LargestBiconnectedGraph(pair_with_scale, largest_component_nodes);
        LOG(INFO) << "Estiamte global translation with scaled image pairs";
    }
    if(image_pairs.empty())
    {
        LOG(ERROR) << "no nodes are bi-edge connected";
        return false;
    }
    pair_with_rotation.clear();
    // 输出一下更多的关于图像对的信息，debug方便些,删去这部分也不影响
    {    
        size_t points_with_depth = 0;
        size_t histo[10] = {0};
        for(const MatchPair& pair : image_pairs)
        {
            points_with_depth += pair.points_with_depth;
            size_t histo_idx = pair.points_with_depth / 10;
            histo_idx = max(0, min((int)histo_idx, 9));
            histo[histo_idx]++;
        }
        LOG(INFO) << "Image pair statistic: points with depth " << 
                    "\n 0-9: " << histo[0] << 
                    "\n 10-19: " << histo[1] << 
                    "\n 20-29: " << histo[2] <<
                    "\n 30-39: " << histo[3] <<
                    "\n 40-49: " << histo[4] <<
                    "\n 50-59: " << histo[5] <<
                    "\n 60-69: " << histo[6] <<
                    "\n 70-79: " << histo[7] <<
                    "\n 80-89: " << histo[8] <<
                    "\n 90-inf: " << histo[9] <<
                    "\n points with depth per image pair: " << 1.f * points_with_depth / image_pairs.size();
    }

    PrintRelativePose(config.sfm_result_path + "relpose-TA.txt");
    
    LOG(INFO) << "Global translation estimation:\n       prepare to estimate " << largest_component_nodes.size() <<
                " camera translations, with " << image_pairs.size() << " relative motion";

    srand((unsigned)time(NULL));
    eigen_vector<Eigen::Vector3d> global_translations(largest_component_nodes.size(), Eigen::Vector3d::Zero());
    for(size_t i = 1; i < global_translations.size(); i++)
        global_translations[i] = Eigen::Vector3d::Random();

    // 1.进行全局的重映射，把图像id映射为连续的n个数字
    map<size_t, size_t> old_to_new_global, new_to_old_global;
    ReIndex(image_pairs, old_to_new_global, new_to_old_global);
    for(MatchPair& pair : pair_with_scale)
    {
        pair.image_pair.first = old_to_new_global[pair.image_pair.first];
        pair.image_pair.second = old_to_new_global[pair.image_pair.second];
    }
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = old_to_new_global[pair.image_pair.first];
        pair.image_pair.second = old_to_new_global[pair.image_pair.second];
    }
    size_t origin_idx = 0;
    bool success;

    // 2.处理有尺度的图像对，使用DLT算出他们的绝对平移,这里还要再经过一次id重映射，因为可能不是所有图像都被覆盖了
    // 或者使用GPS作为初始的相机位姿
    if(config.init_translation_DLT && !pair_with_scale.empty())
    {
        map<size_t, size_t> old_to_new, new_to_old;
        pair_with_scale = LargestBiconnectedGraph(pair_with_scale, largest_component_nodes);
        if(largest_component_nodes.size() <= 3)
        {
            LOG(ERROR) << "pairs with scale are not enough";
        }
        ReIndex(pair_with_scale, old_to_new, new_to_old);
        for(MatchPair& pair : pair_with_scale)
        {
            pair.image_pair.first = old_to_new[pair.image_pair.first];
            pair.image_pair.second = old_to_new[pair.image_pair.second];
        }
        LOG(INFO) << "Use DLT to init global translations: " << pair_with_scale.size() << " pairs, " << 
                    largest_component_nodes.size() << " images";
        eigen_vector<Eigen::Vector3d> global_translation_scale(largest_component_nodes.size());
        
        success = TranslationAveragingDLT(image_pairs, global_translation_scale);
        if(!success)
        {
            LOG(ERROR) << "Translation average DLT failed";
            return false;
        }
        // 在后面的方法里，一般都要要求设置某个图像为原点，也就是把某个图像的平移设置为(0,0,0)并固定不动，这里就把最小二乘法得到的
        // 结果中的第一个设置为原点
        origin_idx = new_to_old_global[new_to_old[0]];
        for(size_t i = 0; i < global_translation_scale.size(); i++)
        {
            Eigen::Matrix3d R_wc = frames[new_to_old_global[new_to_old[i]]].GetPose().block<3,3>(0,0);
            Eigen::Vector3d t_wc = -R_wc * global_translation_scale[i];
            frames[new_to_old_global[new_to_old[i]]].SetTranslation(t_wc);
        }
        // 把所有的图像位姿都变换到以origin_idx为坐标原点的世界坐标系下
        // 如果origin_idx=0那这个其实就没用，因为当前的位姿就是在以0为原点的坐标系下计算得到的
        SetToOrigin(origin_idx);
        for(size_t i = 0; i < global_translations.size(); i++)
        {
            map<size_t, size_t>::iterator it = new_to_old_global.find(i);
            if(it == new_to_old_global.end())
                continue;
            if(!frames[it->second].IsPoseValid())
                continue;
            global_translations[i] = frames[it->second].GetPose().inverse().block<3,1>(0,3);
        }
        // origin_idx已经是对应于图像的id了，但是由于不是所有图像都被image pair覆盖了，所以还要把origin_idx变换到新的索引之下
        origin_idx = old_to_new_global[origin_idx];

        // 用于debug，保存一下DLT结果并显示
        CameraCenterPCD(config.sfm_result_path + "/camera_center_DLT.pcd", GetGlobalTranslation(true));
        
    }
    
    if(config.init_translation_GPS && !config.gps_path.empty())
    {
        if(!LoadGPS(config.gps_path))
        {
            LOG(ERROR) << "fail to load GPS";
            return false;
        }

        success = InitGlobalTranslationGPS(frames, global_translations, new_to_old_global);
        if(!success)
        {
            LOG(ERROR) << "Use GPS to set init translation failed";
            return false;
        }
        for(size_t i = 0; i < global_translations.size(); i++)
        {
            Eigen::Matrix3d R_wc = frames[new_to_old_global[i]].GetPose().block<3,3>(0,0);
            Eigen::Vector3d t_wc = -R_wc * global_translations[i];
            frames[new_to_old_global[i]].SetTranslation(t_wc);
        }
        SetToOrigin(new_to_old_global[origin_idx]);
        // 用于debug，保存一下DLT结果并显示
        CameraCenterPCD(config.sfm_result_path + "/camera_center_GPS.pcd", GetGlobalTranslation(true));
    }
    
    // 3.进行平移平均
    if(method == TRANSLATION_AVERAGING_SOFTL1)
    {
        LOG(INFO) << "Translation average Soft L1 ";
        success = TranslationAveragingSoftL1(image_pairs, global_translations, origin_idx, 0.01, config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average Soft L1 failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_CHORDAL)
    {
        LOG(INFO) << "Translation average chordal ";
        success = TranslationAveragingL2Chordal(image_pairs, frames, global_translations, new_to_old_global, origin_idx, 0.5, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average chordal failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_L1)
    {
        LOG(INFO) << "Translation average L1 ";
        success = TranslationAveragingL1(image_pairs, global_translations, origin_idx, new_to_old_global);
        if(!success)
        {
            LOG(ERROR) << "Translation average L1 failed";
            return false;
        }
    }
    else if (method == TRANSLATION_AVERAGING_L2IRLS)
    {
        LOG(INFO) << "Translation average L2 IRLS";
        success = TranslationAveragingL2IRLS(image_pairs, global_translations, origin_idx, config.num_iteration_L2IRLS,
                                            config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average L2 IRLS failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_BATA)
    {
        LOG(INFO) << "Translation average BATA ";
        success = TranslationAveragingBATA(image_pairs, frames, global_translations, new_to_old_global, origin_idx, config.sfm_result_path);
        if(!success)
        {
            LOG(ERROR) << "Translation average BATA failed";
            return false;
        }
    }
    else if(method == TRANSLATION_AVERAGING_LUD)
    {
        LOG(INFO) << "Translation average LUD ";
        success = TranslationAveragingLUD(image_pairs, frames, global_translations, new_to_old_global, origin_idx, config.num_iteration_L2IRLS, 
                                    config.upper_scale_ratio, config.lower_scale_ratio, config.num_threads);
        if(!success)
        {
            LOG(ERROR) << "Translation average LUD failed";
            return false;
        }
    }
    else 
    {
        LOG(ERROR) << "Translaton average method not supported";
        return false;
    }
    
    for(MatchPair& pair : image_pairs)
    {
        pair.image_pair.first = new_to_old_global[pair.image_pair.first];
        pair.image_pair.second = new_to_old_global[pair.image_pair.second];
    }
    set<size_t> frame_with_translation;
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        // global translation 算出来的是 t_cw, frame里保存的是 t_wc
        const Eigen::Vector3d& t_cw = global_translations[i];
        const Eigen::Matrix3d& R_wc = frames[new_to_old_global[i]].GetPose().block<3,3>(0,0);
        Eigen::Vector3d t_wc = - R_wc * t_cw;
        frames[new_to_old_global[i]].SetTranslation(t_wc);
        frame_with_translation.insert(new_to_old_global[i]);
    }
    // 仅保留有全局位姿的匹配对
    vector<MatchPair> good_pair;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].image_pair.first;
        const size_t idx2 = image_pairs[i].image_pair.second;
        if(frame_with_translation.count(idx1) && frame_with_translation.count(idx2) &&
            frame_with_rotation.count(idx1) && frame_with_rotation.count(idx2))
        {
            good_pair.push_back(image_pairs[i]);
        }
    }
    good_pair.swap(image_pairs);
    LOG(INFO) << "Image pairs with global pose: " << image_pairs.size() ;

    LOG(INFO) << "==================== Estimate Global Translation end =================";
    return true;
}

bool SfM::EstimateStructure()
{
    LOG(INFO) << "==================== Estimate Initial Structure start =================";
    structure = TriangulateTracks(frames, image_pairs);
    
    track_triangulated = true;
    if(config.colorize_structure)
    {
        LOG(INFO) << "Start to colorize initial structure";
        ColorizeStructure();
    }

    LOG(INFO) << "==================== Estimate Initial Structure end =================";
    return true;
}

bool SfM::GlobalBundleAdjustment(int residual_type, float redisual_threshold, bool refine_structure, bool refine_rotation, bool refine_translation)
{
    if(!SfMGlobalBA(frames, structure, residual_type, 
                    config.num_threads, refine_structure, refine_rotation, refine_translation))
    {
        LOG(ERROR) << "Global BA failed";
        return false;
    }
    size_t num_filter;
    if(residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_1 || residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_2)
    {
        num_filter = FilterTracksAngleResidual(frames, structure, redisual_threshold);
        LOG(INFO) << "Filter " << num_filter << " tracks by angle residual, " << structure.size() << " points left";
    }
    else if(residual_type == RESIDUAL_TYPE::PIXEL_RESIDUAL)
    {
        num_filter = FilterTracksPixelResidual(frames, structure, redisual_threshold);
        LOG(INFO) << "Filter " << num_filter << " tracks by pixel residual, " << structure.size() << " points left";
    }
    return true;
    
}

bool SfM::SetToOrigin(size_t frame_idx)
{
    if(frame_idx > frames.size())
    {
        LOG(ERROR) << "Invalid frame idx, no frame in frame list";
        return false;
    }
    if(!frames[frame_idx].IsPoseValid())
    {
        LOG(WARNING) << "Frame " << frame_idx << " pose is invalid, set another frame";
        for(frame_idx = 0; frame_idx < frames.size(); frame_idx++)
        {
            if(frames[frame_idx].IsPoseValid())
                break;
        }
        LOG(INFO) << "Set frame " << frame_idx << " as world coordinate";
    }
    // 这里的下标c 代表center，是指的新的世界坐标系
    const Eigen::Matrix4d T_wc = frames[frame_idx].GetPose();
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_iw = frames[i].GetPose().inverse();
        Eigen::Matrix4d T_ic = T_iw * T_wc;     // 新的世界坐标系到相机坐标系的变换
        frames[i].SetPose(T_ic.inverse());
    }
    if(track_triangulated)
    {
        Eigen::Matrix4d T_cw = T_wc.inverse();
        for(PointTrack& track : structure)
        {
            track.point_3d = (T_cw * track.point_3d.homogeneous()).hnormalized();
        }
    }
    return true;
}

bool SfM::ColorizeStructure()
{
    if(structure.empty())
    {
        LOG(ERROR) << "No structure to colorize";
        return false;
    }
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());

    map<int, set<int>> structures_each_frame;
    for(int i = 0; i < structure.size(); i++)
    {
        for(const pair<uint32_t, uint32_t>& pair : structure[i].feature_pairs)
        {
            structures_each_frame[pair.first].insert(i);
        }
    }
    // 先使用Vector3d来记录所有颜色之和，因为使用Vector3i可能会数据溢出
    eigen_vector<Eigen::Vector3d> structure_color(structure.size(), Eigen::Vector3d::Zero());
    #pragma omp parallel for
    for(int frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        map<int, set<int>>::const_iterator it = structures_each_frame.find(frame_id);
        if(it == structures_each_frame.end())
            continue;
        cv::Mat img_color = frames[frame_id].GetImageColor();
        for(const int& structure_idx : it->second)
        {
            for(const pair<uint32_t, uint32_t>& pair : structure[structure_idx].feature_pairs)
            {
                if(pair.first != frame_id)
                    continue;
                
                const cv::Point2f pt = frames[pair.first].GetKeyPoints()[pair.second].pt;
                const cv::Point2i pt_round = cv::Point2i(round(pt.x), round(pt.y));
                if(!frames[frame_id].IsInside(pt_round))
                    continue;
                const cv::Vec3b bgr = img_color.at<cv::Vec3b>(pt_round);
                #pragma omp critical
                {
                    structure_color[structure_idx].x() += bgr[2];
                    structure_color[structure_idx].y() += bgr[1];
                    structure_color[structure_idx].z() += bgr[0];
                }
            }
        }
    }
    for(int structure_idx = 0; structure_idx < structure.size(); structure_idx++)
    {
        structure_color[structure_idx] /= structure[structure_idx].feature_pairs.size();
        structure[structure_idx].rgb = structure_color[structure_idx].cast<int>();
    }
    return true;
}

int SfM::CheckRT(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, 
            const std::vector<bool>& is_inlier, const std::vector<cv::DMatch>& matches, 
            const std::vector<cv::Point3f>& keypoints1, 
            const std::vector<cv::Point3f>& keypoints2,
            double& parallax, eigen_vector<Eigen::Vector3d>& triangulated_points,
            std::vector<size_t>& inlier_idx)
{
    assert(is_inlier.size() == matches.size());
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    double cos_parallax_threshold = cos(0.5 / 180.0 * M_PI);
    double sq_reproj_error_threshold = 6.0 * 6.0;
    vector<double> parallaxes;

    const Eigen::Vector3d camera_center1 = Eigen::Vector3d::Zero();
    const Eigen::Vector3d camera_center2 = - R_21.transpose() * t_21;   // t_12

    int num_points_tri = 0;
    triangulated_points.resize(0);
    inlier_idx.clear();
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(!is_inlier[i])
            continue;
        const cv::Point3f& p1 = keypoints1[matches[i].queryIdx];
        const cv::Point3f& p2 = keypoints2[matches[i].trainIdx];
        Eigen::Vector3d point_triangulated = Triangulate2View(R_21, t_21, p1, p2);
        // Eigen::Vector3d point_triangulated = Triangulate2ViewIDWM(R_21, t_21, p1, p2);
        if (!std::isfinite(point_triangulated(0))
            || !std::isfinite(point_triangulated(1))
            || !std::isfinite(point_triangulated(2))) 
            continue;
        // 计算视差
        Eigen::Vector3d norm1 = (point_triangulated - camera_center1).normalized();
        Eigen::Vector3d norm2 = (point_triangulated - camera_center2).normalized();

        double cos_parallax = norm2.dot(norm1);
        // 角度越小，那么对应的余弦就越大，因此如果当前夹角的余弦值大于阈值，就说明当前的夹角是很小的
        // 这种情况下三角化很可能就出错了
        const bool parallax_is_small = cos_parallax_threshold < cos_parallax;

        Eigen::Vector3d p1_eigen(p1.x, p1.y, p1.z);
        Eigen::Vector3d p2_eigen(p2.x, p2.y, p2.z);

        Eigen::Vector3d point_in_frame2 = R_21 * point_triangulated + t_21;

        // 计算三角化得到的点和它对应的图像点之间的夹角，如果夹角太大就认为三角化错误
        double reproj_error_angle1 = VectorAngle3D(norm1.data(), p1_eigen.data()) * 180.0 / M_PI;
        if(reproj_error_angle1 > 3)
            continue;

        double reproj_error_angle2 = VectorAngle3D(point_in_frame2.data(), p2_eigen.data()) * 180.0 / M_PI;
        if(reproj_error_angle2 > 3)
            continue;
        num_points_tri ++;
        parallaxes.push_back(cos_parallax);
        triangulated_points.push_back(point_triangulated);
        inlier_idx.push_back(i);
    }
    if(num_points_tri > 0)
    {
        // 把视差的余弦按照从小到大的顺序排列，然后找到其中的第50个，计算他所对应的视差
        // 其实就是把所有视差按从大到小排列，找到第50大的视差
        sort(parallaxes.begin(), parallaxes.end());
        size_t idx = min(50, static_cast<int>(parallaxes.size()));
        parallax = acos(parallaxes[idx - 1]) * 180.0 / M_PI;
    }
    else 
        parallax = 0;
    return num_points_tri;
}

void SfM::VisualizeTrack(const PointTrack& track, const string path)
{
    int track_id = track.id;
    Eigen::Vector3d point_world = track.point_3d;
    for(const auto& pair : track.feature_pairs)
    {
        cv::Mat img = frames[pair.first].GetImageColor();
        cv::Point2f pt = frames[pair.first].GetKeyPoints()[pair.second].pt;
        Equirectangular eq(img.rows, img.cols);
        cv::circle(img, pt, 20, cv::Scalar(0,0,255), 5);
        Eigen::Vector3d point_camera = (frames[pair.first].GetPose().inverse() * point_world.homogeneous()).hnormalized();
        Eigen::Vector2d point_image = eq.CamToImage(point_camera);
        pt.x = static_cast<float>(point_image.x());
        pt.y = static_cast<float>(point_image.y());
        cv::circle(img, pt, 20, cv::Scalar(255,0,0), 5);
        cv::imwrite(path + "/track" + num2str(track_id) + "_" + num2str(pair.first) + ".jpg", img);
    }
}

bool SfM::ExportMatchPairTXT(const std::string file_name)
{
    LOG(INFO) << "Save match pair at " << file_name;
    ofstream f(file_name);
    if(!f.is_open())
        return false;
    for(MatchPair& p:image_pairs)
    {
        f << p.image_pair.first << " " << p.image_pair.second << endl;
        f << p.R_21(0, 0) << " " << p.R_21(0, 1) << " " << p.R_21(0, 2) << " " << p.t_21(0) << " " 
          << p.R_21(1, 0) << " " << p.R_21(1, 1) << " " << p.R_21(1, 2) << " " << p.t_21(1) << " "
          << p.R_21(2, 0) << " " << p.R_21(2, 1) << " " << p.R_21(2, 2) << " " << p.t_21(2) << endl;
        f << "points with depth: " << p.points_with_depth << endl;
    }
    f.close();
    return true;
}

bool SfM::LoadMatchPairTXT(const std::string file_name)
{
    ifstream f(file_name);
    if(!f.is_open())
    {
        LOG(ERROR) << "Can not open file " << file_name;
        return false;
    }
    image_pairs.clear();
    size_t largets_idx = 0;
    while(!f.eof())
    {
        // 从文件中读取数据并保存到image pairs
        // 同时还要检查一下数据是否正确，可以通过检查读取的图像index来判断一下
        // 如果图像的index大于已有的图像数量，那么就可以肯定这个数据不对
        size_t i , j;
        Eigen::Matrix3d R_21;
        Eigen::Vector3d t_21;
        f >> i >> j;
        f >> R_21(0, 0) >> R_21(0, 1) >> R_21(0, 2) >> t_21(0) 
          >> R_21(1, 0) >> R_21(1, 1) >> R_21(1, 2) >> t_21(1)        
          >> R_21(2, 0) >> R_21(2, 1) >> R_21(2, 2) >> t_21(2);
        string str;
        getline(f, str);
        getline(f, str);
        str = str.substr(19);
        int num_points = str2num<int>(str);
        MatchPair p(i,j);
        p.R_21 = R_21;
        p.t_21 = t_21;
        p.points_with_depth = num_points;
        // 三维向量到反对称矩阵
        Eigen::Matrix3d t_21_hat = Eigen::Matrix3d::Zero();
        t_21_hat <<    0,          -t_21.z(),      t_21.y(),
                    t_21.z(),           0,        -t_21.x(),
                    -t_21.y(),      t_21.x(),           0;
        p.E_21 = t_21_hat * R_21;

        image_pairs.push_back(p);

        if(i > largets_idx)
            largets_idx = i;
        if(j > largets_idx)
            largets_idx = j;

        if(f.peek() == EOF)
            break;
    }
    if(largets_idx >= frames.size())
    {
        LOG(ERROR) << "Fail to load match pair at" << file_name << endl;
        image_pairs.clear();
        return false;
    }
    LOG(INFO) << "Successfully load " << image_pairs.size() << " match pairs from txt";
    return true;
}

bool SfM::ExportMatchPairBinary(const std::string folder)
{
    return ExportMatchPair(folder, image_pairs);
}

bool SfM::LoadMatchPairBinary(const std::string folder)
{
    // 读取的时候不需要太多的线程，没办法提升效率，而且可能还会导致效率降低
    if(!ReadMatchPair(folder, image_pairs, min(config.num_threads, 4)))
        return false;
    if(image_pairs[image_pairs.size() - 1].image_pair.second > frames.size())
    {
        image_pairs.clear();
        LOG(ERROR) << "Fail to load match pair, #images(in pairs) more than #frames";
        return false;
    }
    LOG(INFO) << "Successfully load " << image_pairs.size() << " match pairs from " << folder;
    return true;

}

bool SfM::ExportFrameBinary(const std::string folder)
{
    if(frames.empty())
        return false;
    return ExportFrame(folder, frames);
}

bool SfM::ExportStructureBinary(const std::string file_name)
{
    if(structure.empty())
        return false;
    return ExportPointTracks(file_name, structure);
}

bool SfM::LoadStructureBinary(const std::string file_name)
{
    structure.clear();
    return ReadPointTracks(file_name, structure);
}

bool SfM::LoadGPS(const std::string file_name)
{
    eigen_vector<Eigen::Vector3d> gps_list;
    vector<string> name_list;
    ReadGPS(config.gps_path, gps_list, name_list);
    if(gps_list.size() != frames.size())
    {
        LOG(ERROR) << "Fail to load GPS file, number of GPS != number of Frame";
        return false;
    }
    for(int i = 0; i < frames.size(); i++)
        frames[i].SetGPS(gps_list[i]);
    return true;
}

eigen_vector<Eigen::Matrix3d> SfM::GetGlobalRotation(bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> global_rotation;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = f.GetPose();
        global_rotation.push_back(T_wc.block<3,3>(0,0));
    }
    return global_rotation;
}

eigen_vector<Eigen::Vector3d> SfM::GetGlobalTranslation(bool with_invalid)
{
    eigen_vector<Eigen::Vector3d> global_translation;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = f.GetPose();
        global_translation.push_back(T_wc.block<3,1>(0,3));
    }
    return global_translation;
}

std::vector<std::string> SfM::GetFrameNames(bool with_invalid)
{
    vector<string> names;
    for(const Frame& f : frames)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if(f.IsPoseValid() == false && with_invalid == false)
            continue;
        names.push_back(f.name);
    }
    return names;
}

bool SfM::ExportStructurePCD(const string file_name)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    for(const PointTrack& t : structure)
    {
        const Eigen::Vector3d point = t.point_3d;
        const Eigen::Vector3i color = t.rgb;
        pcl::PointXYZRGB pt;
        pt.x = point.x();
        pt.y = point.y();
        pt.z = point.z();
        pt.r = static_cast<uchar>(color.x());
        pt.g = static_cast<uchar>(color.y());
        pt.b = static_cast<uchar>(color.z());
        cloud.push_back(pt);
    }
    pcl::io::savePCDFileASCII(file_name, cloud);
    return true;
}

const std::vector<Frame>& SfM::GetFrames() const 
{
    return frames;
}

const std::vector<Velodyne>& SfM::GetLidars() const 
{
    return lidars;
}

void SfM::SetLidars(const std::vector<Velodyne>& _lidars)
{
    lidars = _lidars;
}
SfM::~SfM()
{
}