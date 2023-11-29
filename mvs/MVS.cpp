/*
 * @Author: Diantao Tu
 * @Date: 2022-06-29 13:55:06
 */
#include "MVS.h"


using namespace std;

inline cv::Vec2f Normal2Dir(cv::Vec3f n)
{
    #ifdef USE_FAST_ATAN2
    return cv::Vec2f(FastAtan2(n[1], n[0]), std::acos(n[2]));
    #else 
	return cv::Vec2f(std::atan2(n[1], n[0]), std::acos(n[2]));
    #endif
}

inline cv::Vec3f Dir2Normal(cv::Vec2f d)
{
	float s = sin(d[1]);
	return cv::Vec3f(cos(d[0])*s, sin(d[0])*s, cos(d[1]));
}

MVS::MVS(const std::vector<Frame>& _frames, const std::vector<Velodyne>& _lidars, const Config& _config):
                    frames(_frames), config(_config), lidars(_lidars),
                    eq(Equirectangular(frames[0].GetImageRows(), frames[0].GetImageCols()))
{
    assert(frames.size() == lidars.size());
    rng = cv::RNG((unsigned)time(NULL));
    ncc_window_size = 2 * config.ncc_half_window + 1;
    num_texels = Square((ncc_window_size) / config.ncc_step + (config.ncc_step > 1));
    sigma_color = -1.f / (2 * 0.2 * 0.2);
	sigma_spatial = -1.f / (2.f * config.ncc_half_window * config.ncc_half_window);
    eq.PreComputeI2C();
    frame_mutex = vector<mutex>(frames.size());
    frame_gray_count = vector<int>(frames.size(), 0);;
    frame_conf_count = vector<int>(frames.size(), 0);;
    frame_depth_count = vector<int>(frames.size(), 0);;
    frame_normal_count = vector<int>(frames.size(), 0);;
    frame_depth_filter_count = vector<int>(frames.size(), 0);;
}

bool MVS::EstimateStructure()
{
    // 把图像尺度恢复到原始的尺度，因为之前的步骤中图像都是以原始尺寸计算的，如果不恢复就会导致BA出现错误
    #pragma omp parallel for
    for(Frame& f : frames)
        f.SetImageScale(0);
    vector<MatchPair> image_pairs;
    if(!ReadMatchPair(config.match_pair_joint_path, image_pairs, 4))
        return false;
    structure = TriangulateTracks(frames, image_pairs);
    #pragma omp parallel for
    for(Frame& f : frames)
        f.SetImageScale(config.scale);
    ExportPointTracks(config.mvs_result_path + "points.bin", structure);
    return structure.size() > 0;
}

bool MVS::LoadStructure(std::string file_name)
{
    return ReadPointTracks(file_name, structure);
}

bool MVS::SelectNeighborViews(int neighbor_size, int method, float min_distance)
{
    neighbors.clear();
    neighbors.resize(frames.size());
    if(method == NeighborSelection::NEAREST_NEIGHBOR)
        return SelectNeighborKNN(neighbor_size, Square(min_distance));
    else if(method == NeighborSelection::SFM_POINTS)
        return SelectNeighborSFM(neighbor_size, Square(min_distance));
    else 
    {
        LOG(ERROR) << "Unknown neighbor selection method ";
        return false;
    }
}

bool MVS::EstimateDepthMaps(int method, const cv::Mat& mask)
{
    LOG(INFO) << "============== Estimate depth maps begin ===============";
    for(Frame& f : frames)
        f.ReleaseKeyPoints();
    omp_set_num_threads(config.num_threads);
    bool enable_parallel = (method == Propagate::SEQUENTIAL);
    LOG(INFO) << "propagate based on photometric consistency";
    ProcessBar bar(frames.size(), 0.1);
    /* 使用棋盘格传播和顺序传播的时候，线程的安排是不同的，因为棋盘格传播时每张图像可以多线程并行计算，
       所以每次只处理一张图像。顺序传播的时候无法并行计算，那么就多个线程，每个线程处理一张图像*/
    // 首先只使用光度一致性进行深度图的计算
    #pragma omp parallel for schedule(dynamic) if(enable_parallel)
    for(int i = 0; i < frames.size(); i++)
    {
        if(Initialize(i, mask, !enable_parallel, config.mvs_use_lidar))
        {
            EstimateDepthMapSingle(i, method, config.mvs_use_geometric ? 3 : 5, -0.7, false);     
            // 不使用几何一致性的情况下，到这里就是深度图计算的最终步骤了，所以输出可视化
            if(!config.mvs_use_geometric)
            {
                RemoveSmallSegments(i);
                // GapInterpolation(i);
                cv::imwrite(config.mvs_result_path + "conf_" + num2str(i) + "_final.jpg", DepthImageRGB(frames[i].conf_map));
                cv::imwrite(config.mvs_result_path + "normal_" + num2str(i) + "_final.jpg", DrawNormalImage(frames[i].normal_map, true));
                cv::imwrite(config.mvs_result_path + "depth_" + num2str(i) + "_final.jpg", DepthImageRGB(frames[i].depth_map, config.max_depth_visual, config.min_depth)); 
                #if 1
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(0);
                cv::imwrite(config.mvs_result_path + "depth_" + num2str(i) + "_final16.png", DepthImageGray16(frames[i].depth_map), compression_params);
                #endif
            }
            FinishEstimation(i, false); 
        }
        bar.Add();
    }
    /*
    TODO: 使用几何一致性的情况下还是有一些问题的，目前没精力解决，而且MVS也不是重点，所以暂时先不管了。
    几何一致性的传播目标是让邻域图像之间的深度图更加一致，所以应该是每迭代一次，就更新一次邻域图像的深度图，然后再进行下一次迭代。
    也就是说应该是是所有图像先迭代一次，然后更新自己的深度图，接着所有图像再迭代一次，这样才能满足深度图一致性的需要 
    然而目前由于程序的架构问题，没法做到以上目标，只能是每张图像连续迭代两次，
    这样就会导致最后的深度图和使用光度一致性的深度图具有较高的连续性，会产生一些伪影（我师弟把OpenMVS的代码改成
    在不更新邻域深度图的情况下，连续三次几何一致性迭代，发现深度图出现了伪影。他是不小心改错的，实际应该是一次迭代）。
    */
    if(config.mvs_use_geometric)
    {
        LOG(INFO) << "propagate based on photometric + geometric consistency";
        ResetFrameCount();
        ProcessBar bar2(frames.size(), 0.1);
        // 然后增加几何一致性计算
        #pragma omp parallel for schedule(dynamic) if(enable_parallel)
        for(int ref_id = 0; ref_id < frames.size(); ref_id++)
        {
            if(Initialize(ref_id, mask, !enable_parallel, config.mvs_use_lidar, true))
            {
                EstimateDepthMapSingle(ref_id, method, 2, -0.3, true);
                RemoveSmallSegments(ref_id);
                // GapInterpolation(ref_id);
                cv::imwrite(config.mvs_result_path + "conf_" + num2str(ref_id) + "_final.jpg", DepthImageRGB(frames[ref_id].conf_map));
                cv::imwrite(config.mvs_result_path + "normal_" + num2str(ref_id) + "_final.jpg", DrawNormalImage(frames[ref_id].normal_map, true));
                cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_final.jpg", DepthImageRGB(frames[ref_id].depth_map, config.max_depth_visual, config.min_depth));
                #if 1
                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(0);
                cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_final16.png", DepthImageGray16(frames[ref_id].depth_map), compression_params);
                #endif
                FinishEstimation(ref_id, true);
            }
            bar2.Add();
        }
    }
    for(const Frame& f : frames)
    {
        if(!f.conf_map.empty())
            LOG(ERROR) << "frame " << f.id << " conf map is not empty, frame_conf_count = " << frame_conf_count[f.id];
        if(!f.depth_map.empty())
            LOG(ERROR) << "frame " << f.id << " depth map is not empty, frame_depth_count = " << frame_depth_count[f.id];
        if(!f.normal_map.empty())
            LOG(ERROR) << "frame " << f.id << " normal map is not empty, frame_normal_count = " << frame_normal_count[f.id];
    }

    LOG(INFO) << "============== Estimate depth maps end ===============";
    return true;
}


bool MVS::FilterDepthMaps()
{
    LOG(INFO) << "Filter depth maps begin";
    ResetFrameCount();
    #pragma omp parallel for schedule(dynamic)
    for(int ref_id = 0; ref_id < frames.size(); ref_id++)
    {
        if(!frames[ref_id].IsPoseValid())
            continue;
        vector<int> ids = {ref_id};
        for(const NeighborInfo& n : neighbors[ref_id])
            ids.push_back(n.id);
        for(const int& id : ids)
        {
            lock_guard<mutex> guard(frame_mutex[id]);
            if(frames[id].depth_map.empty())
                ReadFrameDepth(config.mvs_depth_path + num2str(id) + (config.mvs_use_geometric ? "_geo" : "_pho") + ".bin", frames[id], false);
            frame_depth_count[id]++;
            if(frames[id].conf_map.empty())
            {
                ReadFrameConf(config.mvs_conf_path + num2str(id) + (config.mvs_use_geometric ? "_geo" : "_pho") + ".bin", frames[id]);
                ConvertNCC2Conf(frames[id].conf_map);
            }
            frame_conf_count[id]++;
        }
        FilterDepthImageRefine(ref_id);
        // FilterDepthImage(ref_id);
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_filter.jpg", DepthImageRGB(frames[ref_id].depth_filter, config.max_depth_visual, config.min_depth));
        #if 0
        // 输出16位深度图，可以用于别的程序
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(0);
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_filter16.png", DepthImageGray16(frames[ref_id].depth_filter), compression_params);
        #endif
        // 减少对深度图的引用计数，一旦某个计数减为0，就代表这个图像对应的初始深度图不需要保留了，直接释放内存
        for(const int& id : ids)
        {
            lock_guard<mutex> guard(frame_mutex[id]);
            if(--frame_depth_count[id] <= 0)
            {
                if(!frames[id].depth_map.empty())
                    frames[id].depth_map.release();
            }
            if(--frame_conf_count[id] <= 0)
            {
                if(!frames[id].conf_map.empty())
                    frames[id].conf_map.release();
            }
        }
    }
    LOG(INFO) << "Filter depth maps end";
    return true;
}

bool MVS::FuseDepthMaps()
{
    // 把所有深度直接投影到世界坐标系然后融合到一起
    pcl::io::savePCDFileBinary(config.mvs_result_path + "MVS-merge.pcd", MergeDepthImages(2));
    
    // 把深度按照置信度加权，并且除去不可靠的点，从openMVS抄的
    pcl::io::savePCDFileBinary(config.mvs_result_path + "MVS-fuse.pcd", FuseDepthImages());
    return true;
}

void MVS::ResetFrameCount()
{
    for(auto& i : frame_conf_count)
        i = 0;
    for(auto& i : frame_normal_count)
        i = 0;
    for(auto& i : frame_gray_count)
        i = 0;
    for(auto& i : frame_depth_count)
        i = 0;
    for(auto& i : frame_depth_filter_count)
        i = 0;
}

bool MVS::SelectNeighborSFM(int neighbor_size, float sq_distance_threshold)
{
    if(structure.empty())
    {
        LOG(ERROR) << "SfM points are empty, unable to select neighbor views";
        return false;
    }
    // 记录每张图像作为其他图像的近邻时的评分，第i行第j列 代表 第j张图像作为第i张图像的近邻的评分
    // 也就是说，行数作为reference image，列数作为neighbor image
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> neighbor_score;
    neighbor_score.resize(frames.size(), frames.size());
    neighbor_score.fill(0);

    const double angle_threshold = 10;
    const double scale_threshold = 1.6;

    for(const PointTrack& t : structure)
    {
        // 找到当前track所对应的所有图像
        vector<uint32_t> views;
        for(const pair<uint32_t, uint32_t>& p : t.feature_pairs)
            views.push_back(p.first);
        for(size_t i = 0; i < views.size(); i++)
        {
            for(size_t j = i + 1; j < views.size(); j++)
            {
                uint32_t id1 = views[i], id2 = views[j];
                Eigen::Vector3d C1 = frames[id1].GetPose().block<3,1>(0,3);
                Eigen::Vector3d C2 = frames[id2].GetPose().block<3,1>(0,3);
                Eigen::Vector3d V1 = t.point_3d - C1;
                Eigen::Vector3d V2 = t.point_3d - C2;
                double depth1 = V1.norm();
                double depth2 = V2.norm();
                double angle = VectorAngle3D(V1.data(), V2.data()) * 180 / M_PI;
                angle = min(powf(angle / angle_threshold, 1.5f), 1.f); // 夹角因子
                // 这个scale是由id1到id2的
                double scale = depth1 / depth2;
                if (scale > scale_threshold)  // 比值因子
					scale = scale_threshold * scale_threshold / scale / scale;
                else if (scale >= 1.f)
					scale = 1.f;
				else
					scale = scale * scale;
                neighbor_score(id1, id2) += scale * angle;
                // 这个scale是由id2到id1的
                scale = depth2 / depth1;
                if (scale > scale_threshold)  // 比值因子
					scale = scale_threshold * scale_threshold / scale / scale;
                else if (scale >= 1.f)
					scale = 1.f;
				else
					scale = scale * scale;
                neighbor_score(id2, id1) += scale * angle;

            }
        }
    }
    for(size_t row = 0; row < neighbor_score.rows(); row++)
    {
        vector<pair<float, size_t>> candidate_neighbor_socre;
        for(size_t col = 0; col < neighbor_score.cols(); col++)
        {
            float score = neighbor_score(row, col);
            if(score > 0)
                candidate_neighbor_socre.push_back({score, col});
        }
        // 按照pair中第一个数字降序排列，也就是按照分值排列
        sort(candidate_neighbor_socre.begin(), candidate_neighbor_socre.end(), std::greater<>());
        vector<NeighborInfo> curr_neighbors; 
        for(size_t i = 0; i < candidate_neighbor_socre.size() && curr_neighbors.size() < neighbor_size; i++)
        {
            const size_t& neighbor_id = candidate_neighbor_socre[i].second;
            Eigen::Matrix4d T_nr = frames[neighbor_id].GetPose().inverse() * frames[row].GetPose(); 
            Eigen::Vector3d t_nr = T_nr.block<3,1>(0,3);
            if(t_nr.norm() > sq_distance_threshold)
                curr_neighbors.push_back(NeighborInfo(neighbor_id, T_nr.block<3,3>(0,0), t_nr));
        }
        if(curr_neighbors.size() < neighbor_size)
            LOG(WARNING) << "Frame " << row << " unable to find enough neighbors, #neighbors = " << curr_neighbors.size();
        neighbors[row] = curr_neighbors;
    }

    return true;

}

bool MVS::SelectNeighborKNN(int neighbor_size, float sq_distance_threshold)
{
    // 记录下每个图像帧的位置, 为了之后进行近邻搜索
    pcl::PointCloud<pcl::PointXYZI> camera_center;
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        pcl::PointXYZI center;
        Eigen::Vector3d t_wl = frames[i].GetPose().block<3,1>(0,3);
        center.x = t_wl.x();
        center.y = t_wl.y();
        center.z = t_wl.z();
        // 设置intensity只是为了知道当前的点对应于哪一帧图像，因为可能有的图像没有位姿就没被记录下来
        center.intensity = i;   
        camera_center.push_back(center);
    }
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_center(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kd_center->setInputCloud(camera_center.makeShared());
    // 找到最近邻的几个数据，并进行相互关联
    for(size_t ref_id = 0; ref_id < frames.size(); ref_id++)
    {
        if(!frames[ref_id].IsPoseValid())
            continue;
        Eigen::Vector3d t_wl = frames[ref_id].GetPose().block<3,1>(0,3);
        pcl::PointXYZI center(ref_id);
        EigenVec2PclPoint(t_wl, center);
        vector<int> candidate_neighbor_ids;
        vector<float> sq_neighbor_distance;
        kd_center->nearestKSearch(center, neighbor_size * 3, candidate_neighbor_ids, sq_neighbor_distance);
        // 之前的neighbors保存的是在lidar center这个点云里的索引，但不是所有的雷达帧位姿都被保存在那里，所以
        // 点云里的索引不是真正的lidar的索引，每个点的intensity才是这个点对应的LiDAR的索引
        // 要从 i = 1开始，因为最近的永远都是自己
        vector<int> neighbor_ids;
        vector<NeighborInfo> curr_neighbors;
        for(int i = 1; i < candidate_neighbor_ids.size() && curr_neighbors.size() < neighbor_size; i++)
        {
            // 距离要大于一定阈值才行，不然距离太近会导致估计的深度误差较大
            if(sq_neighbor_distance[i] < sq_distance_threshold)
                continue;
            int neighbor_id = camera_center[candidate_neighbor_ids[i]].intensity;
            Eigen::Matrix4d T_nr = frames[neighbor_id].GetPose().inverse() * frames[ref_id].GetPose(); 
            curr_neighbors.push_back(NeighborInfo(neighbor_id, T_nr.block<3,3>(0,0), T_nr.block<3,1>(0,3)));
        }
        neighbors[ref_id] = curr_neighbors;
    }
    return true;
}

bool MVS::RefineCameraPose()
{
    if(structure.empty())
        return false;
    // 假设当前的雷达和相机已经对齐了，那么就需要保存一下雷达和相机之间的相对位姿，因为经过之后的BA，相机位姿会改变，
    // 那么就需要雷达位姿也随之改变。但是雷达和相机之间的相对位姿是不能变的。
    eigen_vector<Eigen::Matrix4d> T_cl_list(lidars.size());
    if(!lidars.empty())
    {
        for(size_t i = 0; i < lidars.size(); i++)
        {
            if(!lidars[i].IsPoseValid() || !frames[i].IsPoseValid())
                continue;
            T_cl_list[i] = frames[i].GetPose().inverse() * lidars[i].GetPose();
        }
    }

    CameraCenterPCD(config.mvs_result_path + "camera_center_mvs_init.pcd", GetGlobalTranslation());
    // 把图像尺度恢复到原始的尺度，因为之前的步骤中图像都是以原始尺寸计算的，如果不恢复就会导致BA出现错误
    #pragma omp parallel for
    for(Frame& f : frames)
        f.SetImageScale(0);
    if(!SfMGlobalBA(frames, structure, RESIDUAL_TYPE::PIXEL_RESIDUAL, 
                    config.num_threads, true, true, true))
    {
        LOG(ERROR) << "Global BA failed";
        return false;
    }
    #pragma omp parallel for
    for(Frame& f : frames)
        f.SetImageScale(config.scale);
    // 根据相对位姿计算得到雷达位姿
    if(!lidars.empty())
    {
        for(size_t i = 0; i < lidars.size(); i++)
        {
            if(!lidars[i].IsPoseValid() || !frames[i].IsPoseValid())
                continue;
            lidars[i].SetPose(frames[i].GetPose() * T_cl_list[i]);
        }
    }
    CameraCenterPCD(config.mvs_result_path + "camera_pose_mvs_refine.pcd", GetGlobalTranslation());
    ExportPoseT(config.mvs_result_path + "camera_pose_after_refine.txt", GetGlobalRotation(), GetGlobalTranslation(), GetFrameNames());
    LOG(INFO) << "Finish refine camera pose";
    return true;
}

bool MVS::Initialize(int ref_id, const cv::Mat& mask, const bool enable_parallel, const bool use_lidar, const bool use_geometry)
{
    Frame& frame = frames[ref_id];
    /* 初始化深度图和法向量图 */
    if(!use_geometry)
    {
        if(!InitDepthNormal(ref_id, mask, use_lidar))
        {
            LOG(ERROR) << "fail to init depth normal for frame " << ref_id;
            return false;
        }
    }
    else 
    {
        lock_guard<mutex> guard(frame_mutex[ref_id]);
        bool success = ReadFrameNormal(config.mvs_normal_path + num2str(ref_id) + "_pho.bin", frames[ref_id]);
        success &= ReadFrameDepth(config.mvs_depth_path + num2str(ref_id) + "_pho.bin", frames[ref_id], false);
        if(!success)
            return false;
    }

    /* 读取参考图像和其他近邻图像的灰度图 
        这里根据传播方式决定是否进行并行化，如果当前是棋盘格传播，那么就并行化。
        如果是顺序传播，那么就单线程。这是因为如果当前是顺序传播，那么此时会有多个线程
        同时在进行初始化（因为是同时计算多张图像），那么在初始化的过程中就不需要多线程了
    */
    vector<int> ids = {ref_id};
    for(const NeighborInfo& n : neighbors[ref_id])
        ids.push_back(n.id);
    #pragma omp parallel for if(enable_parallel)
    for(const int& id : ids)
    {
        lock_guard<mutex> guard(frame_mutex[id]);
        if(frames[id].GetImageGrayRef().empty())
            frames[id].LoadImageGray(frames[id].name);
        frame_gray_count[id] += 1;
        if(use_geometry)  
        {
            // 在进行含有几何一致性的NCC的时候，需要读取原本的基于光度一致性得到的深度图，并且把深度图保存到depth_filter备用
            if(frames[id].depth_filter.empty())
                ReadFrameDepth(config.mvs_depth_path + num2str(id) + "_pho.bin", frames[id], true);
            frame_depth_filter_count[id]++;
        }
    }
    /* 初始化图像上每个patch的信息 */
    InitPatchMap(ref_id, enable_parallel);
        
    /* 初始化置信度图 */
    frame.conf_map = cv::Mat(frame.depth_map.size(), CV_32F, cv::Scalar(-1));
    InitConfMap(ref_id, enable_parallel, use_geometry);
    // 在整个系统初始化的状态下，输出一下初始的深度、法向量、置信度
    if(!use_geometry)
    {
        cv::imwrite(config.mvs_result_path + "conf_" + num2str(ref_id) + "_init.jpg", DepthImageRGB(frames[ref_id].conf_map));
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_init.jpg", 
            DepthImageRGB(frames[ref_id].depth_map, config.max_depth_visual, config.min_depth));
        cv::imwrite(config.mvs_result_path + "normal_" + num2str(ref_id) + "_init.jpg", DrawNormalImage(frames[ref_id].normal_map, true));
    }
    lock_guard<mutex> guard(frame_mutex[ref_id]);
    frame_conf_count[ref_id]++;
    frame_normal_count[ref_id]++;
    frame_depth_count[ref_id]++;
    return true;
}


bool MVS::InitDepthNormal(int ref_id, const cv::Mat& mask, bool use_lidar)
{
    assert(mask.empty() || mask.type() == CV_32F);
    Frame& frame = frames[ref_id];
    if(!frame.IsPoseValid())
        return false;
    if(use_lidar && lidars[ref_id].IsPoseValid())
    {
        Velodyne lidar(lidars[ref_id]);
        lidar.LoadLidar(lidar.name);
        /* 初始化深度图 */
        Eigen::Matrix4d T_cl = frame.GetPose().inverse() * lidar.GetPose();
        #if 0
        frame.depth_map = DepthCompletion(
                    ProjectLidar2PanoramaDepth(lidar.cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);
        #elif 1
        frame.depth_map = ProjectLidar2PanoramaDepth(lidar.cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 2);
        frame.depth_map.convertTo(frame.depth_map, CV_32F);
        frame.depth_map /= 256.f;
        #elif 0
        Eigen::Vector4f plane_coeff;
        pcl::PointCloud<PointType> ground, nonground;
        lidar.ReOrderVLP2();
        lidar.ExtractGroundPointCloud(ground, nonground, plane_coeff);
        cv::Mat ground_pixel = ExtractGroundPixel(ProjectLidar2PanoramaDepth(ground, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3));
        // 用非地面点进行深度补全
        // frame.depth_map = DepthCompletion(
        //     ProjectLidar2PanoramaDepth(nonground, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);
        frame.depth_map = ProjectLidar2PanoramaDepth(nonground, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3);
        frame.depth_map.convertTo(frame.depth_map, CV_32F);
        frame.depth_map /= 256.f;
        // 对地面点进行平面拟合然后插值得到深度图
        Eigen::Vector4f plane_coeff_camera = TranslePlane(plane_coeff, Eigen::Matrix4f(T_cl.cast<float>()));
        for(int row = 0; row < frame.GetImageRows(); row ++)
        {
            for(int col = 0; col < frame.GetImageCols(); col++)
            {
                if(ground_pixel.at<uchar>(row, col) == 0)
                    continue;
                cv::Point3f X0 = eq.ImageToCam(cv::Point2i(col, row));
                Vector6f view_ray(0, 0, 0, X0.x, X0.y, X0.z);
                Eigen::Vector3f p = PlaneLineIntersect<float, Eigen::Vector3f>(plane_coeff_camera.data(), view_ray.data());
                if(isinf(p.x()) )
                    continue;
                frame.depth_map.at<float>(row, col) = p.norm();
            }
        }
        #elif 0
        frame.depth_map = DepthCompletionDelaunay(frame.GetImageRows(), frame.GetImageCols(), lidar.cloud, T_cl);
        #endif 
        
        cv::Mat depth_random = cv::Mat::zeros(frame.depth_map.size(), CV_32F);
        rng.fill(depth_random, cv::RNG::UNIFORM, config.max_depth, config.min_depth);
        // 对深度图进行处理，有雷达深度的地方保留，其他的地方填充随机值，有mask的地方保留为0。
        // 此时的depth image是雷达深度图，那么就根据此深度得到一个lidar-mask，其中有深度的地方设置为0，
        // 没有深度的地方设置为1，然后把这个lidar-mask和随机的深度相乘，那么雷达区域就没有随机深度了
        // 最后再加上雷达深度，就达到了把雷达深度和随机深度结合到一起的目的
        cv::Mat lidar_mask;
        cv::threshold(frame.depth_map, lidar_mask, 0, 1, CV_THRESH_BINARY_INV);
        cv::add(frame.depth_map, depth_random.mul(lidar_mask), frame.depth_map);  

        // 把雷达投影的位置设置为固定深度不变
        if(config.keep_lidar_constant)
        {
            frame.depth_constant = 1 - lidar_mask;
            frame.depth_constant.convertTo(frame.depth_constant, CV_8U);
        }
    }
    else 
    {
        frame.depth_map = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);
        rng.fill(frame.depth_map, cv::RNG::UNIFORM, config.max_depth, config.min_depth);
    }
    // 最后把深度图和本身的mask相乘，把mask的地方的深度设置为0
    frame.depth_map = frame.depth_map.mul(mask);

    /* 初始化法向量图 */
    frame.normal_map = cv::Mat::zeros(frame.depth_map.size(), CV_32FC3);
    for(size_t row = 0; row < frame.normal_map.rows; row++)
    {
        for(size_t col = 0; col < frame.normal_map.cols; col++)
        {
            if(mask.at<float>(row, col) < 1)
                continue;
            frame.normal_map.at<cv::Vec3f>(row, col) = GenerateRandomNormal(cv::Point2i(col, row));
        }
    }
    return true;
}

bool MVS::InitConfMap(int ref_id, bool enable_parallel, bool use_geometry)
{
    Frame& frame = frames[ref_id];
    #pragma omp parallel for if(enable_parallel)
    for(int row = 0; row < frame.conf_map.rows; row++)
    {
        for(int col = 0; col < frame.conf_map.cols; col++)
        {
            if(frame.depth_map.at<float>(row, col) <= 0)
                continue;
            float conf = -1;
            const float& depth = frame.depth_map.at<float>(row, col);
            const cv::Point3f X0 = eq.ImageToCam(cv::Point2i(col, row)) * depth;
            const cv::Vec3f& normal = frame.normal_map.at<cv::Vec3f>(row, col);
            const cv::Vec4f plane(normal(0), normal(1), normal(2), -normal.dot(X0));
            const PixelPatch& patch = frame.patch_map[row * frame.GetImageCols() + col];
            if(patch.sq0 > 0)
            {
                vector<pair<float,int>> score_neighbor;
                conf = ScorePixel(ref_id, cv::Point2i(col, row), normal, depth, patch, use_geometry, score_neighbor, plane);
            }
            #pragma omp critical
            {
                frame.conf_map.at<float>(row, col) = conf;
                if(conf <= -1)
                {
                    frame.depth_map.at<float>(row, col) = 0;
                    frame.normal_map.at<cv::Vec3f>(row, col) = cv::Vec3f(0,0,0);
                }
            }
        }
    }
    return true;
}

bool MVS::InitPatchMap(int ref_id, bool enable_parallel)
{
    Frame& frame = frames[ref_id];
    frame.patch_map = PatchMap(frame.GetImageRows() * frame.GetImageCols(), PixelPatch(num_texels));
    
    #pragma omp parallel if(enable_parallel)
    for(int row = 0; row < frame.GetImageRows(); row++)
    {
        for(int col = 0; col < frame.GetImageCols(); col++)
        {
            FillPixelPatch(ref_id, cv::Point2i(col, row), frame.patch_map[row * frame.GetImageCols() + col]);
        }
    }
    return true;
}

bool MVS::FillPixelPatch(int ref_id, const cv::Point2i& pt, PixelPatch& patch)
{
    const Frame& frame = frames[ref_id];
    
    const cv::Mat& img_gray = frame.GetImageGrayRef();
    if(!frame.IsInside(pt, config.ncc_half_window, config.ncc_half_window))
        return false;

    const uchar& center = img_gray.at<uchar>(pt);
    int k = 0;
    for (int row = pt.y - config.ncc_half_window; row <= pt.y + config.ncc_half_window; row += config.ncc_step)
    {
        for (int col = pt.x - config.ncc_half_window; col <= pt.x + config.ncc_half_window; col += config.ncc_step)
        {
            const uchar& texels = img_gray.at<uchar>(row, col);
            float wColor = (texels - center) / 255.f;
            wColor = wColor * wColor * sigma_color;

            float wSpatial = ((float)Square(col - pt.x) + (float)Square(row - pt.y)) * sigma_spatial;
            patch.weight[k] = std::exp(wColor + wSpatial);
            patch.texels0[k] = texels; // texels0存储当前图像中以x为中心的方形区域灰度值
            k++;
        }
    }
    float sum = accumulate(patch.weight.begin(), patch.weight.end(), 0.f);
    for(int i = 0; i < num_texels; i++)
        patch.weight[i] /= sum;
    sum = 0;
    for(int i = 0; i < num_texels; i++)
        sum += patch.weight[i] * patch.texels0[i];
    for(int i = 0; i < num_texels; i++)
        patch.texels0[i] -= sum;
    patch.sq0 = 0; 
    for(int i = 0; i < num_texels; i++)
    {
        float tmp = patch.texels0[i] * patch.weight[i];
        patch.sq0 += patch.texels0[i] * tmp;
        patch.texels0[i] = tmp;             // texels0 存储的是加权的灰度
    }
    
    if(patch.sq0 <= 1e-6)
        return false;
    return true;
}

bool MVS::EstimateDepthMapSingle(int ref_id, int propagate, int max_iter, float conf_threshold, bool use_geometry)
{
    // 进行传播
    for(int iter = 0; iter < max_iter; iter++)
    {
        if(propagate == Propagate::CHECKER_BOARD)
            PropagateCheckerBoard(ref_id, use_geometry);
        else if(propagate == Propagate::SEQUENTIAL)
            PropagateSequential(ref_id, iter, use_geometry);
        else 
        {
            LOG(ERROR) << "Propagate strategy is unsupport";
            return false;
        }
    }

    // 去除一致性程度低于阈值的点
	for (int i = 0; i < frames[ref_id].GetImageCols(); ++i)
	{
		for (int j = 0; j < frames[ref_id].GetImageRows(); ++j)
		{
			cv::Point2i pt(i, j);
            if(!frames[ref_id].depth_constant.empty() && frames[ref_id].depth_constant.at<uchar>(pt))
                continue;
			if (frames[ref_id].conf_map.at<float>(pt) < conf_threshold)
			{
				frames[ref_id].depth_map.at<float>(pt) = 0.f;
				frames[ref_id].conf_map.at<float>(pt) = -1.f;
				frames[ref_id].normal_map.at<cv::Vec3f>(pt) = cv::Vec3f(0,0,0);
			}
		}
	}

    return true;
    
}

// 把近邻像素的深度值和法向量传播给当前像素，如果近邻的结果更好，那么就把近邻的结果给当前像素
// 然后对深度值和法向量进行一定程度的扰动
bool MVS::ProcessPixel(int ref_id, const cv::Point2i& pt, const vector<cv::Point2i>& neighbor, 
                        const PixelPatch& patch,
                        const bool geometric_consistency
                        )
{
    bool keep_depth_constant = false;
    if(!frames[ref_id].depth_constant.empty())
        keep_depth_constant = frames[ref_id].depth_constant.at<uchar>(pt);

	float& depth = frames[ref_id].depth_map.at<float>(pt);
	cv::Vec3f& normal = frames[ref_id].normal_map.at<cv::Vec3f>(pt);
	float& conf = frames[ref_id].conf_map.at<float>(pt);
    // 离得最近的4个像素的信息
    vector<NeighborPixel> close_neighbors;
    vector<cv::Point2i> n_pt = {cv::Point2i(pt.x - 1, pt.y), cv::Point2i(pt.x, pt.y - 1), 
                                cv::Point2i(pt.x, pt.y + 1), cv::Point2i(pt.x + 1, pt.y)};
    for(const cv::Point2i& n : n_pt)
    {
        if(!frames[ref_id].IsInside(n))
            continue;
        const float& d = frames[ref_id].depth_map.at<float>(n);
        if(d <= 0)
            continue;
        // 保存的是邻居像素对应的三维点以及深度和法向量
        close_neighbors.push_back(NeighborPixel{eq.ImageToCam(n) * d, frames[ref_id].normal_map.at<cv::Vec3f>(n), d});
    }
    for (const cv::Point2i& nx : neighbor)
	{
		if (!frames[ref_id].IsInside(nx))
			continue;
		float depth_neighbor = frames[ref_id].depth_map.at<float>(nx);
        if (depth_neighbor <= 0)
            continue;
		cv::Vec3f normal_neighbor = frames[ref_id].normal_map.at<cv::Vec3f>(nx);
        depth_neighbor = keep_depth_constant ? depth : InterpolatePixel(pt, nx, depth_neighbor, normal_neighbor);
        CorrectNormal(pt, normal_neighbor);
        cv::Point3f X0 = eq.ImageToCam(pt) * depth_neighbor;
        // 平面参数 ax+by+cz+d=0 中的 a b c d
        cv::Vec4f plane(normal_neighbor(0), normal_neighbor(1), normal_neighbor(2), -normal_neighbor.dot(X0));
        vector<pair<float, int>> score_neighbor;
        const float newconf = ScorePixel(ref_id, pt, normal_neighbor, depth_neighbor, patch, geometric_consistency, score_neighbor, plane, close_neighbors);

        if (conf < newconf)
        {
            conf = newconf;
            depth = depth_neighbor;
            normal = normal_neighbor;
        }
	}
    PerturbDepthNormal3(ref_id, pt, patch, close_neighbors, geometric_consistency, !keep_depth_constant);
    return true;
}

float MVS::ScorePixel(int ref_id, const cv::Point2i& pt, const cv::Vec3f& normal, const float depth, 
                    const PixelPatch& patch_info,
                    const bool geometric_consistency,
                    std::vector<std::pair<float,int>>& score_neighbor,
                    const cv::Vec4f& plane,
                    const std::vector<NeighborPixel>& close_neighbors
                    )
{
    cv::Point3f X0 = eq.ImageToCam(pt) * depth;
    const float d = X0.dot(normal);
    if(d > 0)
        return -1;
    score_neighbor.clear();
    const Frame& frame = frames[ref_id];

    vector<float> texels1(num_texels);
    
    for(const NeighborInfo& neighbor : neighbors[ref_id])
    {
        const cv::Matx33f H = neighbor.R_nr + (1.f / d) * neighbor.t_nr * normal.t();
        int k = 0;
		float sq1 = 0, nrm, sq01 = 0, ncc, score, sum = 0;
        for (int i = 0; i < ncc_window_size; i += config.ncc_step)
		{
			for (int j = 0; j < ncc_window_size; j += config.ncc_step)
			{
                // 根据H计算参考图像上当前点的近邻点在邻域图像上的投影
                cv::Point3f X1;
                X1 = H * eq.ImageToCam(cv::Point2i(pt.x - config.ncc_half_window + j, pt.y - config.ncc_half_window + i));

#if 0
                // 不使用单应矩阵把三维点从参考图像变换到邻域图像，而是使用点到空间平面的交点，然后再把交点投影到邻域图像
                // 这个方法和单应矩阵其实是一样的，因为单应矩阵就是这么推导出来的
                // 但是这个可以处理空间平面和视线平行的情况，这种情况下视线（view ray）和空间平面的法向量垂直，此时理论上
                // 视线应该和空间平面无交点，那么自然也无法变换到邻域图像上。
                // 然而在使用单应矩阵的时候，即使无交点，也能得到一个数值解，很明显不太对
                cv::Point3f view_ray = eq.ImageToCam(cv::Point2i(pt.x - config.ncc_half_window + j, pt.y - config.ncc_half_window + i));
                const cv::Vec6f line(0,0,0, view_ray.x, view_ray.y, view_ray.z);
                cv::Point3f point_inersect = PlaneLineIntersect<float, cv::Point3f>(plane.val, line.val);
                X1 = neighbor.R_nr * point_inersect + cv::Point3f(neighbor.t_nr);
                if(isinf(X1.x) || isnan(X1.x))
                    goto next_image;
#endif

                cv::Point2f x1 = eq.CamToImage(X1);
                if(!frame.IsInside(x1, 1, 1))
                    goto next_image;
                texels1[k] = Sample(frames[neighbor.id].GetImageGrayRef(), x1);
				k++;
			}
		}
        
        for(int i = 0; i < num_texels; i++)
            sum += texels1[i] * patch_info.weight[i];
        for(int i = 0; i < num_texels; i++)
            texels1[i] -= sum;
        for(int i = 0; i < num_texels; i++)
            sq1 += texels1[i] * texels1[i] * patch_info.weight[i];
        
        nrm = patch_info.sq0 * sq1;
        for(int i = 0; i < num_texels; i++)
            sq01 += patch_info.texels0[i] * texels1[i];
        
       
		if (nrm <= 0.f) // 如nrm过小，则NCC分母过小，不再进行NCC计算
		{
            continue;
        }
		ncc = sq01 / sqrt(nrm); // 计算NCC
        score = std::min(std::max(ncc, -1.f), 1.f);     // 将NCC值转换到[-1,1]，值越大一致性越高
        if(!close_neighbors.empty())
        {
            score = 1 - score;  // 变成[0-2]，0最好 2最差
            for(const NeighborPixel& n : close_neighbors)
            {
                float diff_distance = PointToPlaneDistance(plane, n.point, true) / depth;
                const float factorDepth = exp(diff_distance * diff_distance * smoothSigmaDepth);
                float diff_angle = VectorAngle3D(normal.val, n.normal.val, true);
                const float factorNormal = exp(diff_angle * diff_angle * smoothSigmaNormal);
                score *= (1.f - smoothBonusDepth * factorDepth) * (1.f - smoothBonusNormal * factorNormal);
            }
            score = 1 - score;  // 变回[-1,1], -1最差 1最好
            score = min(1.f, max(-1.f, score));
        }
        if(geometric_consistency)
        {
            const Frame& n_frame = frames[neighbor.id];
            // 注意：这里的 depth filter 存储的并不是经过过滤的深度图，而是不用几何一致性（即仅依靠光度一致性）算出的深度图。
            // geometric_consistency = true 代表的是利用邻域图像的深度图对评分进行调整，使其在邻域图像上达到几何一致（深度一致）。
            // 这种情况下使用的深度图是仅靠光度一致性得到的结果
            assert(!n_frame.depth_filter.empty());
            float geometric_weight = 0.2, consistency = 2;
            cv::Point3f X1 = neighbor.R_nr * X0 + cv::Point3f(neighbor.t_nr);
            const float depth0 = cv::norm(X1);
            cv::Point2f x1 = eq.CamToImage(X1);
            score = 1 - score;  // 变成[0-2]，0最好 2最差
            if(n_frame.IsInside(x1, 1, 1)) 
            {
                float depth1 = Sample(n_frame.depth_filter, x1, 
                            [&depth0](const float& d){return abs(depth0 - d) / depth0 < 0.03f;}
                            );
                if(!isinf(depth1))
                {
                    cv::Matx33f R_rn = neighbor.R_nr.t();
                    cv::Point3f t_rn = cv::Point3f(-R_rn * neighbor.t_nr);
                    cv::Point3f X0_back = R_rn * eq.ImageToCam(x1, depth1) + t_rn;
                    #if 0
                    cv::Point2f x0_back = eq.CamToImage(X0_back);
                    // 计算投影回来的点和当前点之间的距离
                    const float dist = sqrt(Square(x0_back.x - float(pt.x)) + Square(x0_back.y - float(pt.y))) ;
                    consistency = min(sqrt(dist * (dist + 2.f)), consistency);
                    #else 
                    // 计算投影回来的点和当前点之间的夹角
                    const float diff_angle = VectorAngle3D(X0, X0_back, false) * 180.0 / M_PI;
                    consistency = min(diff_angle, consistency);
                    #endif 
                }
            }
            score += geometric_weight * consistency;
            score = 1 - score;  // 变回[-1,1], -1最差 1最好
            score = min(1.f, max(-1.f, score));
            
        }
        
        // 把当前邻域的得分和邻域的id保存下来
        score_neighbor.push_back({score, neighbor.id});
        next_image:;
    }

    if(score_neighbor.size() == 0)
        return -1;
    else if(score_neighbor.size() == 1)
        return score_neighbor[0].first;
    else 
    {
        float avg_score = 0;
        // 找到排第2个的元素
        sort(score_neighbor.begin(), score_neighbor.end(), 
            [](const pair<float,int>& a, const pair<float,int>& b){return a.first > b.first;});
        int count = 0, max_count = 2;
        for(const auto& p : score_neighbor)
        {
            if(count >= max_count)
                break;
            avg_score += p.first;
            count++;
        }
        return avg_score / count;
    }
}

float MVS::ScorePixelSphere(int ref_id, const cv::Point2i& pt, const cv::Vec3f& normal, const float depth, 
                        std::vector<std::pair<float,int>>& score_neighbor,
                        const cv::Vec4f& plane, const vector<NeighborPixel>& close_neighbors
                        )  
{
    cv::Point3f X0 = eq.ImageToCam(pt) * depth;
    const float d = X0.dot(normal);
    if(d > 0)
        return -1;
    score_neighbor.clear();
    const Frame& frame = frames[ref_id];
    Eigen::Array<float, Eigen::Dynamic, 1> texelsMean(num_texels), texelsTmp(num_texels);
    cv::Point2f pt_sphere = eq.ImageToSphere(cv::Point2f(pt));
    // 配置文件里的half window指的是像素大小，乘以0.1化作角度，然后再变成弧度
    float half_window_sphere = config.ncc_half_window * 0.1 * M_PI / 180.f;        

    const cv::Mat& img_gray = frame.GetImageGrayRef();
    const cv::Point2f lt0(pt_sphere.x - half_window_sphere, pt_sphere.y - half_window_sphere); // 以x为中心的图像块左上角
    const cv::Point2f rb0(pt_sphere.x + half_window_sphere, pt_sphere.y + half_window_sphere); // 以x为中心的图像块右下角
    
    Eigen::Array<float, Eigen::Dynamic, 1> texels0(num_texels), weight(num_texels);
    uchar center = img_gray.at<uchar>(pt);
    int k = 0;
    for (int m = 0; m < ncc_window_size; m += config.ncc_step)
    {
        for (int n = 0; n < ncc_window_size; n += config.ncc_step)
        {
            cv::Point2f ptn(lt0.x + n * 0.1 * M_PI / 180.f, lt0.y + m * 0.1 * M_PI / 180.f);
            cv::Point2f ptn_pixel = eq.SphereToImage(ptn);
            if(!frame.IsInside(ptn_pixel, 1, 1))
                return -1;
            texels0(k) = Sample(img_gray, ptn_pixel);   // texels0存储当前图像中以x为中心的方形区域灰度值
            float wColor = (texels0(k) - center) / 255.f;

            wColor = wColor * wColor * sigma_color;
            cv::Point2f ptd((float)ptn.x - (float)pt_sphere.x, (float)ptn.y - (float)pt_sphere.y);
            ptd *= 180.f / M_PI;
            float wSpatial = (ptd.x * ptd.x * 100 + ptd.y * ptd.y * 100) * sigma_spatial;
            weight(k) = std::exp(wColor + wSpatial);
            k++;
        }
    }
    weight /= weight.sum();

    texelsMean = texels0 * weight;
    texels0 -= texelsMean.sum();
    texelsTmp = texels0 * texels0 * weight;
    float sq0 = texelsTmp.sum();
    if (sq0 <= 0.f) // 如sq0过小，不再进行NCC计算
        return -1;Eigen::Array<float, Eigen::Dynamic, 1> texels1(num_texels);
    
    for(const NeighborInfo& neighbor : neighbors[ref_id])
    {
        const cv::Matx33f H = neighbor.R_nr + (1.f / d) * neighbor.t_nr * normal.t();
        int k = 0;
		float sq1, nrm, sq01, ncc, score;
        for (int i = 0; i < ncc_window_size; i += config.ncc_step)
		{
			for (int j = 0; j < ncc_window_size; j += config.ncc_step)
			{
                // 根据H计算参考图像上当前点的近邻点在邻域图像上的投影
                cv::Point3f X1;
                cv::Point3f view_ray = eq.SphereToCam(
                        cv::Point2f(pt_sphere.x - half_window_sphere + j * 0.1 * M_PI / 180.f, 
                                pt_sphere.y - half_window_sphere + i * 0.1 * M_PI / 180.f), 1.f);
                const cv::Vec6f line(0,0,0, view_ray.x, view_ray.y, view_ray.z);
                cv::Point3f point_inersect = PlaneLineIntersect<float, cv::Point3f>(plane.val, line.val);
                X1 = neighbor.R_nr * point_inersect + cv::Point3f(neighbor.t_nr);
               
                cv::Point2f x1 = eq.CamToImage(X1);
                if(!frame.IsInside(x1, 1, 1))
                    goto next_image;
                texels1(k) = Sample(frames[neighbor.id].GetImageGrayRef(), x1);
				k++;
			}
		}
        texelsMean = texels1 * weight;
		texels1 -= texelsMean.sum();
		texelsTmp = texels1 * texels1 * weight;
		sq1 = texelsTmp.sum();
		// if (sq1 < 0.01) // 如sq1过小，表明邻域图像对应区域是弱纹理区域，不再进行NCC计算
		// 	continue;
		nrm = sq0 * sq1;
		texelsTmp = texels0 * texels1 * weight;
		sq01 = texelsTmp.sum();
		if (nrm <= 0.f) // 如nrm过小，则NCC分母过小，不再进行NCC计算
			continue;
		ncc = sq01 / sqrt(nrm); // 计算NCC
        score = std::min(std::max(ncc, -1.f), 1.f);     // 将NCC值转换到[-1,1]，值越大一致性越高
        if(!close_neighbors.empty())
        {
            score = 1 - score;  // 变成[0-2]，0最好 2最差
            for(const NeighborPixel& n : close_neighbors)
            {
                float diff_distance = PointToPlaneDistance(plane, n.point, true) / depth;
                const float factorDepth = exp(diff_distance * diff_distance * smoothSigmaDepth);
                float diff_angle = VectorAngle3D(normal.val, n.normal.val, true);
                const float factorNormal = exp(diff_angle * diff_angle * smoothSigmaNormal);
                score *= (1.f - smoothBonusDepth * factorDepth) * (1.f - smoothBonusNormal * factorNormal);
            }
            score = 1 - score;  // 变回[-1,1], -1最差 1最好
        }
        else 
            score = std::min(std::max(ncc, -1.f), 1.f); // 将NCC值转换到[-1,1]，值越大一致性越高
        
        // 把当前邻域的得分和邻域的id保存下来
        score_neighbor.push_back({score, neighbor.id});
        next_image:;
    }
    
    if(score_neighbor.size() == 0)
        return -1;
    else if(score_neighbor.size() == 1)
        return score_neighbor[0].first;
    else 
    {
        float avg_score = 0;
        // 找到排第2个的元素
        sort(score_neighbor.begin(), score_neighbor.end(), 
            [](const pair<float,int>& a, const pair<float,int>& b){return a.first > b.first;});
        int count = 0, max_count = 2;
        for(const auto& p : score_neighbor)
        {
            if(count >= max_count)
                break;
            avg_score += p.first;
            count++;
        }
        return avg_score / count;
    }
}

bool MVS::PropagateSequential(int ref_id, const int iter, const bool geometric_consistency)
{
    Frame& frame = frames[ref_id];
    // 从左上到右下传播
    if(iter % 2 == 0)
        for(int row = 0; row < frame.GetImageRows(); row++)
        {
            for(int col = 0; col < frame.GetImageCols(); col++)
            {
                cv::Point2i pt(col, row);
                if(frame.depth_map.at<float>(pt) <= 0)
                    continue;
                const PixelPatch& patch = frame.patch_map[row * frame.GetImageCols() + col];
                if(patch.sq0 <= 0)
                    continue;

                vector<cv::Point2i> neighbor_pts = {cv::Point2i(col - 1, row), cv::Point2i(col, row - 1)};
                ProcessPixel(ref_id, pt, neighbor_pts, patch, geometric_consistency);
                
            }
        }
    // 从右下到左上传播
    else 
        for(int row = frame.GetImageRows() - 1; row >= 0 ; row--)
        {
            for(int col = frame.GetImageCols() - 1; col >= 0; col--)
            {
                cv::Point2i pt(col, row);
                if(frame.depth_map.at<float>(row, col) <= 0)
                    continue;
                const PixelPatch& patch = frame.patch_map[row * frame.GetImageCols() + col];
                if(patch.sq0 <= 0)
                    continue;

                vector<cv::Point2i> neighbor_pts = {cv::Point2i(col + 1, row), cv::Point2i(col, row + 1)};
                ProcessPixel(ref_id, pt, neighbor_pts, patch, geometric_consistency);
            }
        }
    return true;
}

bool MVS::PropagateCheckerBoard(int ref_id, const bool geometric_consistency)
{
    Frame& frame = frames[ref_id];
    // 把传播过程分解成黑白的棋盘格，分成两次，第一次只用黑色格子，第二次只用白色格子
    // 黑色格子的点的邻域从白色格子选择，白色格子的点就从黑色格子选
    for(int offset = 0; offset <= 1; offset++)
    {
        for(int row = 0; row < frames[0].GetImageRows(); row++)
        {
            #pragma omp parallel for schedule(dynamic)
            for(int col = (row % 2 + offset) % 2; col < frames[0].GetImageCols(); col += 2)
            {
                cv::Point2i pt(col, row);
                if(frame.depth_map.at<float>(row, col) <= 0)
                    continue;
                vector<cv::Point2i> neighbor_pts = {cv::Point2i(col - 1, row), cv::Point2i(col, row - 1), 
                                                    cv::Point2i(col + 1, row), cv::Point2i(col, row + 1)};
                const PixelPatch& patch = frame.patch_map[row * frame.GetImageCols() + col];
                if(patch.sq0 <= 1e-6)
                    continue;
                
                ProcessPixel(ref_id, pt, neighbor_pts, patch, geometric_consistency);
                // ProcessPixel(ref_id, pt, CheckerBoardSampling(frame, pt, 4),
                //                  frame.depth_map, frame.normal_map, frame.conf_map);

            }
        }
    }
    return true;
}

# if 0
bool MVS::PerturbDepthNormal(int ref_id, const cv::Point2i& pt, const bool geometric_consistency)
{
    struct Hypothese
    {
        float score;
        float depth;
        cv::Vec3f normal;
        Hypothese(const float& d, const cv::Vec3f& n):depth(d),normal(n){}
    };
    Frame& frame = frames[ref_id];
    float perturb_scale = 1;
    cv::Point3f view_ray = eq.ImageToCam(pt);
    cv::Vec3f& normal_origin = frame.normal_map.at<cv::Vec3f>(pt);
    float& depth_origin = frame.depth_map.at<float>(pt);
    float& conf_origin = frame.conf_map.at<float>(pt);
    
    for(int iter = 0; iter < 3; iter++)
    {
        // 对法向量进行扰动，如果扰动后方向不是指向相机的，就再扰动一次，最多只能扰动三次
        // 每次扰动程度都降低。如果三次都不行，那就仍然保留这种
        cv::Vec3f normal_perturb = PerturbNormal(normal_origin, perturb_scale * M_PI);
        if(normal_perturb.dot(view_ray) > 0)
            normal_perturb = PerturbNormal(normal_origin, perturb_scale / 2.0 * M_PI);
        if(normal_perturb.dot(view_ray) > 0)
            normal_perturb = PerturbNormal(normal_origin, perturb_scale / 4.0 * M_PI);
        // if(normal_perturb.dot(view_ray) > 0)
        //     continue;
        float depth_perturb = PerturbDepth(depth_origin, perturb_scale);
        cv::Vec3f normal_random = GenerateRandomNormal(pt);
        float depth_random = rng.uniform(depth_origin / 2.f, config.max_depth);
        vector<Hypothese> hypotheses = {Hypothese(depth_origin, normal_perturb),
                                        Hypothese(depth_perturb, normal_origin),
                                        Hypothese(depth_perturb, normal_perturb)};
                                        // Hypothese(depth_origin, normal_random),
                                        // Hypothese(depth_perturb, normal_random),
                                        // Hypothese(depth_random, normal_origin),
                                        // Hypothese(depth_random, normal_perturb),
                                        // Hypothese(depth_random, normal_random)
                                        // };
        for(Hypothese& h : hypotheses)
        {
            vector<pair<float, int>> score_neighbor;
            h.score = ScorePixel(ref_id, pt, h.normal, h.depth, geometric_consistency, score_neighbor);
        }
        sort(hypotheses.begin(), hypotheses.end(), [](const Hypothese& h1, const Hypothese& h2){return h1.score > h2.score;});
        
        if(hypotheses[0].score > conf_origin)
        {
            conf_origin = hypotheses[0].score;
            normal_origin = hypotheses[0].normal;
            depth_origin = hypotheses[0].depth;
            perturb_scale /= 2.0;
        }
        
    }
            
    return true;
}

bool MVS::PerturbDepthNormal2(int ref_id, const cv::Point2i& pt, const bool geometric_consistency)
{
    Frame& frame = frames[ref_id];
    cv::Vec3f& normal_origin = frame.normal_map.at<cv::Vec3f>(pt);
    float& depth_origin = frame.depth_map.at<float>(pt);
    float& conf_origin = frame.conf_map.at<float>(pt);
    cv::Point3f view_ray = eq.ImageToCam(pt);

    float thConfSmall(0.55 * 0.2f);
	float thConfBig(0.55 * 0.4f);
	float thConfRand(0.55 * 0.9f);
	float thRobust(0.55 * 1.2f);
    unsigned idxScaleRange(0);
    if(1-conf_origin <= thConfSmall)
        idxScaleRange = 2;
    else if(1-conf_origin <= thConfBig)
        idxScaleRange = 1;
    else if(1-conf_origin >= thConfRand)
    {
        for(int iter = 0; iter < 6; iter++)
        {
            float depth_random = rng.uniform(config.min_depth, config.max_depth);
            cv::Vec3f normal_random = GenerateRandomNormal(pt);
            vector<pair<float, int>> score_neighbor;
            float nconf = ScorePixel(ref_id, pt, normal_random, depth_random, geometric_consistency, score_neighbor);
            if(nconf > conf_origin)
            {
                conf_origin = nconf;
                depth_origin = depth_random;
                normal_origin = normal_random;
                if(1 - nconf < thConfRand)
                    goto Refine;
            }
        }
        return false;
    }
    Refine:
    float scaleRange(scaleRanges[idxScaleRange]);
    float depthRange = depth_origin * 0.02;
    float angleRange = 30.f / 180.f * M_PI;
    for(int iter = 0; iter < 6; iter++)
    {
        float depth_perturb = PerturbDepth(depth_origin, scaleRange * depthRange);
        cv::Vec3f normal_perturb = PerturbNormal(normal_origin, scaleRange * angleRange);

        if(normal_perturb.dot(view_ray) >= 0)
            continue;
        cv::Point3f X0 = view_ray * depth_perturb;
        // 平面参数 ax+by+cz+d=0 中的 a b c d
        cv::Vec4f plane(normal_perturb(0), normal_perturb(1), normal_perturb(2), -normal_perturb.dot(X0));
        vector<pair<float, int>> score_neighbor;
        float nconf = ScorePixel(ref_id, pt, normal_perturb, depth_perturb, geometric_consistency, score_neighbor, plane);
        if(nconf > conf_origin)
        {
            conf_origin = nconf;
            depth_origin = depth_perturb;
            normal_origin = normal_perturb;
            idxScaleRange++;
            scaleRange = scaleRanges[idxScaleRange];
        }
    }
    return true;
}
#endif

bool MVS::PerturbDepthNormal3(int ref_id, const cv::Point2i& pt, const PixelPatch& patch, const vector<NeighborPixel>& neighbor_pixels,
                const bool geometric_consistency, const bool perturb_depth, const bool perturb_normal)
{
    Frame& frame = frames[ref_id];
    cv::Vec3f& normal_origin = frame.normal_map.at<cv::Vec3f>(pt);
    float& depth_origin = frame.depth_map.at<float>(pt);
    float& conf_origin = frame.conf_map.at<float>(pt);
    cv::Point3f view_ray = eq.ImageToCam(pt);

    float thConfSmall(0.55 * 0.2f);
	float thConfBig(0.55 * 0.4f);
	float thConfRand(0.55 * 0.9f);
	float thRobust(0.55 * 1.2f);
    unsigned idxScaleRange(0);
    if(1-conf_origin <= thConfSmall)
        idxScaleRange = 2;
    else if(1-conf_origin <= thConfBig)
        idxScaleRange = 1;
    else if(1-conf_origin >= thConfRand)
    {
        for(int iter = 0; iter < 6; iter++)
        {
            float depth_random = perturb_depth ? rng.uniform(config.min_depth, config.max_depth) : depth_origin;
            cv::Vec3f normal_random = perturb_normal ? GenerateRandomNormal(pt) : normal_origin;
            vector<pair<float, int>> score_neighbor;
            float nconf = ScorePixel(ref_id, pt, normal_random, depth_random, patch, geometric_consistency, score_neighbor);
            if(nconf > conf_origin)
            {
                conf_origin = nconf;
                depth_origin = depth_random;
                normal_origin = normal_random;
                if(1 - nconf < thConfRand)
                    goto Refine;
            }
        }
        return false;
    }
    Refine:

    float scaleRange(scaleRanges[idxScaleRange]);
    float depthRange = depth_origin * 0.02;
    float angleRange = 30.f / 180.f * M_PI;
    for(int iter = 0; iter < 6; iter++)
    {
        float depth_perturb = perturb_depth ? PerturbDepth(depth_origin, scaleRange * depthRange) : depth_origin;
        cv::Vec3f normal_perturb = perturb_normal ? PerturbNormal(normal_origin, scaleRange * angleRange) : normal_origin;
        if(normal_perturb.dot(view_ray) >= 0)
            continue;
        cv::Point3f X0 = view_ray * depth_perturb;
        // 平面参数 ax+by+cz+d=0 中的 a b c d
        cv::Vec4f plane(normal_perturb(0), normal_perturb(1), normal_perturb(2), -normal_perturb.dot(X0));
        vector<pair<float, int>> score_neighbor;
        float nconf = ScorePixel(ref_id, pt, normal_perturb, depth_perturb, patch, geometric_consistency, score_neighbor, plane, neighbor_pixels);
        if(nconf > conf_origin)
        {
            conf_origin = nconf;
            depth_origin = depth_perturb;
            normal_origin = normal_perturb;
            idxScaleRange++;
            scaleRange = scaleRanges[idxScaleRange];
        }
    }
    return true;
}

vector<cv::Point2i> MVS::CheckerBoardSampling(const Frame& frame, const cv::Point2i& pt, const int neighbor_size)
{
    vector<cv::Point2i> candidate;
    // 上下左右四个临近的像素
    candidate.push_back(cv::Point2i(pt.x - 1, pt.y));
    candidate.push_back(cv::Point2i(pt.x + 1, pt.y));
    candidate.push_back(cv::Point2i(pt.x, pt.y - 1));
    candidate.push_back(cv::Point2i(pt.x, pt.y + 1));
    // 四个V形区域的其他像素
    for (int i = 2; i < 5; i++)
    {
        // 左侧V形区域
        candidate.push_back(cv::Point2i(pt.x - i, pt.y + i - 1));
        candidate.push_back(cv::Point2i(pt.x - i, pt.y - i + 1));
        // 右侧V形区域
        candidate.push_back(cv::Point2i(pt.x + i, pt.y + i - 1));
        candidate.push_back(cv::Point2i(pt.x + i, pt.y - i + 1));
        // 上方V形区域
        candidate.push_back(cv::Point2i(pt.x + i - 1, pt.y - i));
        candidate.push_back(cv::Point2i(pt.x - i + 1, pt.y - i));
        // 下方V形区域
        candidate.push_back(cv::Point2i(pt.x + i - 1, pt.y + i));
        candidate.push_back(cv::Point2i(pt.x - i + 1, pt.y + i));
    }
    // 四个长条区域
    for (int i = 3; i < 25; i += 2)
    {
        candidate.push_back(cv::Point2i(pt.x, pt.y - i)); // 上
        candidate.push_back(cv::Point2i(pt.x, pt.y + i)); // 下
        candidate.push_back(cv::Point2i(pt.x - i, pt.y)); // 左
        candidate.push_back(cv::Point2i(pt.x + i, pt.y)); // 右
    }

    vector<pair<float, cv::Point2i>> score_neighbor;
    for(const cv::Point2i& p : candidate)
    {
        if(frame.depth_map.at<float>(p) <= 0)
            continue;
        score_neighbor.push_back({frame.conf_map.at<float>(p), p});
    }
    // 按照pair中的第一个数字降序排列，也就是按照分值排列
    sort(score_neighbor.begin(), score_neighbor.end(), 
            [](const pair<float, cv::Point2i>& p1, const pair<float, cv::Point2i>& p2){return p1.first > p2.first;});
    candidate.clear();
    for(int i = 0; i < score_neighbor.size() && candidate.size() < neighbor_size; i++)
        candidate.push_back(score_neighbor[i].second);
    return candidate;
}

cv::Vec3f MVS::PerturbNormal(const cv::Vec3f& normal, const float perturbation)
{
    const float a1 = (rng.operator float() - 0.5f) * perturbation;
    const float a2 = (rng.operator float() - 0.5f) * perturbation;
    const float a3 = (rng.operator float() - 0.5f) * perturbation;
    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    // R = Rx * Ry * Rz
    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = -cos_a2 * sin_a3;
    R[2] = sin_a2;
    R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
    R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
    R[5] = -cos_a2 * sin_a1;
    R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
    R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
    R[8] = cos_a1 * cos_a2;

    cv::Vec3f prt_normal(R[0] * normal[0] + R[1] * normal[1] + R[2] * normal[2],
                         R[3] * normal[0] + R[4] * normal[1] + R[5] * normal[2],
                         R[6] * normal[0] + R[7] * normal[1] + R[8] * normal[2]);
    return prt_normal;
}

float MVS::PerturbDepth(const float depth, const float perturbation)
{
    float max_depth = (1 + perturbation) * depth;
    float min_depth = (1 - perturbation) * depth;
    return rng.uniform(0.f, 1.f) * (max_depth - min_depth) + min_depth;
}

cv::Vec3f MVS::GenerateRandomNormal(const cv::Point2i& pt)
{
    float v1 = 0.0f;
    float v2 = 0.0f;
    float s = 2.0f;
    while (s >= 1.0f)
    {
        v1 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
        v2 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
        s = v1 * v1 + v2 * v2;
    }
    cv::Vec3f normal(0, 0, 0);
    const float s_norm = sqrt(1.0f - s);
    normal[0] = 2.0f * v1 * s_norm;
    normal[1] = 2.0f * v2 * s_norm;
    normal[2] = 1.0f - 2.0f * s;

    // Make sure normal is looking away from camera.
    cv::Point3f view_ray = eq.ImageToCam(pt);
    if (normal.dot(view_ray) > 0)
    {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
    return normal;
}

float MVS::Sample(const cv::Mat& img_gray, const cv::Point2f& pt)
{
    const int lx = (int)pt.x;
    const int ly = (int)pt.y;
    const float x = pt.x - lx;
    const float y = pt.y - ly;
    const float x1 = 1.f - x;
    const float y1 = 1.f - y;
    return (img_gray.at<uchar>(ly, lx) * x1 + img_gray.at<uchar>(ly, lx + 1) * x) * y1 + 
            (img_gray.at<uchar>(ly + 1, lx) * x1 + img_gray.at<uchar>(ly + 1, lx + 1) * x) * y;
}

template<typename Functor>
float MVS::Sample(const cv::Mat& img, const cv::Point2f& pt, const Functor& functor)
{
    cv::Mat_<float> a;
    const int lx = (int)pt.x;
    const int ly = (int)pt.y;
    const float x = pt.x - lx;
    const float y = pt.y - ly;
    const float x1 = 1.f - x;
    const float y1 = 1.f - y;
    const float x0y0 = img.at<float>(ly, lx);
    const float x1y0 = img.at<float>(ly, lx + 1);
    const float x0y1 = img.at<float>(ly + 1, lx);
    const float x1y1 = img.at<float>(ly + 1, lx + 1);
    const bool b00(functor(x0y0));
	const bool b10(functor(x1y0));
	const bool b01(functor(x0y1));
	const bool b11(functor(x1y1));
    if(!b00 && !b10 && !b01 && !b11)
        return std::numeric_limits<float>::infinity();
    return float(y1*(x1 * (b00 ? x0y0 : (b10 ? x1y0 : (b01 ? x0y1 : x1y1))) + x * (b10 ? x1y0 : (b00 ? x0y0 : (b11 ? x1y1 : x0y1)))) +
		   y *(x1 * (b01 ? x0y1 : (b11 ? x1y1 : (b00 ? x0y0 : x1y0))) + x * (b11 ? x1y1 : (b01 ? x0y1 : (b10 ? x1y0 : x0y0)))));
}


void MVS::FinishEstimation(int ref_id, bool geometric)
{
    // 随着程序运行，逐渐释放灰度图以及深度图占用的内存
    vector<int> ids = {ref_id};
    for(const NeighborInfo& n : neighbors[ref_id])
        ids.push_back(n.id);
    for(const int id : ids)
    {
        lock_guard<mutex> guard(frame_mutex[id]);
        frame_gray_count[id] -= 1;
        if(frame_gray_count[id] <= 0)
            frames[id].ReleaseImageGray();
    }

    PatchMap().swap(frames[ref_id].patch_map);
    
    ExportOpenCVMat(config.mvs_depth_path + "/" + num2str(ref_id) + (geometric ? "_geo.bin" : "_pho.bin"), frames[ref_id].depth_map);       
    ExportOpenCVMat(config.mvs_normal_path + "/" + num2str(ref_id) + (geometric ? "_geo.bin" : "_pho.bin"), frames[ref_id].normal_map);
    ExportConfMap(config.mvs_conf_path + "/" + num2str(ref_id) + (geometric ? "_geo.bin" : "_pho.bin"), frames[ref_id].conf_map);
    
    lock_guard<mutex> guard(frame_mutex[ref_id]);
    frame_conf_count[ref_id] -= 1;
    if(frame_conf_count[ref_id] <= 0 && !frames[ref_id].conf_map.empty())
        frames[ref_id].conf_map.release();

    frame_normal_count[ref_id] -= 1;
    if(frame_normal_count[ref_id] <= 0 && !frames[ref_id].normal_map.empty())
        frames[ref_id].normal_map.release();

    frame_depth_count[ref_id] -= 1;
    if(frame_depth_count[ref_id] <= 0 && !frames[ref_id].depth_map.empty())
        frames[ref_id].depth_map.release();
}

void MVS::RemoveSmallSegments(int ref_id)
{
    Frame& frame = frames[ref_id];
    cv::Mat done_map = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_8U); // done_map存放以访问过的像素点
	vector<cv::Point2i> seg_list(frame.GetImageRows() * frame.GetImageCols()); // seg_list存放相邻且深度值类似的像素点
	unsigned seg_list_count;
	unsigned seg_list_curr;
	vector<cv::Point2i> neighbor(4);
	int removeNum = 0;

	for (int u = 0; u < frame.GetImageCols(); u++)
	{
		for (int v = 0; v < frame.GetImageRows(); v++)
		{
			// 如果当前像素已经访问过，则继续
			if (done_map.at<uchar>(v, u))
				continue;

			// 将当前像素添加为seg_list中第一个元素
			seg_list[0] = cv::Point2i(u, v);
			seg_list_count = 1;
			seg_list_curr = 0;

			// 从当前像素开始寻找周边位置连续且深度值类似的像素点
			while (seg_list_curr < seg_list_count)
			{
				const cv::Point2i addr_curr(seg_list[seg_list_curr]);

				// 寻找四邻域
				neighbor[0] = cv::Point2i(addr_curr.x - 1, addr_curr.y);
				neighbor[1] = cv::Point2i(addr_curr.x + 1, addr_curr.y);
				neighbor[2] = cv::Point2i(addr_curr.x, addr_curr.y - 1);
				neighbor[3] = cv::Point2i(addr_curr.x, addr_curr.y + 1);

				const float& depth_curr = frame.depth_map.at<float>(addr_curr);
				for (int i = 0; i < 4; i++)
				{
					const cv::Point2i& addr_neighbor(neighbor[i]);
					if (frame.IsInside(addr_neighbor))
					{
						// 判断addr_neighbor是否已添加到seg_list中
						uchar& done = done_map.at<uchar>(addr_neighbor);
						if (!done)
						{
							// 如果深度值类似则将addr_neighbor添加到seg_list中
							const float& depth_neighbor = frame.depth_map.at<float>(addr_neighbor);
							if (depth_neighbor > 0 && abs((depth_curr - depth_neighbor) / depth_curr) < config.depth_diff_threshold)
							{
								seg_list[seg_list_count++] = addr_neighbor;
								done = 1;
							}
						}
					}
				}
				++seg_list_curr;
				done_map.at<uchar>(addr_curr) = true;
			}

			// 如果seg_list面积太小则将seg_list中所有像素的深度值去除
			if (seg_list_count < config.min_segment)
			{
				for (unsigned i = 0; i < seg_list_count; i++)
				{
					frame.depth_map.at<float>(seg_list[i]) = 0;
					frame.normal_map.at<cv::Vec3f>(seg_list[i]) = cv::Vec3f(0,0,0);
					frame.conf_map.at<float>(seg_list[i]) = -1;
					removeNum++;
				}
			}
		}
	}

	// LOG(INFO) << "Remove " << removeNum << " isolated points in frame " << frame.id;
}

// try to fill small gaps in the depth map
void MVS::GapInterpolation(int ref_id)
{
	const float fDepthDiffThreshold(config.depth_diff_threshold * 2.5f);
	unsigned nIpolGapSize = 10;
	cv::Mat &depthMap = frames[ref_id].depth_map;
	cv::Mat &normalMap = frames[ref_id].normal_map;
	cv::Mat &confMap = frames[ref_id].conf_map;
	const cv::Point2i size(frames[ref_id].GetImageCols(), frames[ref_id].GetImageRows());
	int interpNum = 0;

	// 1. Row-wise:
	// for each row do
	for (int v = 0; v < size.y; ++v)
	{
		// init counter
		unsigned count = 0;

		// for each element of the row do
		for (int u = 0; u < size.x; ++u)
		{
			// get depth of this location
			const float &depth = depthMap.at<float>(v, u);

			// if depth not valid => count and skip it
			if (depth <= 0)
			{
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)u > count)
			{
				// first value index for interpolation
				int u_curr = u - count;
				const int u_first = u_curr - 1;
				// compute mean depth
				const float &depthFirst = depthMap.at<float>(v, u_first);
				if (abs((depthFirst - depth) / depthFirst) < fDepthDiffThreshold)
				{
					// interpolate values
					const float diff = (depth - depthFirst) / (count + 1);
					float d = depthFirst;
					const float c = confMap.empty() ? -1.f : std::min(confMap.at<float>(v, u_first), confMap.at<float>(v, u));
					if (normalMap.empty())
					{
						do
						{
							depthMap.at<float>(v, u_curr) = (d += diff);
							interpNum++;
							if (!confMap.empty())
								confMap.at<float>(v, u_curr) = c;
						} while (++u_curr < u);
					}
					else
					{
						cv::Point2f dir1, dir2;
						dir1 = Normal2Dir(normalMap.at<cv::Vec3f>(v, u_first));
						dir2 = Normal2Dir(normalMap.at<cv::Vec3f>(v, u));
						const cv::Point2f dirDiff((dir2 - dir1) / float(count + 1));
						do
						{
							depthMap.at<float>(v, u_curr) = (d += diff);
							dir1 += dirDiff;							
							normalMap.at<cv::Vec3f>(v, u_curr) = Dir2Normal(dir1);
							interpNum++;
							if (!confMap.empty())
								confMap.at<float>(v, u_curr) = c;
						} while (++u_curr < u);
					}
				}
			}

			// reset counter
			count = 0;
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int u = 0; u < size.x; ++u)
	{

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v = 0; v < size.y; ++v)
		{
			// get depth of this location
			const float &depth = depthMap.at<float>(v, u);

			// if depth not valid => count and skip it
			if (depth <= 0)
			{
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count)
			{
				// first value index for interpolation
				int v_curr = v - count;
				const int v_first = v_curr - 1;
				// compute mean depth
				const float &depthFirst = depthMap.at<float>(v_first, u);
				if (abs((depthFirst - depth) / depthFirst) < fDepthDiffThreshold)
				{
					// interpolate values
					const float diff = (depth - depthFirst) / (count + 1);
					float d = depthFirst;
					const float c = confMap.empty() ? -1.f : std::min(confMap.at<float>(v_first, u), confMap.at<float>(v, u));
					if (normalMap.empty())
					{
						do
						{
							depthMap.at<float>(v_curr, u) = (d += diff);
							interpNum++;
							if (!confMap.empty())
								confMap.at<float>(v_curr, u) = c;
						} while (++v_curr < v);
					}
					else
					{
						cv::Point2f dir1, dir2;
						dir1 = Normal2Dir(normalMap.at<cv::Vec3f>(v_first, u));
						dir2 = Normal2Dir(normalMap.at<cv::Vec3f>(v, u));
						const cv::Point2f dirDiff((dir2 - dir1) / float(count + 1));
						do
						{
							depthMap.at<float>(v_curr, u) = (d += diff);
							dir1 += dirDiff;
							normalMap.at<cv::Vec3f>(v_curr, u) = Dir2Normal(dir1);
							interpNum++;
							if (!confMap.empty())
								confMap.at<float>(v_curr, u) = c;
						} while (++v_curr < v);
					}
				}
			}

			// reset counter
			count = 0;
		}
	}

	// LOG(INFO) << "Interpolate " << interpNum << " points in frame " << frames[ref_id].id;
}

bool MVS::FilterDepthImage(int ref_id)
{
    // 把所有近邻图像的深度都投影到参考图像上
    vector<cv::Mat> depth_images(neighbors[ref_id].size());
    for(int i = 0; i < neighbors[ref_id].size(); i++)
        ProjectDepthConfToRef(ref_id, neighbors[ref_id][i], true, false, depth_images[i]);
    // 深度差异（百分比）
    const float depth_diff_loose = config.depth_diff_threshold * 1.2f;
    const float depth_diff_strict = config.depth_diff_threshold * 0.8f;
    vector<cv::Point2i> offset_pts = {cv::Point2i(-1, 0), cv::Point2i(1, 0), cv::Point2i(0, 1), cv::Point2i(0, -1)};
    Frame& frame = frames[ref_id];
    frame.depth_filter = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);
    cv::Mat new_conf_map = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);
    for(int row = 0; row < frame.GetImageRows(); row++)
    {
        for(int col = 0; col < frame.GetImageCols(); col++)
        {
            cv::Point2i curr_pt(col, row);
            const float& depth = frame.depth_map.at<float>(curr_pt);
            if(depth <= 0)
                continue;
            int num_similar = 0;
            for(const cv::Mat& depth_image : depth_images)
            {
                const float& depth_neighbor = depth_image.at<float>(curr_pt);
                if((depth_neighbor > 0) && (abs((depth - depth_neighbor) / depth) < depth_diff_strict))
                    num_similar++;
            }
            if(num_similar < 2)
                continue;
            num_similar = 0;
            for(const cv::Mat& depth_image : depth_images)
            {
                for(const cv::Point2i& offset : offset_pts)
                {
                    if(!frame.IsInside(curr_pt + offset))
                        continue;
                    const float& depth_neighbor = depth_image.at<float>(curr_pt + offset);
                    if((depth_neighbor > 0) && (abs((depth - depth_neighbor) / depth) < depth_diff_loose))
                        num_similar++;
                }
            }
            bool keep_constant = (!frame.depth_constant.empty() && frame.depth_constant.at<uchar>(curr_pt));
            if(num_similar < 5 && !keep_constant)
                continue;
            frame.depth_filter.at<float>(curr_pt) = depth;
            if(!frame.conf_map.empty())
                new_conf_map.at<float>(curr_pt) = frame.conf_map.at<float>(curr_pt);
        }
    }
    if(!frame.conf_map.empty())
    {
        cv::imwrite(config.mvs_result_path + "conf_" + num2str(ref_id) + "_filter.jpg", DepthImageRGB(new_conf_map));
        return ExportConfMap(config.mvs_conf_path + num2str(ref_id) + "_filter.bin", new_conf_map);
    }
    return true;
}

bool MVS::FilterDepthImageRefine(int ref_id)
{
    // 把所有近邻图像的深度和置信度都投影到参考图像上
    vector<cv::Mat> depth_images(neighbors[ref_id].size());
    vector<cv::Mat> conf_images(neighbors[ref_id].size());
    for(int i = 0; i < neighbors[ref_id].size(); i++)
        ProjectDepthConfToRef(ref_id, neighbors[ref_id][i], true, false, depth_images[i], true, conf_images[i]);
    // 深度差异（百分比）
    const float depth_diff_loose = config.depth_diff_threshold * 1.2f;
    const float depth_diff_strict = config.depth_diff_threshold * 0.8f;
    vector<cv::Point2i> offset_pts = {cv::Point2i(-1, 0), cv::Point2i(1, 0), cv::Point2i(0, 1), cv::Point2i(0, -1)};
    Frame& frame = frames[ref_id];
    if(frame.depth_map.empty())
    {
        LOG(ERROR) << "Depth map of frame " << ref_id << " is empty, unable to filter depth map";
        return false;
    }
    frame.depth_filter = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);
    cv::Mat new_conf_map = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);

    // average similar depths, and decrease confidence if depths do not agree
    // (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
    for(int row = 0; row < frame.GetImageRows(); row++)
    {
        for(int col = 0; col < frame.GetImageCols(); col++)
        {
            cv::Point2i curr_pt(col, row);
            const float& depth = frame.depth_map.at<float>(curr_pt);
            if(depth <= 0)
            {
                frame.conf_map.at<float>(curr_pt) = 0;
                continue;
            }
            // update best depth and confidence estimate with all estimates
            float positive_conf = frame.conf_map.at<float>(curr_pt);
            float negative_conf = 0;
            float avg_depth = depth * positive_conf;
            int num_positive_views = 0, num_negative_views = 0;
            for(int n = depth_images.size() - 1; n >= 0; n--)
            {
                if(depth_images[n].empty() || conf_images[0].empty())
                    continue;
                const float& depth_neighbor = depth_images[n].at<float>(curr_pt);
                const float& conf_neighbor = conf_images[n].at<float>(curr_pt);
                if(depth_neighbor <= 0)
                {
                    if(num_positive_views + num_negative_views + n < 2)
                        goto bad_depth;
                }
                if(abs((depth - depth_neighbor) / depth) < depth_diff_loose )
                {
                    avg_depth += depth_neighbor * conf_neighbor;
                    positive_conf += conf_neighbor;
                    num_positive_views += 1;
                }
                else 
                {
                    // 被遮挡的情况 occlusion
                    // 从reference view 发出的射线在到达目标三维点之前就遇到了另一个从邻域图像投影过来的三维点
                    // 也就是邻域深度小于当前深度
                    if(depth_neighbor < depth)
                        negative_conf += conf_neighbor;
                    // free-space violation
                    else 
                    {
                        cv::Point3f X0 = eq.ImageToCam(curr_pt) * depth;
                        cv::Point2f x1 = eq.CamToImage(neighbors[ref_id][n].R_nr * X0 + cv::Point3f(neighbors[ref_id][n].t_nr));
                        cv::Point2i x1_round(round(x1.x), round(x1.y));
                        if(frames[neighbors[ref_id][n].id].IsInside(x1_round))
                        {
                            const float& c = frames[neighbors[ref_id][n].id].conf_map.at<float>(x1_round);
                            negative_conf += (c > 0 ? c : conf_images[n].at<float>(curr_pt));
                        }
                        else 
                            negative_conf += conf_images[n].at<float>(curr_pt);
                    }
                    num_negative_views += 1;
                }
            }
            avg_depth /= positive_conf;
            if(num_positive_views >= 2 && positive_conf > negative_conf && IsInside(avg_depth, config.min_depth, config.max_depth))
            {
                frame.depth_filter.at<float>(curr_pt) = avg_depth;
                new_conf_map.at<float>(curr_pt) = positive_conf - negative_conf;
                continue;
            }
            bad_depth:
            // 如果要求深度固定，那么当前位置的深度就直接保留
            if(!frame.depth_constant.empty() && frame.depth_constant.at<uchar>(curr_pt))
            {
                frame.depth_filter.at<float>(curr_pt) = depth;
                new_conf_map.at<float>(curr_pt) = 1.f;
            }
        }
    }
    cv::imwrite(config.mvs_result_path + "conf_" + num2str(ref_id) + "_filter.jpg", DepthImageRGB(new_conf_map, 1.f, 0.f));
    return ExportConfMap(config.mvs_conf_path + num2str(ref_id) + "_filter.bin", new_conf_map);
}

bool MVS::FilterDepthImage2(int ref_id)
{
    Frame& ref_frame = frames[ref_id];
    ref_frame.depth_filter = cv::Mat::zeros(ref_frame.GetImageRows(), ref_frame.GetImageCols(), CV_32F);
    for(int row = 0; row < ref_frame.GetImageRows(); row++)
    {
        for(int col = 0; col < ref_frame.GetImageCols(); col++)
        {
            const float& ref_depth = ref_frame.depth_map.at<float>(row, col);
            if(ref_depth <= 0)
                continue;
            cv::Point3f X0 = eq.ImageToCam(cv::Point2i(col, row)) * ref_depth;
            int num_similar = 0;
            for(const NeighborInfo& info : neighbors[ref_id])
            {
                cv::Point3f X1 = info.R_nr * X0 + cv::Point3f(info.t_nr);
                float depth = cv::norm(X1);
                cv::Point2f x1 = eq.CamToImage(X1);
                const float& depth_neighbor = frames[info.id].depth_map.at<float>(int(round(x1.y)), int(round(x1.x)));
                if((depth_neighbor > 0) && (abs((depth - depth_neighbor) / depth) < config.depth_diff_threshold))
                        num_similar++;
            }
            if(num_similar >= 2)
                ref_frame.depth_filter.at<float>(row, col) = ref_depth;
        }
    }
    return true;
}

// interpolate given pixel's estimate to the current position
float MVS::InterpolatePixel(const cv::Point2i& pt, const cv::Point2i& nx, const float& depth, const cv::Vec3f& normal)
{
    cv::Point3f view_ray = eq.ImageToCam(pt);
    cv::Point3f X1 = eq.ImageToCam(nx) * depth;
    float dnorm = view_ray.dot(normal);
    if(abs(dnorm) < 1e-6)
        return depth;
    float depth_new =  (X1.dot(normal) / dnorm);  
    if(IsInside(depth_new, config.min_depth, config.max_depth))
        return depth_new;
    else 
        return depth;
}

float MVS::InterpolatePixel2(const cv::Point2i& pt, const cv::Point2i& nx, const float& depth, const cv::Vec3f& normal)
{
    cv::Point3f view_ray = eq.ImageToCam(pt);
    cv::Point3f X1 = eq.ImageToCam(nx) * depth;
    cv::Vec4f plane(normal(0), normal(1), normal(2), -normal.dot(X1));
    const cv::Vec6f line(0,0,0, view_ray.x, view_ray.y, view_ray.z);
    cv::Point3f X0 = PlaneLineIntersect<float, cv::Point3f>(plane.val, line.val);
    if(isinf(X0.x))
        return depth;
    float depth_new = cv::norm(X0);
    if(IsInside(depth_new, config.min_depth, config.max_depth))
        return depth_new;
    else 
        return depth;
}

void MVS::CorrectNormal(const cv::Point2i& pt, cv::Vec3f& normal) 
{
    const cv::Point3f viewDir = eq.ImageToCam(pt);
    const float cosAngLen(normal.dot(viewDir));
    if (cosAngLen >= 0)
    {
        // normal = RMatrixBaseF(normal.cross(viewDir), MINF((ACOS(cosAngLen/norm(viewDir))-FD2R(90.f))*1.01f, -0.001f)) * normal;
        cv::Vec3f axis = normal.cross(viewDir);
        float rad = min((acos(cosAngLen) - float(M_PI_2)) * 1.01f, -0.001f);
        Eigen::Matrix3f R(Eigen::AngleAxisf(rad, Eigen::Vector3f(axis(0), axis(1), axis(2))));
        float x = R(0,0) * normal(0) + R(0,1) * normal(1) + R(0,2) * normal(2);
        float y = R(1,0) * normal(0) + R(1,1) * normal(1) + R(1,2) * normal(2);
        float z = R(2,0) * normal(0) + R(2,1) * normal(1) + R(2,2) * normal(2);
        normal(0) = x;
        normal(1) = y;
        normal(2) = z;
    }
}

cv::Mat MVS::ProjectDepthToNeighbor(int ref_id, const NeighborInfo& info)
{
    const cv::Mat& ref_depth = frames[ref_id].depth_map;
    const Frame& target_frame = frames[info.id];
    if(ref_depth.empty())
    {
        LOG(ERROR) << "depth image of frame " << ref_id << " is empty ";
        return cv::Mat();
    }
    cv::Mat project_depth = cv::Mat::zeros(ref_depth.size(), ref_depth.type());
    vector<cv::Point2i> pts(4);
    for(int row = 0; row < ref_depth.rows; row++)
    {
        for(int col = 0; col < ref_depth.cols; col++)
        {
            cv::Point3f pt_cam_ref = eq.ImageToCam(cv::Point2i(col, row)) * ref_depth.at<float>(row, col);
            cv::Point3f pt_cam_target = info.R_nr * pt_cam_ref + cv::Point3f(info.t_nr);
            float range = cv::norm(pt_cam_target);
            cv::Point2f pt_image = eq.CamToImage(pt_cam_target);
            pts[0] = cv::Point2i(ceil(pt_image.x), ceil(pt_image.y));
            pts[1] = cv::Point2i(floor(pt_image.x), ceil(pt_image.y));
            pts[2] = cv::Point2i(ceil(pt_image.x), floor(pt_image.y));
            pts[3] = cv::Point2i(floor(pt_image.x), floor(pt_image.y));
            // 把投影点的四个近邻点都设置为相同的深度
            // 如果已经有深度了，那么就保留较大的深度
            for(const cv::Point2i& pt : pts)
            {
                if(!target_frame.IsInside(pt))
                    continue;
                float& depth = project_depth.at<float>(pt);
                if(depth != 0 && depth < range)
                    continue;
                depth = range;
            }
        }
    }
    return project_depth;
}

void MVS::ProjectDepthConfToRef(int ref_id, const NeighborInfo& info,
                        bool project_depth, bool use_filtered_depth, cv::Mat& depth_projected, 
                        bool project_conf, cv::Mat& conf_project)
{
    if(!project_depth && !project_conf)
        return;
    const cv::Mat& nei_depth = use_filtered_depth ? frames[info.id].depth_filter : frames[info.id].depth_map;
    const cv::Mat& nei_conf = frames[info.id].conf_map;
    const Frame& ref_frame = frames[ref_id];
    // 投影需要深度图，所以深度图绝对不能为空
    if(nei_depth.empty())
    {
        LOG(ERROR) << "depth image of frame " << info.id << " is empty, unable to project to reference frame";
        return;
    }
    if(nei_conf.empty() && project_conf)
    {
        LOG(ERROR) << "confidence image of frame " << info.id << " is empty, unable to project to reference frame";
        return;
    }
    cv::Matx33f R_rn = info.R_nr.t();
    cv::Point3f t_rn = cv::Point3f(-R_rn * info.t_nr);
    if(project_depth)
        depth_projected = cv::Mat::zeros(nei_depth.size(), nei_depth.type());
    if(project_conf)
        conf_project = cv::Mat::zeros(nei_conf.size(), nei_conf.type());
    vector<cv::Point2i> pts(4);
    for(int row = 0; row < nei_depth.rows; row++)
    {
        for(int col = 0; col < nei_depth.cols; col++)
        {
            cv::Point3f pt_cam_nei = eq.ImageToCam(cv::Point2i(col, row)) * nei_depth.at<float>(row, col);
            cv::Point3f pt_cam_ref = R_rn * pt_cam_nei + t_rn;
            float range = cv::norm(pt_cam_ref);
            cv::Point2f pt_image = eq.CamToImage(pt_cam_ref);
            pts[0] = cv::Point2i(ceil(pt_image.x), ceil(pt_image.y));
            pts[1] = cv::Point2i(floor(pt_image.x), ceil(pt_image.y));
            pts[2] = cv::Point2i(ceil(pt_image.x), floor(pt_image.y));
            pts[3] = cv::Point2i(floor(pt_image.x), floor(pt_image.y));
            // 把投影点的四个近邻点都设置为相同的深度
            // 如果已经有深度了，那么就保留较大的深度
            for(const cv::Point2i& pt : pts)
            {
                if(!ref_frame.IsInside(pt))
                    continue;
                if(project_depth)
                {
                    float& depth = depth_projected.at<float>(pt);
                    if(depth != 0 && depth < range)
                        continue;
                    depth = range;
                }
                if(project_conf)
                {
                    conf_project.at<float>(pt) = nei_conf.at<float>(row, col);
                }
            }
        }
    }
}


pcl::PointCloud<pcl::PointXYZRGB> MVS::DepthImageToCloud(int ref_id, bool use_filtered_depth)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    const cv::Mat& img_depth = (use_filtered_depth ? frames[ref_id].depth_filter : frames[ref_id].depth_map);
    cv::Mat img_color = frames[ref_id].GetImageColor();
    for(int row = 0; row < frames[ref_id].GetImageRows(); row++)
    {
        for(int col = 0; col < frames[ref_id].GetImageCols(); col++)
        {
            const float depth = img_depth.at<float>(row, col);
            if(depth <= 0 || depth >= config.max_depth*0.8)
                continue;
            cv::Point3f point_cam = eq.ImageToCam(cv::Point2i(col, row)) * depth;
            cv::Point3f point_world = TranslatePoint<float, double>(point_cam, frames[ref_id].GetPose());
            pcl::PointXYZRGB p;
            p.x = point_world.x;
            p.y = point_world.y;
            p.z = point_world.z;
            const cv::Vec3b bgr = img_color.at<cv::Vec3b>(row, col);
            // 颜色检查，天蓝色直接滤掉
            cv::Vec3f hsv = BGR2HSV(bgr);
            hsv[0] *= 180.f; 
            hsv[1] *= 255.f; 
            hsv[2] *= 255.f;
            if(IsInside(hsv[0], 100.f, 124.f) && IsInside(hsv[1], 43.f, 200.f) && IsInside(hsv[2], 150.f, 255.f))
                continue;
            p.r = bgr[2];
            p.g = bgr[1];
            p.b = bgr[0];
            cloud.push_back(p);
        }
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal> MVS::DepthNormalToCloud(int ref_id, bool use_filtered_depth)
{
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
    const cv::Mat& img_depth = (use_filtered_depth ? frames[ref_id].depth_filter : frames[ref_id].depth_map);
    const cv::Mat& normal_map = frames[ref_id].normal_map;
    cv::Mat img_color = frames[ref_id].GetImageColor();
    Eigen::Matrix3d R_wc = frames[ref_id].GetPose().block<3,3>(0,0);
    for(int row = 0; row < frames[ref_id].GetImageRows(); row++)
    {
        for(int col = 0; col < frames[ref_id].GetImageCols(); col++)
        {
            const float depth = img_depth.at<float>(row, col);
            if(depth <= 0 || depth >= config.max_depth * 0.8)
                continue;
            cv::Point3f point_cam = eq.ImageToCam(cv::Point2i(col, row)) * depth;
            cv::Point3f point_world = TranslatePoint<float, double>(point_cam, frames[ref_id].GetPose());
            pcl::PointXYZRGBNormal p;
            p.x = point_world.x;
            p.y = point_world.y;
            p.z = point_world.z;
            const cv::Vec3b bgr = img_color.at<cv::Vec3b>(row, col);
            p.r = bgr[2];
            p.g = bgr[1];
            p.b = bgr[0];
            const cv::Vec3f& n = normal_map.at<cv::Vec3f>(row, col);
            Eigen::Vector3d normal_camera(n(0), n(1), n(2));
            Eigen::Vector3d normal_world(R_wc * normal_camera);
            p.normal_x = normal_world.x();
            p.normal_y = normal_world.y();
            p.normal_z = normal_world.z();
            cloud.push_back(p);
        }
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> MVS::MergeDepthImages(int skip, bool use_filtered_depth)
{
    assert(skip >= 1);
    LOG(INFO) << "Merge depth maps begin, skip = " << skip;
    vector<pcl::PointCloud<pcl::PointXYZRGB>> cloud_each_thread(config.num_threads);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < frames.size(); i += skip)
    {
        if(use_filtered_depth && frames[i].depth_filter.empty())
            continue;
        else if(!use_filtered_depth && frames[i].depth_map.empty())
            continue;
        cloud_each_thread[omp_get_thread_num()] += DepthImageToCloud(i, use_filtered_depth);
    }
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    for(auto& c : cloud_each_thread)
    {
        cloud += c;
        pcl::PointCloud<pcl::PointXYZRGB>().swap(c);
    }
    LOG(INFO) << "Merge depth maps end";
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> MVS::FuseDepthImages(bool use_filtered_depth)
{
    LOG(INFO) << "Fuse depth maps begin";
    ResetFrameCount();

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    vector<cv::Mat> occupied;
    // 这样做可以处理图像尺寸不同的情况
    for(const Frame& f : frames)
    {
        occupied.push_back(cv::Mat(f.GetImageRows(), f.GetImageCols(), CV_16U, cv::Scalar_<uint16_t>(65535)));
    }
    vector<pair<int,int>> idx_connections;
    for(int i = 0; i < frames.size(); i++)
    {
        idx_connections.push_back({i, neighbors[i].size()});
        frame_conf_count[i] = neighbors[i].size() + 1;
        frame_depth_filter_count[i] = neighbors[i].size() + 1;
        frame_gray_count[i] = neighbors[i].size() + 1;
    }
    // 对每张图像根据邻居的数量进行排序，先处理邻居多的图像
    sort(idx_connections.begin(), idx_connections.end(), [](const pair<int,int>& a, const pair<int,int>& b){return a.second > b.second;});
    
    // 记录当前三维点投影到各个图像上时的投影点 first-图像id  second-投影点坐标
    vector<pair<size_t, cv::Point2i>> view_project; 
    // 记录近邻中不可靠的深度所在的位置 
    vector<pair<size_t, cv::Point2i>> invalid_depth; 
    for(const pair<int,int>& p : idx_connections)
    {
        int ref_idx = p.first;
        Frame& ref_frame = frames[ref_idx];
        const cv::Mat& ref_depth_map = (use_filtered_depth ? ref_frame.depth_filter : ref_frame.depth_map);
        vector<int> ids = {ref_idx};
        for(const auto& neighbor : neighbors[ref_idx])
            ids.push_back(neighbor.id);
       
        if(ref_depth_map.empty())
            goto next_image;

        // 读取各种图像
        #pragma omp parallel for
        for(const int& id : ids)
        {
            // 读取深度图
            if(use_filtered_depth && frames[id].depth_filter.empty())
                ReadFrameDepth(config.mvs_depth_path + num2str(id) + (config.mvs_use_geometric ? "_geo" : "_pho") + ".bin", frames[id], true);
            else if(!use_filtered_depth && frames[id].depth_map.empty())
                ReadFrameDepth(config.mvs_depth_path + num2str(id) + (config.mvs_use_geometric ? "_geo" : "_pho") + ".bin", frames[id], false);
            // 读取置信度图
            if(frames[id].conf_map.empty())
            {
                if(use_filtered_depth)
                    ReadFrameConf(config.mvs_conf_path + num2str(id) + "_filter.bin", frames[id]);
                else
                {
                    ReadFrameConf(config.mvs_conf_path + num2str(id) + (config.mvs_use_geometric ? "_geo" : "_pho") + ".bin", frames[id]);
                    ConvertNCC2Conf(frames[id].conf_map);
                }
            }
            // 读取彩色图
            if(frames[id].GetImageColorRef().empty())
                frames[id].LoadImageColor(frames[id].name);
        }
        
        for(int row = 0; row < ref_frame.GetImageRows(); row++)
        {
            for(int col = 0; col < ref_frame.GetImageCols(); col++)
            {
                float depth = ref_depth_map.at<float>(row, col);
                if(depth <= 0 || depth >= config.max_depth * 0.8)
                    continue;
                uint16_t& occupied_id = occupied[ref_idx].at<uint16_t>(row, col);
                if(occupied_id != UINT16_MAX)
                    continue;
                occupied_id = ref_frame.id;
                cv::Vec3f color = ref_frame.GetImageColorRef().at<cv::Vec3b>(row, col);
                float confidence = ConfToWeight(ref_frame.conf_map.at<float>(row, col), depth);
                cv::Point3f X0 = eq.ImageToCam(cv::Point2i(col, row)) * depth;
                // X 和 color 是最终的三维点的空间位置以及颜色
                cv::Point3f X = TranslatePoint<float, double>(X0, ref_frame.GetPose()) * confidence;
                color *= confidence;
                 
                // 遍历所有的邻居，把邻居上深度一致的点都融合为同一个三维空间点
                for(const NeighborInfo& neighbor : neighbors[ref_idx])
                {
                    const size_t& n_idx = neighbor.id;
                    Frame& n_frame = frames[n_idx];
                    const cv::Mat& n_depth_map = (use_filtered_depth ? n_frame.depth_filter : n_frame.depth_map);
                    if(n_depth_map.empty())
                        continue;
                    cv::Point3f X1 = neighbor.R_nr * X0 + cv::Point3f(neighbor.t_nr);
                    cv::Point2f x1 = eq.CamToImage(X1);
                    cv::Point2i pt_proj(round(x1.x), round(x1.y));
                    if(!n_frame.IsInside(pt_proj))
                        continue;
                    const float& n_depth = n_depth_map.at<float>(pt_proj);
                    if(n_depth <= 0)
                        continue;
                    uint16_t& n_occupied_id = occupied[n_idx].at<uint16_t>(pt_proj);
                    if(n_occupied_id != UINT16_MAX)
                        continue;
                    if(abs((depth - n_depth) / depth) < config.depth_diff_threshold)
                    {
                        view_project.push_back({n_idx, pt_proj});
                        float n_confidence = ConfToWeight(n_frame.conf_map.at<float>(pt_proj), n_depth);
                        cv::Point3f n_point = eq.ImageToCam(pt_proj) * n_depth;
                        X += TranslatePoint<float, double>(n_point, n_frame.GetPose()) * n_confidence;
                        color += cv::Vec3f(n_frame.GetImageColorRef().at<cv::Vec3b>(pt_proj)) * n_confidence;
                        confidence += n_confidence;
                        n_occupied_id = occupied_id;
                    }
                    // 投影深度小于深度图上投影点对应的深度，说明这个投影点的深度不太准确，需要被过滤掉
                    if(cv::norm(X1) < n_depth)
                    {
                        invalid_depth.push_back({n_idx, pt_proj});
                    }
                }
                // 深度一致的数量太少，代表当前深度不可靠
                if(view_project.size() < 2)
                {
                    for(const pair<size_t, cv::Point2i>& p : view_project)
                        occupied[p.first].at<uint16_t>(p.second) = UINT16_MAX;
                    occupied_id = UINT16_MAX;
                }
                else 
                {
                    float nrm = 1.f / confidence;
                    cv::Point3f point_world = X * nrm;
                    color = color * nrm;
                    pcl::PointXYZRGB p;
                    p.x = point_world.x;
                    p.y = point_world.y;
                    p.z = point_world.z;
                    p.r = static_cast<uchar>(color(2));
                    p.g = static_cast<uchar>(color(1));
                    p.b = static_cast<uchar>(color(0));
                    for(const pair<size_t, cv::Point2i>& p : invalid_depth)
                    {
                        cv::Mat& depth_map = (use_filtered_depth ? frames[p.first].depth_filter : frames[p.first].depth_map);
                        if(!depth_map.empty())
                            depth_map.at<float>(p.second) = 0;
                    }
                    // 颜色检查，天蓝色直接滤掉
                    cv::Vec3f hsv = BGR2HSV(cv::Vec3b(color));
                    hsv[0] *= 180.f; 
                    hsv[1] *= 255.f; 
                    hsv[2] *= 255.f;
                    if(hsv[0] >= 100 && hsv[0] <= 124 && hsv[1] >= 43 && hsv[1] <= 200 && hsv[2] >= 150 && hsv[2] <= 255)
                        continue;
                    cloud.push_back(p);
                }
                invalid_depth.clear();
                view_project.clear();
            }
        }
        next_image:
        for(const int& id : ids)
        {
            if(--frame_conf_count[id] <= 0)
                frames[id].conf_map.release();
            if(--frame_depth_filter_count[id] <= 0)
                frames[id].depth_filter.release();
            if(--frame_gray_count[id] <= 0)
                frames[id].ReleaseImageColor();
        }
    }
    LOG(INFO) << "Fuse depth maps end";
    return cloud;
}

float MVS::ConfToWeight(const float& conf, const float& depth)
{
    return 1.f / (max(1.f - conf, 0.03f) * Square(depth));
}

void MVS::ConvertNCC2Conf(cv::Mat& conf)
{
    for(int row = 0; row < conf.rows; row++)
    {
        for(int col = 0; col < conf.cols; col++)
        {
            float c = conf.at<float>(row, col);
            conf.at<float>(row, col) = (c > 0 ? c : 0);
        }
    }
}

const std::vector<std::vector<NeighborInfo>>& MVS::GetNeighbors() const 
{
    return neighbors;
}


eigen_vector<Eigen::Matrix3d> MVS::GetGlobalRotation(bool with_invalid)
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

eigen_vector<Eigen::Vector3d> MVS::GetGlobalTranslation(bool with_invalid)
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

std::vector<std::string> MVS::GetFrameNames(bool with_invalid)
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

MVS::~MVS()
{
}