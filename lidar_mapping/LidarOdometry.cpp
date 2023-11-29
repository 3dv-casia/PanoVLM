/*
 * @Author: Diantao Tu
 * @Date: 2021-12-27 17:03:40
 */

#include "LidarOdometry.h"

using namespace std;

LidarOdometry::LidarOdometry(const std::vector<Velodyne>& _lidars, const Config& _config):
    lidars(_lidars),config(_config)
{
}

bool LidarOdometry::RefinePose(double& cost, int& steps, bool use_segment)
{
    for(Velodyne& lidar : lidars)
    {
        if(lidar.IsPoseValid() && !lidar.IsInWorldCoordinate())
            lidar.Transform2LidarWorld();
    }
    // 记录下每个雷达帧位姿，用于优化
    eigen_vector<Eigen::Vector3d> angleAxis_lw_list(lidars.size(), Eigen::Vector3d::Ones());
    eigen_vector<Eigen::Vector3d> t_lw_list(lidars.size(), Eigen::Vector3d::Ones());
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        Eigen::Matrix4d T_lw = lidars[i].GetPose().inverse();
        Eigen::Matrix3d R_lw = T_lw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_lw.data(), angleAxis_lw_list[i].data());
        t_lw_list[i] = T_lw.block<3,1>(0,3);
    }
    
    vector<vector<int>> neighbors_all = FindNeighbors(lidars, 6);
    ceres::Problem problem;

    if(config.point_to_line_residual)
        AddLidarPointToLineResidual(neighbors_all, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                    config.point_to_line_dis_threshold, use_segment, config.angle_residual, config.normalize_distance);
    if(config.line_to_line_residual && use_segment)
    {
        #if 0
        AddLidarLineToLineResidual(neighbors_all, lidars, angleAxis_lw_list, t_lw_list, problem,
                                    config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance);
        #else
        LidarLineMatch matcher(lidars);
        matcher.SetNeighborSize(4);
        matcher.SetMinTrackLength(3);
        matcher.GenerateTracks();
        AddLidarLineToLineResidual2(neighbors_all, lidars, angleAxis_lw_list, t_lw_list, problem, matcher.GetTracks(),
                                    config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance);
        #endif
    }
    if(config.point_to_plane_residual)
        AddLidarPointToPlaneResidual(neighbors_all, lidars, angleAxis_lw_list, t_lw_list, problem,
                                    config.point_to_plane_dis_threshold, config.lidar_plane_tolerance, 
                                    config.angle_residual, config.normalize_distance);
    for(int i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        problem.SetParameterBlockConstant(angleAxis_lw_list[i].data());
        problem.SetParameterBlockConstant(t_lw_list[i].data());
        break;
    }

    if(problem.NumResidualBlocks() == 0)
    {
        LOG(ERROR) << "no residual";
        return false;
    }
    else 
    {
        LOG(INFO) << "Residual block number: " << problem.NumResidualBlocks();
    }

    ceres::Solver::Options options = SetOptionsLidar(config.num_threads, lidars.size());
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    LOG(INFO) << "successful: " << summary.num_successful_steps << ", unsuccessful: " << summary.num_unsuccessful_steps; 
    if(!summary.IsSolutionUsable())
    {
        LOG(ERROR) << "LiDAR odometry failed";
        LOG(INFO) << summary.FullReport();
        CameraCenterPCD(config.odo_result_path + "/failed_center.pcd", GetGlobalTranslation());
        ofstream f(config.odo_result_path + "/lidar_pose_failed.txt");
        for(int i = 0; i < angleAxis_lw_list.size(); i++)
        {
            f << i << " " << angleAxis_lw_list[i].x() << " " << angleAxis_lw_list[i].y() << " " << angleAxis_lw_list[i].z() << " " 
                << t_lw_list[i].x() << " " << t_lw_list[i].y() << " " << t_lw_list[i].z() << endl;
        }
        f.close();
    }
    else 
        LOG(INFO) << summary.BriefReport();

    // 雷达数据从世界坐标系变回雷达坐标系
    for(int i  = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].valid || !lidars[i].IsPoseValid())
            continue;
        lidars[i].Transform2Local();
        Eigen::Matrix3d R_lw ;
        ceres::AngleAxisToRotationMatrix(angleAxis_lw_list[i].data(), R_lw.data());
        Eigen::Matrix3d R_wl = R_lw.transpose();
        lidars[i].SetRotation(R_wl);
        lidars[i].SetTranslation(-R_wl * t_lw_list[i]);
    }
    cost = summary.final_cost;
    steps = summary.num_successful_steps;
    return summary.IsSolutionUsable();
}

bool LidarOdometry::EstimatePose(const int max_iteration)
{
    LOG(INFO) << "============== LiDAR pose refine begin ======================";
    LOG(INFO) << "Prepare to estimate " << lidars.size() << " lidar pose";
    LOG(INFO) << "Configuration: use angle - " << (config.angle_residual ? "true\r\n" : "false\r\n") << 
                "\t\t use point to line - " << (config.point_to_line_residual ? "true\r\n" : "false\r\n") << 
                "\t\t use line to line - " << (config.line_to_line_residual ? "true\r\n" : "false\r\n") <<
                "\t\t use point to plane - " << (config.point_to_plane_residual ? "true\r\n" : "false\r\n") <<
                "\t\t lidar plane tolerance - " << config.lidar_plane_tolerance << "\r\n" << 
                "\t\t use normalized distance - " << (config.normalize_distance ? "true\r\n" : "false\r\n");
    

    // 提取特征只需要做一次，后面的迭代过程中特征是不变的
    // 只有点云本身改变的时候，特征才会改变
    LOG(INFO) << "Extract lidar features";
    omp_set_num_threads(config.num_threads);
    #pragma omp parallel for
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].valid || !lidars[i].IsPoseValid() )
        {
            lidars[i].SetRotation(Eigen::Matrix3d::Zero());
            lidars[i].SetTranslation(Eigen::Vector3d::Ones() * numeric_limits<double>::infinity());
            continue;
        }
        // lidars[i].SaveFeatures(config.odo_result_path);
        // lidars[i].ReOrderVLP2();
        lidars[i].ReOrderVLP();
        lidars[i].ExtractFeatures(config.max_curvature, config.intersection_angle_threshold, 
                                config.extraction_method, config.lidar_segmentation);
        lidars[i].Transform2LidarWorld();
        // lidars[i].SaveFeatures(config.odo_result_path);
    }
    // return true;
    bool segmented = false;
    // 遍历所有雷达，只要有任何一个雷达是经过分割的，就认为所有雷达是经过分割的
    for(Velodyne& lidar : lidars)
    {
        segmented = !lidar.edge_segmented.empty();
        if(segmented)
            break;
    }
    if(!segmented && config.line_to_line_residual)
        LOG(INFO) << "lidar lines are not segmented, line to line residual is disabled";
    if(config.point_to_line_residual && config.line_to_line_residual)
    {
        LOG(WARNING) << "point to line residual and line to line residual are both enable, it may lead to wrong result";
    }
    double curr_cost = 0, last_cost = 0;
    int curr_setp = INT16_MAX, last_step = INT16_MAX;
    for(size_t iter = 0; iter < max_iteration; iter++)
    {
        LOG(INFO) << "iteration : " << iter;
        RefinePose(curr_cost, curr_setp, segmented);
        CameraCenterPCD(config.odo_result_path + "/lidar_center_refined_" + num2str(iter) + ".pcd", GetGlobalTranslation());
        if(abs(curr_cost - last_cost) / last_cost < 0.01)
        {
            LOG(INFO) << "early termination condition fulfilled. cost change is less than 1%";
            break;
        }
        if(curr_setp < 5 && last_step < 5)
        {
            LOG(INFO) << "early termination condition fulfilled. iteration step is less than 5 steps in last two iterations";
            break;
        }
        last_cost = curr_cost;
        last_step = curr_setp;
    }
    LOG(INFO) << "============== LiDAR pose refine end ======================";

    return true;
}

bool LidarOdometry::UndistortLidars(const float gap_time)
{
    LOG(INFO) << "LiDAR undistortion begin";
    if(boost::filesystem::exists(config.lidar_path_undistort))
        boost::filesystem::remove_all(config.lidar_path_undistort);
    boost::filesystem::create_directories(config.lidar_path_undistort);
    // 改变LiDAR的坐标系，原本是X向右，Y向前，Z向上
    // 变成 X向右，Y向下，Z向前，这种顺序适应相机的坐标系
    Eigen::Matrix4f T_cam_lidar;
    T_cam_lidar << 1, 0, 0, 0 ,
                    0, 0, -1, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1;

    double lidar_duration = 0.1;
    #pragma omp parallel for
    for(int i = 0; i < lidars.size(); i++)
    {
        // 当前雷达最后一个点对应的位姿，一般情况下这个值都是下一帧的位姿，但有时候下一帧雷达出现了问题，
        // 导致位姿不可靠，这个时候就需要一直向后搜索，直到找到一个可靠的位姿，然后通过插值得到当前雷达
        // 最后一个点对应的位姿
        Eigen::Matrix4d pose;
        // 如果当前雷达位姿不可用，那就直接跳过畸变矫正，把原始的点云当做矫正后的结果保存
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
        {
            goto save_undistort;                    
        }
        // 如果当前雷达还没到最后一个，那就向后寻找到第一个可用的位姿
        else if(i < lidars.size() - 1)
        {
            int idx = i + 1;
            while(idx < lidars.size() && !lidars[idx].IsPoseValid() && !lidars[idx].valid)
                idx++;
            if(idx >= lidars.size())
                goto save_undistort;
            pose = SlerpPose(lidars[i].GetPose(), lidars[idx].GetPose(), lidar_duration / ((idx - i) * (lidar_duration + gap_time)));
        }
        // 如果当前雷达是最后一帧雷达，那就向前寻找到最近的可用位姿
        else if (i == lidars.size() - 1)
        {
            int idx = i - 1;
            while(idx >= 0 && !lidars[idx].IsPoseValid() && !lidars[i].valid)
                idx--;
            if(idx <= 0)
                goto save_undistort;
            // 这个插值得到的结果相当于是当前帧的前一帧的位姿，而我们实际需要的是后一帧的位姿
            // 同时假设这三帧之间的相对运动是保持不变的
            pose = SlerpPose(lidars[idx].GetPose(), lidars[i].GetPose(), 1.0 - lidar_duration / ((idx - i) * (lidar_duration + gap_time)));
            // 从插值位姿到当前位姿的变换，T_cs = T_cw * T_ws  c = current  s = start
            Eigen::Matrix4d T_cs = lidars[i].GetPose().inverse() * pose;
            // 计算当前帧的后一帧位姿 T_world_next = T_world_current * T_current_next
            pose = lidars[i].GetPose() * T_cs;
        }
        lidars[i].UndistortCloud(pose);
        save_undistort:
        // 对雷达矫正完畸变后，要把它变回自己最初的坐标系，也就是X向右，Y向前，Z向上
        // 这样才能保存到本地，否则的话以后再用这个数据会出现坐标系的问题
        pcl::PointCloud<pcl::PointXYZI> cloud_color;
        cloud_color = ColorizeCloudByTime(lidars[i].cloud);
        pcl::transformPointCloud(cloud_color, cloud_color, T_cam_lidar.inverse());
        string::size_type pos = lidars[i].name.find_last_of('/');
        string file_name = lidars[i].name.substr(pos + 1);
        // 如果雷达点是空的，那就说明当前实际上是没有采集到雷达数据的，但是为了数据的完整性就生成了一个空点云
        // 所以在保存的时候，也要保存一个空点云。但pcl不支持保存空点云，那么就要写入一个点
        if(cloud_color.empty())
        {
            pcl::PointXYZI p(1);
            p.x = p.y = p.z = 0;
            cloud_color.push_back(p);
        }
        pcl::io::savePCDFileASCII(config.lidar_path_undistort + file_name, cloud_color);
    }
    LOG(INFO) << "LiDAR undistortion end";
    return true;
}

bool LidarOdometry::SetToOrigin(size_t lidar_idx)
{
    if(lidar_idx > lidars.size())
    {
        LOG(ERROR) << "Invalid frame idx, no frame in frame list";
        return false;
    }
    if(!lidars[lidar_idx].IsPoseValid() || !lidars[lidar_idx].valid)
    {
        LOG(WARNING) << "LiDAR " << lidar_idx << " pose is invalid, set another lidar";
        for(lidar_idx = 0; lidar_idx < lidars.size(); lidar_idx++)
        {
            if(lidars[lidar_idx].IsPoseValid() && lidars[lidar_idx].valid)
                break;
        }
        LOG(INFO) << "Set lidar " << lidar_idx << " as world coordinate";
    }
    // 这里的下标c 代表center，是指的新的世界坐标系
    const Eigen::Matrix4d T_wc = lidars[lidar_idx].GetPose();
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_iw = lidars[i].GetPose().inverse();
        Eigen::Matrix4d T_ic = T_iw * T_wc;     // 新的世界坐标系到相机坐标系的变换
        lidars[i].SetPose(T_ic.inverse());
    }
    return true;
}

void LidarOdometry::ResetAllLidars()
{
    for(Velodyne& lidar : lidars)
    {
        lidar.Reset();
        // 确保所有的LiDAR都有数据
        if(lidar.cloud.empty() && lidar.IsPoseValid())
            lidar.LoadLidar(lidar.name);
    }
}

bool LidarOdometry::LoadLidars(const std::string path)
{
    vector<string> names = IterateFiles(path, ".pcd");
    // if(names.size() != lidars.size())
    // {
    //     LOG(ERROR) << "Fail to load lidar data in lidar odometry, num of lidar file != num of lidar data";
    //     return false;
    // }
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < lidars.size(); i++)
    {
        lidars[i].SetName(names[i]);
        lidars[i].LoadLidar(names[i]);
    }
    return true;
}

pcl::PointCloud<PointType> LidarOdometry::FuseLidar(int skip, double min_range, double max_range)
{
    pcl::PointCloud<PointType> cloud_fused;
    double sq_min_range = min_range * min_range, sq_max_range = max_range * max_range;
    for(size_t i = 0; i < lidars.size(); i += (skip + 1))
    {
        if(!lidars[i].valid || !lidars[i].IsPoseValid())
            continue;
        if(lidars[i].cloud.empty())
            lidars[i].LoadLidar(lidars[i].name);
        pcl::PointCloud<pcl::PointXYZI> cloud_filtered;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_raw = lidars[i].cloud_scan.makeShared();
        if(cloud_raw->empty())
            cloud_raw = lidars[i].cloud.makeShared();
        for(const pcl::PointXYZI& pt : cloud_raw->points)
        {
            double range = pt.x * pt.x + pt.y * pt.z + pt.z * pt.z;
            if(range > sq_max_range || range < sq_min_range)
                continue;
            cloud_filtered.push_back(pt);
        }
        pcl::transformPointCloud(cloud_filtered, cloud_filtered, lidars[i].GetPose());
        cloud_fused += cloud_filtered;
    }
    return cloud_fused;
}

const std::vector<Velodyne>& LidarOdometry::GetLidarData() const
{
    return lidars;
}

eigen_vector<Eigen::Matrix3d> LidarOdometry::GetGlobalRotation(bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> global_rotation;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        // 雷达位姿不可用包括两种情况，1.位姿真的不行；2.雷达数据本身有问题
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = l.GetPose();
        global_rotation.push_back(T_wc.block<3,3>(0,0));
    }
    return global_rotation;
}

eigen_vector<Eigen::Vector3d> LidarOdometry::GetGlobalTranslation(bool with_invalid)
{
    eigen_vector<Eigen::Vector3d> global_translation;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = l.GetPose();
        global_translation.push_back(T_wc.block<3,1>(0,3));
    }
    return global_translation;
}

std::vector<string> LidarOdometry::GetLidarNames(bool with_invalid)
{
    vector<string> names;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        names.push_back(l.name);
    }
    return names;
}

LidarOdometry::~LidarOdometry()
{
}