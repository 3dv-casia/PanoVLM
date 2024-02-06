/*
 * @Author: Diantao Tu
 * @Date: 2021-10-21 15:13:07
 */

#include "./base/common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <chrono>
#include <glog/logging.h>

#include "./base/Config.h"
#include "./base/Serialization.h"
#include "./sensors/Velodyne.h"
#include "./sensors/Frame.h"
#include "./joint_optimization/CameraLidarOptimizer.h"
#include "./joint_optimization/PanoramaLineMatch.h"
#include "./util/Util.h"
#include "./util/PanoramaLine.h"

#include "./util/Visualization.h"
#include "./util/FileIO.h"
#include "./sfm/SfM.h"

#include "./lidar_mapping/LidarLineMatch.h"
#include "./lidar_mapping/LidarOdometry.h"
#include "./mvs/MVS.h"
#include "./mvs/Texture.h"

using namespace std;

void TimeReport( map<string, double> time_spent);
void SetLog(string log_path);
int InitCameraPose(const Config& config, map<string, double>& time_spent);
int InitLidarPose(const Config& config, map<string, double>& time_spent);
int JointOptimization(const Config& config, map<string, double>& time_spent);
int ColorizeLidarMap(const Config& config, map<string, double>& time_spent);
int JointMVS(const Config& config, map<string, double>& time_spent);
int DependentMVS(const Config& config, map<string, double>& time_spent);

int main(int argc, char** argv)
{  
    map<string, double> time_spent;     // 各个阶段用时

    omp_set_num_threads(omp_get_max_threads() - 1);
    string log_path = "./logs/";
    SetLog(log_path);

    
    assert(argc == 3);
    string stage = argv[1];
    string config_path = argv[2];   // 配置文件路径
    
    // 读取config文件
    Config config(config_path);
    LOG(INFO) << "config file path: " << config_path;
    LOG(INFO) << "image data path: " << config.image_path;
    LOG(INFO) << "lidar data path: " << config.lidar_path;   
    LOG(INFO) << "threads : " << config.num_threads;

    if(stage == "init_camera_pose")
    {
        InitCameraPose(config, time_spent);
    }
    else if(stage == "init_lidar_pose")
    {
        InitLidarPose(config, time_spent);
    }
    else if(stage == "joint_optimization")
    {
        JointOptimization(config, time_spent);
    }
    else if(stage == "colorize_lidar_map")
    {
        ColorizeLidarMap(config, time_spent);
    }
    else if(stage == "joint_mvs")
    {
        JointMVS(config, time_spent);
    }
    else if(stage == "dependent_mvs")
    {
        DependentMVS(config, time_spent);
    }
    else
    {
        LOG(ERROR) << "stage error, input stage is " << stage;
    }
    
    TimeReport(time_spent);

    return 0;
}

int InitCameraPose(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    if(!boost::filesystem::exists(config.sfm_result_path))
        boost::filesystem::create_directories(config.sfm_result_path);

    // 读取mask
    cv::Mat mask;
    if(!config.mask_path.empty())
    {
        mask = cv::imread(config.mask_path, cv::IMREAD_GRAYSCALE);
        cv::threshold(mask, mask, 0.1, 1, cv::THRESH_BINARY);
    }

    omp_set_num_threads(config.num_threads);

    vector<string> image_names_sync = IterateFiles(config.image_path, ".jpg");

    /* 以下为SfM流程 */
    SfM sfmer(config);

    /* 1. 如果有雷达数据，那么就把雷达信息加入 sfm中 */
    if(!config.lidar_path.empty())
    {
        vector<Velodyne> lidars;
        vector<string> lidar_names_sync = IterateFiles(config.lidar_path, ".pcd");
        for(int i = 0; i < lidar_names_sync.size(); i++)
        {
            Velodyne lidar(16, i);
            lidar.SetName(lidar_names_sync[i]);
            lidars.push_back(lidar);
        }
        if(!boost::filesystem::exists(config.lidar_path_undistort))
            boost::filesystem::create_directories(config.lidar_path_undistort);
        sfmer.SetLidars(lidars);
    }

    /* 
    * 2. 从本地读取Frame信息，包括了图像的路径、图像的尺寸、图像的位姿、图像的特征点等等信息。
    * 如果本地没有相关的Frame信息，就进行特征提取，然后保存到本地。
    * 计算Frame需要以下几个条件之一：本地没有frame，本地frame数量不对，读取frame出问题
    */
    bool update_depth = false, update_frame = true;
    // goto translation_averaging;
    if(!config.frame_path.empty())
    {
        vector<string> frame_names_sync = IterateFiles(config.frame_path, ".bin");
        if(frame_names_sync.size() == image_names_sync.size())
        {
            LOG(INFO) << "Use existing frame data in " << config.frame_path ;
            // 如果读取frame成功了，那么就会返回true，那么就不需要更新本地保存的frame，所以update=false
            update_frame = !sfmer.LoadFrameBinary(config.image_path, config.frame_path);
        }
        else 
        {
            LOG(INFO) << "number of frame != number of images, extract features";
            update_frame = true;
        }
    }
    // update_frame=ture就代表本地没有所有的frame，或者读取frame出现问题了，自然需要从头计算
    if(update_frame)
    {
        t1 = chrono::high_resolution_clock::now();
        sfmer.ReadImages(image_names_sync, mask);
        // 保存frame到本地
        if(!config.frame_path.empty())
            sfmer.ExportFrameBinary(config.frame_path);
        t2 = chrono::high_resolution_clock::now();
        time_spent["SfM.extract image features"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    }        
    
    /**
     * 3. 从本地读取深度图信息，这里仅仅是读取信息，并没有真的把深度图读取到内存里
     *  如果读取信息出问题，或者深度图数量和frame数量不匹配，那么就重新计算一遍深度图
    */
    if(!config.depth_path.empty())
    {
        vector<string> depth_names_sync = IterateFiles(config.depth_path, ".bin");
        if(depth_names_sync.size() != image_names_sync.size())
        {
            LOG(INFO) << "number of depth_image != number of images " 
                    << depth_names_sync.size() << " - " << image_names_sync.size() 
                    << " , compute depth image"; 
            update_depth = true;
        }
    }
    if(update_depth)
    {
        t1 = chrono::high_resolution_clock::now();
        sfmer.ComputeDepthImage(config.T_cl);
        t2 = chrono::high_resolution_clock::now();
        time_spent["SfM.compute depth maps"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    }


    /** 
    * 4. 图像特征匹配，主要是进行SIFT特征匹配
    */
    t1 = chrono::high_resolution_clock::now();
    // 先读取本地保存的match pair，如果读取失败了，就重新计算一遍
    if(!sfmer.LoadMatchPairBinary(config.match_pair_path))
    { 
        // 尝试读取txt里的经过sift匹配的图像对，这样在后面特征匹配的时候就相当于是可以直接跳过sift匹配的部分，
        // 虽然没有这样下来没有sift匹配的结果，但是能直接跳过错误的图像对之间的匹配，仅仅对正确的图像对之间进行sift匹配
        // 也能一定程度节省时间。
        // 如果连txt读取也失败了，就说明根本没有进行过图像特征匹配，那么就需要先初始化匹配对
        if(!sfmer.LoadMatchPairTXT(config.sfm_result_path + "after_sift_match.txt"))
            // sfmer.InitImagePairs(FrameMatchMethod::GPS_VLAD | FrameMatchMethod::CONTIGUOUS);
            sfmer.InitImagePairs(FrameMatchMethod::VLAD | FrameMatchMethod::CONTIGUOUS);
            // sfmer.InitImagePairs(FrameMatchMethod::CONTIGUOUS);
        
        sfmer.MatchImagePairs(config.sift_match_num_threshold);   
        // sfmer.ExportMatchPairBinary(config.sfm_result_path + "pair_after_sift");
        sfmer.ExportMatchPairBinary(config.match_pair_path);
    }
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.match image pairs"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    sfmer.ExportMatchPairTXT(config.sfm_result_path + "/match_pair.txt");
    
    relative_pose:
    // sfmer.LoadMatchPairBinary(config.sfm_result_path + "pair_after_sift");
    sfmer.LoadMatchPairBinary(config.match_pair_path);
    /**
     * 5. 计算相对位姿，根据匹配的图像对之间的特征点使用8点法计算相对的旋转和平移
     * 如果有深度图，那么会根据深度图设置相对平移的尺度
    */
    t1 = chrono::high_resolution_clock::now();
    sfmer.FilterImagePairs(25, 3, config.keep_pairs_no_scale);
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.estimate relative pose"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    sfmer.ExportMatchPairTXT(config.sfm_result_path + "/relative_pose.txt");
    sfmer.ExportMatchPairBinary(config.sfm_result_path + "pairs_after_relative_pose");
    // sfmer.ExportMatchPairBinary(config.match_pair_path);
    
    global_pose:
    sfmer.LoadFrameBinary(config.image_path , config.sfm_result_path + "frames_after_RA", true);
    sfmer.LoadMatchPairBinary(config.sfm_result_path + "pairs_after_relative_pose");
    // sfmer.LoadMatchPairBinary(config.match_pair_path);
    
    /**
     * 6. 旋转平均
    */
    t1 = chrono::high_resolution_clock::now();
    sfmer.EstimateGlobalRotation(config.rotation_averaging_method);
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.estimate global rotation"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    sfmer.ExportMatchPairBinary(config.sfm_result_path + "/pairs_after_RA");
    sfmer.ExportFrameBinary(config.sfm_result_path + "/frames_after_RA");

    translation_averaging:
    /**
     * 7. 平移平均
    */
    sfmer.LoadFrameBinary(config.image_path, config.sfm_result_path + "/frames_after_RA", true);        
    sfmer.LoadMatchPairBinary(config.sfm_result_path + "/pairs_after_RA");

    t1 = chrono::high_resolution_clock::now();
    sfmer.EstimateGlobalTranslation(config.translation_averaging_method);
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.estimate global translation"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    // 输出相机位姿并保存到本地
    eigen_vector<Eigen::Matrix3d> global_rotation = sfmer.GetGlobalRotation(true); 
    eigen_vector<Eigen::Vector3d> global_translation = sfmer.GetGlobalTranslation(true);
    vector<string> frame_names = sfmer.GetFrameNames(true);
    ExportPoseT(config.sfm_result_path + "/camera_pose_beforeBA.txt", global_rotation, global_translation, frame_names);
    CameraPoseVisualize(config.sfm_result_path + "/camera_pose_beforeBA.ply", global_rotation, global_translation);
    CameraCenterPCD(config.sfm_result_path + "/camera_center_beforeBA.pcd", global_translation);

    /**
     * 8. 计算初始的三维点云(structure)
    */
    t1 = chrono::high_resolution_clock::now();
    sfmer.EstimateStructure();
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.estimate initial structure"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    sfmer.ExportStructurePCD(config.sfm_result_path + "structure_init.pcd");


    /**
     * 9. 全局BA，一共进行了三次
     * 第一次，固定相机的旋转不变，只优化平移和三维点位置，同时过滤掉一些投影误差太大的三维点
     * 第二次，旋转、平移、三维点都改变，同时过滤掉一些投影误差太大的三维点
     * 第三次，旋转、平移、三维点都改变，对三维点的过滤更加严格
    */
    // 对点云以及平移进行优化，固定旋转
    #if 0
    t1 = chrono::high_resolution_clock::now();
    LOG(INFO) << "Start refine structure and translation";
    sfmer.GlobalBundleAdjustment(RESIDUAL_TYPE::PIXEL_RESIDUAL, -1, true, false, true);
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.BA-1"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
    sfmer.ExportStructurePCD(config.sfm_result_path + "structure_trans_refine.pcd");
    // 输出相机位姿并保存到本地
    global_rotation = sfmer.GetGlobalRotation(true); 
    global_translation = sfmer.GetGlobalTranslation(true);
    frame_names = sfmer.GetFrameNames(true);
    ExportPoseT(config.sfm_result_path + "/camera_pose_refine.txt", global_rotation, global_translation, frame_names);
    CameraPoseVisualize(config.sfm_result_path + "/camera_pose_refine.ply", global_rotation, global_translation);
    CameraCenterPCD(config.sfm_result_path + "/camera_center_refine.pcd", global_translation);
    #endif

    // 对点云以及平移进行优化，旋转、平移、空间点坐标都是可以变化的
    t1 = chrono::high_resolution_clock::now();
    LOG(INFO) << "Start refine structure and translation and rotation";
    sfmer.GlobalBundleAdjustment(RESIDUAL_TYPE::PIXEL_RESIDUAL, 40, true, true, true);
    sfmer.GlobalBundleAdjustment(RESIDUAL_TYPE::PIXEL_RESIDUAL, 10, true, true, true);
    t2 = chrono::high_resolution_clock::now();
    time_spent["SfM.BA-2"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    sfmer.ExportStructurePCD(config.sfm_result_path + "/structure_final.pcd");
    sfmer.ExportStructureBinary(config.sfm_result_path + "/points.bin");
    // 输出相机位姿并保存到本地
    global_rotation = sfmer.GetGlobalRotation(true); 
    global_translation = sfmer.GetGlobalTranslation(true);
    frame_names = sfmer.GetFrameNames(true);
    ExportPoseT(config.sfm_result_path + "/camera_pose_final.txt", global_rotation, global_translation, frame_names);
    CameraPoseVisualize(config.sfm_result_path + "/camera_pose_final.ply", global_rotation, global_translation);
    CameraCenterPCD(config.sfm_result_path + "/camera_center_final.pcd", global_translation);


    // 输出雷达的位姿, 可能会用于后续的camera-lidar联合优化
    // 这里要判断一下，如果没有LiDAR名称，就代表着仅仅是进行了纯粹的sfm流程，自然不需要输出LiDAR位姿
    if(!config.lidar_path.empty())
    {
        vector<string> lidar_names_sync = IterateFiles(config.lidar_path, ".pcd");
        // 输出匹配的图像对，用于之后的camera-lidar联合优化
        sfmer.ExportMatchPairBinary(config.match_pair_joint_path);
        eigen_vector<Eigen::Vector3d> lidar_translation;
        eigen_vector<Eigen::Matrix3d> lidar_rotation;
        vector<Velodyne> lidars;
        for(size_t i = 0; i < lidar_names_sync.size(); i++)
        {
            Velodyne lidar(16, i);
            lidar.SetName(lidar_names_sync[i]);
            lidars.push_back(lidar);
        }
        SetLidarPose(sfmer.GetFrames(), lidars, config.T_cl, config.time_offset);
        for(const Velodyne& lidar : lidars)
        {
            lidar_translation.push_back(lidar.GetPose().block<3,1>(0,3));
            lidar_rotation.push_back(lidar.GetPose().block<3,3>(0,0));
        }
        ExportPoseT(config.sfm_result_path + "/lidar_pose.txt", lidar_rotation, lidar_translation, lidar_names_sync);
    }

    // 根据sfm结果把雷达融合在一起
    if(false)
    {
        vector<string> lidar_names_sync = IterateFiles(config.lidar_path, ".pcd");
        eigen_vector<Eigen::Matrix3d> R_wl_list;
        eigen_vector<Eigen::Vector3d> t_wl_list;
        vector<string> name_list;
        ReadPoseT(config.sfm_result_path + "/lidar_pose.txt", false, R_wl_list, t_wl_list, name_list);
        pcl::PointCloud<pcl::PointXYZI> cloud_all;
        pcl::PointCloud<pcl::PointXYZ> lidar_center;
        for(int i = 0; i < name_list.size(); i+=5)
        {
            Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
            T_wl.block<3,3>(0,0) = R_wl_list[i];
            T_wl.block<3,1>(0,3) = t_wl_list[i];

            pcl::PointXYZ l(T_wl(0,3), T_wl(1,3), T_wl(2,3));
            lidar_center.push_back(l);
            
            pcl::PointCloud<pcl::PointXYZI> cloud;
            Velodyne lidar(16, i);
            lidar.LoadLidar(name_list[i]);
            pcl::transformPointCloud(lidar.cloud, cloud, T_wl);
            cloud_all += cloud;
        }
        pcl::io::savePCDFileASCII(config.sfm_result_path + "cloud_fuse_final.pcd", cloud_all);
        // pcl::io::savePCDFileASCII("camera_center_final.pcd", camera_center);
        pcl::io::savePCDFileASCII(config.sfm_result_path + "lidar_cener_final.pcd", lidar_center);
    }
    return 0;
}

int InitLidarPose(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    if(!boost::filesystem::exists(config.odo_result_path))
        boost::filesystem::create_directories(config.odo_result_path);
    // 读取雷达以及对应的位姿，这里读取了所有的雷达，无论位姿是否可用
    vector<Velodyne> lidar_data_list;
    LoadLidarPose(lidar_data_list, config.sfm_result_path + "/lidar_pose.txt", true);
    
    t1 = chrono::high_resolution_clock::now();
    LidarOdometry lidar_odometry(lidar_data_list, config);
    lidar_odometry.LoadLidars(config.lidar_path);
    CameraCenterPCD(config.odo_result_path + "/lidar_center_init.pcd", lidar_odometry.GetGlobalTranslation());
    CameraPoseVisualize(config.odo_result_path + "/lidar_pose_init.ply", lidar_odometry.GetGlobalRotation(), 
                        lidar_odometry.GetGlobalTranslation());
    // pcl::io::savePCDFileBinary(config.odo_result_path + "/lidar_fuse_init.pcd", lidar_odometry.FuseLidar(4, 0, 40));
    // 先去除畸变，这里用的就是sfm的结果
    // lidar_odometry.UndistortLidars(config.data_gap_time);

    if(!lidar_odometry.EstimatePose(config.num_iteration_lidar))
    {
        LOG(ERROR) << "lidar odometry failed";
        return 0;
    }
    t2 = chrono::high_resolution_clock::now();
    time_spent["LiDAR Odometry.estimate pose"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    
    CameraCenterPCD(config.odo_result_path + "/lidar_center_refined.pcd", lidar_odometry.GetGlobalTranslation());
    CameraPoseVisualize(config.odo_result_path + "/lidar_pose_refined.ply", lidar_odometry.GetGlobalRotation(), 
                        lidar_odometry.GetGlobalTranslation());
    
    
    pcl::io::savePCDFileBinary(config.odo_result_path + "/lidar_fuse_refined.pcd", lidar_odometry.FuseLidar(4, 0, 40));
    
    ExportPoseT(config.odo_result_path + "/lidar_pose_refined.txt", lidar_odometry.GetGlobalRotation(true),
                lidar_odometry.GetGlobalTranslation(true), lidar_odometry.GetLidarNames(true));
    /**  使用雷达位姿对雷达进行去畸变 **/
    if(true)
    {
        t1 = chrono::high_resolution_clock::now();
        int max_iter = 1;
        LOG(INFO) << "lidar mapping with undistort lidar point cloud";
        lidar_odometry.LoadLidars(config.lidar_path);
        for(int iter = 0; iter < max_iter; iter ++)
        {
            // 利用已有位姿去畸变并保存去畸变结果到本地文件
            lidar_odometry.UndistortLidars(config.data_gap_time);
            // 重置除了LiDAR本身的点云和位姿之外的所有数据
            lidar_odometry.ResetAllLidars();
            // 计算LiDAR位姿
            if(!lidar_odometry.EstimatePose(config.num_iteration_lidar))
            {
                LOG(ERROR) << "lidar odometry failed";
                return 0;
            }
            // 输出经过畸变矫正后的雷达的位姿
            vector<string> undis_names = lidar_odometry.GetLidarNames(true);
            for(string& name : undis_names)
                name.replace(name.find(config.lidar_path), config.lidar_path.size(), config.lidar_path_undistort);
            ExportPoseT(config.odo_result_path + "/lidar_pose_undis_refined.txt", lidar_odometry.GetGlobalRotation(true),
                lidar_odometry.GetGlobalTranslation(true), undis_names);
            CameraCenterPCD(config.odo_result_path + "/lidar_center_undis_refined.pcd", lidar_odometry.GetGlobalTranslation(true));
            CameraPoseVisualize(config.odo_result_path + "/lidar_center_undis_refined.ply", lidar_odometry.GetGlobalRotation(true), 
                        lidar_odometry.GetGlobalTranslation(true));
            pcl::io::savePCDFileBinary(config.odo_result_path + "/lidar_fuse_undis_refined " + num2str(iter) + ".pcd", 
                                    lidar_odometry.FuseLidar(4, 0, 40));
            // 如果当前迭代不是最后一次，就说明畸变矫正还要继续进行，那么就要重新读取原始的雷达数据
            // 这是因为当前LidarOdometry里的雷达是已经经过畸变矫正的结果了，要进行第二次畸变矫正的时候，必须还原到原本的
            // 有畸变状态，所以这里要重新读取雷达数据，读取的是原始的有畸变结果。在此结果上进行畸变矫正才是合理的
            if(iter != max_iter - 1)
                lidar_odometry.LoadLidars(config.lidar_path);
        }
        t2 = chrono::high_resolution_clock::now();
        time_spent["LiDAR Odometry.undistort"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();       
    }
    return 0;
}

int JointOptimization(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    if(!boost::filesystem::exists(config.joint_result_path))
        boost::filesystem::create_directories(config.joint_result_path);

    vector<string> image_names_sync = IterateFiles(config.image_path, ".jpg");   

    // 这里读取的雷达和图像一定要是所有的雷达，不能只是有位姿的雷达,可以选择用相机的位姿来设置雷达位姿，也可以选择用
    // 雷达位姿设置相机位姿，主要看是相信哪个位姿了。方法都是一样的
    vector<Frame> frames;
    ReadFrames(config.frame_path, config.image_path, frames, config.num_threads, true);
    vector<Velodyne> lidars;
    // 先尝试读取畸变矫正的雷达，如果不成功就读取原始的雷达
    if(!LoadLidarPose(lidars, config.odo_result_path + "/lidar_pose_undis_refined.txt", true))
        LoadLidarPose(lidars, config.odo_result_path + "/lidar_pose_refined.txt", true);

    // 用雷达位姿设置相机位姿
    if(true)
    {
        SetFramePose(frames, lidars, config.T_cl, config.time_offset, config.data_gap_time);       
    }
    // 用相机位姿设置雷达位姿
    else 
    {
        // 用图像位姿设置雷达位姿
        LOG(INFO) << "Use T_cl to set lidar pose";
        // 设置图像位姿
        LoadFramePose(frames, config.sfm_result_path + "/camera_pose_final.txt");
        SetLidarPose(frames, lidars, config.T_cl, config.time_offset, config.data_gap_time);
    }

    #if 0
    // 输出一下图像的id和图像名之间的对应关系，这样方便debug
    ofstream image_id_name(config.result_path + "/image_id_name.txt");
    for(size_t i = 0; i < image_names_sync.size(); i++)
        image_id_name << i << "  " << image_names_sync[i] << endl;
    image_id_name.close();

    ofstream lidar_id_name(config.result_path + "/lidar_id_name.txt");
    for(size_t i = 0; i < lidar_names_sync.size(); i++)
        lidar_id_name << i << "  " << lidar_names_sync[i] << endl;
    lidar_id_name.close();
    #endif

    t1 = chrono::high_resolution_clock::now();
    CameraLidarOptimizer optimizer(config.T_cl, lidars, frames, config);
    optimizer.SetOptimizationMode(MAPPING);
    optimizer.ExtractImageLines(config.image_line_path, true);
    optimizer.ExtractLidarLines();

    pcl::io::savePCDFileBinary(config.joint_result_path + "/lidar_fuse_init.pcd", optimizer.FuseLidar(4, 0, 40));

    optimizer.JointOptimize(true);
    ExportPoseT(config.joint_result_path + "/lidar_pose_joint.txt", 
                optimizer.GetLidarRotation(true), optimizer.GetLidarTranslation(true), optimizer.GetLidarNames(true));
    ExportPoseT(config.joint_result_path + "/camera_pose_joint.txt", 
                optimizer.GetCameraRotation(true), optimizer.GetCameraTranslation(true), optimizer.GetImageNames(true));
    optimizer.ExportStructureBinary(config.joint_result_path + "/points.bin");
    pcl::io::savePCDFileBinary(config.joint_result_path + "/lidar_fuse_final.pcd", optimizer.FuseLidar(4, 0, 40));
    t2 = chrono::high_resolution_clock::now();

    time_spent["camera-lidar optimization"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    return 0;
}

int ColorizeLidarMap(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    if(!boost::filesystem::exists(config.texture_result_path))
        boost::filesystem::create_directories(config.texture_result_path);
    // 进行初始的frame和LiDAR的名字的设置，这样之后在读取位姿的时候就根据name字段来判断是否一致
    vector<Velodyne> lidars;
    vector<Frame> frames;
    // ReadFrames(config.frame_path, config.image_path, frames, config.num_threads, true);
    // LoadFramePose(frames, config.joint_result_path + "/camera_pose_joint.txt");
    // LoadLidarPose(lidars, config.joint_result_path + "/lidar_pose_joint.txt", true);
    ReadFrames(config.sfm_result_path + "frames_after_RA", config.image_path, frames, config.num_threads, true);
    LoadLidarPose(lidars, config.joint_result_path + "/lidar_pose_joint.txt", true);
    LoadFramePose(frames, config.joint_result_path + "/camera_pose_joint.txt");
    // SetFramePose(frames, lidars, config.T_cl, config.time_offset);

    t1 = chrono::high_resolution_clock::now();
    Texture texture(lidars, frames, config);
    texture.ColorizeLidarPointCloud(1.5, 35);
    pcl::io::savePCDFileBinary(config.texture_result_path + "/lidar_colored_fuse.pcd", texture.FuseCloud(4));
    t2 = chrono::high_resolution_clock::now();
    time_spent["colorize lidar cloud"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    return 0;
}

int JointMVS(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    omp_set_num_threads(config.num_threads);
    if(!boost::filesystem::exists(config.mvs_result_path))
        boost::filesystem::create_directories(config.mvs_result_path);
    if(!boost::filesystem::exists(config.mvs_normal_path))
        boost::filesystem::create_directories(config.mvs_normal_path);
    if(!boost::filesystem::exists(config.mvs_conf_path))
        boost::filesystem::create_directories(config.mvs_conf_path);
    if(!boost::filesystem::exists(config.mvs_depth_path))
        boost::filesystem::create_directories(config.mvs_depth_path);

    LOG(INFO) << "MVS config : \n"  
                << "\t \t propagate strategy : " << (config.propagate_strategy == Propagate::SEQUENTIAL ? "sequential" : "checker board") << "\n" 
                << "\t \t rescale image : " << config.scale << "\n"
                << "\t \t ncc half window : " << config.ncc_half_window << "\n"  
                << "\t \t ncc step : " << config.ncc_step << "\n" 
                << "\t \t min segment size : " << config.min_segment << "\n" 
                << "\t \t use lidar depth : " << (config.mvs_use_lidar ? "true" : "false") << "\n" 
                << "\t \t keep lidar depth constant : " << (config.keep_lidar_constant ? "true" : "false") << "\n" 
                << "\t \t use geometric consistency : " << (config.mvs_use_geometric ? "true" : "false") << "\n" 
                ;
    
    /**
     * 1. 读取Frame以及相应的位姿，Frame位姿可以是sfm结果，也可以是联合优化的结果 
     * 同时设置了Frame的尺度，降低分辨率，提升程序运行速度
    */
    vector<Frame> frames;
    // ReadFrames(config.frame_path, config.image_path, frames, config.num_threads, true);
    ReadFrames(config.sfm_result_path + "frames_after_RA", config.image_path, frames, config.num_threads, true);
    // LoadFramePose(frames, config.joint_result_path + "/camera_pose_joint.txt");
    // LoadFramePose(frames, config.sfm_result_path + "camera_pose_final.txt");
    for(Frame& f : frames)
        f.SetImageScale(config.scale);
    
    /**
     * 2. 读取LiDAR以及相应的位姿，LiDAR位姿也有很多种选择，读取odometry的结果，读取joint的结果都行，
     * 甚至还可以用图像的结果然后根据初始的T_cl设置雷达位姿
    */
    vector<Velodyne> lidars;
    LoadLidarPose(lidars, config.joint_result_path + "/lidar_pose_joint.txt", true);
    LoadFramePose(frames, config.joint_result_path + "/camera_pose_joint.txt");
    // LoadLidarPose(lidars, config.odo_result_path + "/lidar_pose_undis_refined.txt", true);
    // LoadLidarPose(lidars, config.sfm_result_path + "lidar_pose.txt", true);
    // SetLidarPose(frames, lidars, config.T_cl, config.time_offset);
    // SetFramePose(frames, lidars, config.T_cl, config.time_offset);
  

    /**
     * 3. 对mask进行resize，使其适合图像尺寸
    */
    // 读取mask
    cv::Mat mvs_mask;
    if(!config.mask_path.empty())
    {
        mvs_mask = cv::imread(config.mask_path, cv::IMREAD_GRAYSCALE);
        cv::threshold(mvs_mask, mvs_mask, 0.1, 1, cv::THRESH_BINARY);
    }

    if(!mvs_mask.empty())
    {
        int s = frames[0].GetImageScale();
        if(s > 0)
            for(; s > 0; s--)
                cv::pyrUp(mvs_mask, mvs_mask);
        else if(s < 0)
            for(; s < 0; s++)
                cv::pyrDown(mvs_mask, mvs_mask);
        mvs_mask.convertTo(mvs_mask, CV_32F);
    }
    else 
        mvs_mask = cv::Mat::ones(frames[0].GetImageRows(), frames[0].GetImageCols(), CV_32F);

    
    MVS mvs(frames, lidars, config);

    /**
     * 4. 读取三维点或者是自己重新三角化一次三维点
    */
    mvs.LoadStructure(config.joint_result_path + "points.bin");
    // mvs.LoadStructure(config.sfm_result_path + "points.bin");
    // mvs.EstimateStructure();

    /**
     * 5. 对相机位姿进行单独优化，因为MVS结果受相机位姿影响很大，所以单独优化一下位姿，保证最好的效果
    */
    t1 = chrono::high_resolution_clock::now(); 
    mvs.RefineCameraPose();
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.refine camera pose"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    /**
     * 6. 对每张图像选择近邻图像
    */
    mvs.SelectNeighborViews(5);
    
    /**
     * 7. 计算深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.EstimateDepthMaps(config.propagate_strategy, mvs_mask);
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.estimate depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    /**
     * 8. 过滤深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.FilterDepthMaps();
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.filter depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    /**
     * 9. 融合深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.FuseDepthMaps();
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.fuse depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    return 0;
}

int DependentMVS(const Config& config, map<string, double>& time_spent)
{
    // 两个计时器
    auto t1 = chrono::high_resolution_clock::now();
    auto t2 = chrono::high_resolution_clock::now();

    omp_set_num_threads(config.num_threads);
    if(!boost::filesystem::exists(config.mvs_result_path))
        boost::filesystem::create_directories(config.mvs_result_path);
    if(!boost::filesystem::exists(config.mvs_normal_path))
        boost::filesystem::create_directories(config.mvs_normal_path);
    if(!boost::filesystem::exists(config.mvs_conf_path))
        boost::filesystem::create_directories(config.mvs_conf_path);
    if(!boost::filesystem::exists(config.mvs_depth_path))
        boost::filesystem::create_directories(config.mvs_depth_path);

    LOG(INFO) << "MVS config : \n"  
            << "\t \t propagate strategy : " << (config.propagate_strategy == Propagate::SEQUENTIAL ? "sequential" : "checker board") << "\n" 
            << "\t \t rescale image : " << config.scale << "\n"
            << "\t \t ncc half window : " << config.ncc_half_window << "\n"  
            << "\t \t ncc step : " << config.ncc_step << "\n" 
            << "\t \t min segment size : " << config.min_segment << "\n" 
            << "\t \t use lidar depth : " << (config.mvs_use_lidar ? "true" : "false") << "\n" 
            << "\t \t keep lidar depth constant : " << (config.keep_lidar_constant ? "true" : "false") << "\n" 
            << "\t \t use geometric consistency : " << (config.mvs_use_geometric ? "true" : "false") << "\n" 
            ;

    // 1. 读取所有的图像并生成frame
    vector<Frame> frames;
    vector<string> image_names = IterateFiles(config.image_path, ".jpg");
    cv::Mat img = cv::imread(image_names[0]);
    for(int i = 0; i < image_names.size(); i++)
    {
        Frame f(img.rows, img.cols, i, image_names[i]);
        frames.push_back(f);
    }
    LoadFramePose(frames, config.camera_pose_prior_path);
    for(Frame& f : frames)
        f.SetImageScale(config.scale);

    // 2. 读取所有的LiDAR并生成lidar
    vector<Velodyne> lidars;

    /**
     * 3. 对mask进行resize，使其适合图像尺寸
    */
    // 读取mask
    cv::Mat mvs_mask;
    if(!config.mask_path.empty())
    {
        mvs_mask = cv::imread(config.mask_path, cv::IMREAD_GRAYSCALE);
        cv::threshold(mvs_mask, mvs_mask, 0.1, 1, cv::THRESH_BINARY);
    }

    if(!mvs_mask.empty())
    {
        int s = frames[0].GetImageScale();
        if(s > 0)
            for(; s > 0; s--)
                cv::pyrUp(mvs_mask, mvs_mask);
        else if(s < 0)
            for(; s < 0; s++)
                cv::pyrDown(mvs_mask, mvs_mask);
        mvs_mask.convertTo(mvs_mask, CV_32F);
    }
    else 
        mvs_mask = cv::Mat::ones(frames[0].GetImageRows(), frames[0].GetImageCols(), CV_32F);

    
    MVS mvs(frames, lidars, config);

    /**
     * 4. 对每张图像选择近邻图像
    */
    mvs.SelectNeighborViews(5);
    
    /**
     * 5. 计算深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.EstimateDepthMaps(config.propagate_strategy, mvs_mask);
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.estimate depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    /**
     * 6. 过滤深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.FilterDepthMaps();
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.filter depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    /**
     * 7. 融合深度图
    */
    t1 = chrono::high_resolution_clock::now();
    mvs.FuseDepthMaps();
    t2 = chrono::high_resolution_clock::now();
    time_spent["MVS.fuse depth map"] = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();

    return 0;
}

void TimeReport( map<string, double> time_spent)
{
    double total_time = 0;
    map<string, double>::iterator it = time_spent.begin();
    for(; it != time_spent.end(); it++)
    {
        if(it->second > 0.001)
        {
            LOG(INFO) << it->first << " : " << it->second << " s" << endl;
            total_time += it->second;
        }
        else 
        {
            LOG(INFO) << it->first << " : 0 s" << endl;
        }
    }
    LOG(INFO) << "total time: " << total_time << " s" << endl;
}

void SetLog(string log_path)
{
    if(!boost::filesystem::exists(log_path))
        boost::filesystem::create_directories(log_path);
    google::InitGoogleLogging("Mapping");
    google::SetLogDestination(google::GLOG_INFO, log_path.c_str());
    google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_logbufsecs = 0;
    LOG(INFO) << "Save log file at " << log_path;
}