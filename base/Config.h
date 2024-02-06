/*
 * @Author: Diantao Tu
 * @Date: 2021-12-26 15:53:10
 */
#ifndef _CONFIG_H_
#define _CONFIG_H_
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <glog/logging.h>
#include <Eigen/Geometry>
#include <omp.h>

#include "common.h"

using namespace boost::program_options;
using namespace std;

struct Config{

private:
    string T_cl_string;
public:
    /***************************************
        相关的路径，每一个路径都不能为空 
    ****************************************/
    // 图像文件目录
    string image_path;  
    // 雷达文件目录 
    string lidar_path;    
    // 深度图文件目录，如果这个目录里深度图数量和图像文件数目不符合，那么之后就要更新这个目录里的深度图
    string depth_path; 
    // 保存frame文件的目录，如果这个目录里frame数量和图像文件数目不符，之后要更新这个目录里的文件   
    string frame_path;
    // 保存图像匹配对的目录，可以把图像的匹配关系保存在这里，方便下一次使用，这里保存的应该是经过相对位姿估计之后的匹配对
    string match_pair_path;
    // 保存匹配对的目录，这个目录保存的是经过完整的SfM之后，依然保留下来的匹配对，这个匹配对可以用于后面的联合优化 joint optimization
    string match_pair_joint_path;
    // 保存从图像上提取的直线特征，这样就可以重复利用了
    string image_line_path;
    // 保存经过去畸变的点云
    string lidar_path_undistort;
    // 保存MVS的深度图，法向量图，置信度图的目录
    string mvs_data_path;
    // 保存运行结果的目录
    string result_path;
    // mask 所在的文件目录
    string mask_path;
    // GPS 文件路径（可以为空）
    string gps_path;
    // 预先计算好的相机位姿（用于单独进行MVS,可以为空）
    string camera_pose_prior_path;
    // 各种步骤的结果保存的路径
    string sfm_result_path;
    string odo_result_path; 
    string joint_result_path;
    string calib_result_path;
    string texture_result_path;
    string mvs_result_path;
    string mvs_depth_path;
    string mvs_normal_path;
    string mvs_conf_path;
    
    int num_threads = -1;                       // 多线程数目
    float data_gap_time = 0;                    // 相邻图像（雷达）之间的时间间隔，不是雷达每一帧的扫描时间
    bool use_cuda = false;                      // 是否使用cuda
    float max_depth = 20;                       // 深度补全和MVS的最大深度
    float min_depth = 0.1;
    float max_depth_visual = 20;                // 用于深度图可视化时候的最大深度

    /***************************************
        LiDAR特征提取相关的配置参数 
    ****************************************/
    int extraction_method = 3;                  // 提取特征的方法，主要是直线特征提取方法的不同
    float max_curvature = 5;                    // 曲率的上限，超过这个值的曲率将被视为噪声，值越大提取的特征越多
    float intersection_angle_threshold = 5;     // 雷达点和邻域平面的夹角，超过这个值才认为可靠，值越小提取的特征越多
    bool ground_segment = true;                 // 提取特征时是否把地面分割出来，这个参数没用，因为效果并不好
    bool lidar_segmentation = true;             // 提取特征时是否对点云进行分割，去除小的物体
    /***************************************
        图像直线特征提取相关的配置参数 
    ****************************************/
    float ncc_threshold = -0.2;                 // NCC 阈值，低于这个值就认为两个点不属于同一个直线，NCC 取值范围[-1,1]
    /***************************************
        SIFT 特征提取与匹配相关的配置参数 
    ****************************************/
    bool root_sift = true;
    int num_sift = 8096;
    float sift_match_dist_threshold = 0.6;      // KNN匹配时第一匹配点和第二匹配之间距离之比
    int sift_match_num_threshold = 40;
    /***************************************
        相对位姿估计的配置参数 
    ****************************************/
    bool keep_pairs_no_scale = true;
    Eigen::Matrix4d T_cl = Eigen::Matrix4d::Identity();       // LiDAR到相机的相对位姿
    double time_offset = 0;                     // LiDAR和相机的时偏，大于0代表雷达先于相机，小于0代表雷达晚于相机，以秒为单位
    /***************************************
        SfM的相关参数
    ****************************************/
    int rotation_averaging_method = 1;
    int translation_averaging_method = 1;
    bool use_all_pairs_ra = true;               // 在旋转平均的时候使用所有的图像对，设为false则只使用有尺度的图像对
    bool use_all_pairs_ta = true;               // 在平移平均的时候使用所有的图像对，设为false则只使用有尺度的图像对
    bool init_translation_DLT = true;
    bool init_translation_GPS = false;
    int num_iteration_L2IRLS = 10;  
    float upper_scale_ratio = 1.3;              // 在进行优化的时候相对平移的尺度的上限（这个上限是可以被突破的，但是突破会有惩罚） upper scale = ratio*scale 
    float lower_scale_ratio = 0.9;              // 相对平移的尺度的下限
    float triangulate_angle_threshold = 25;     // 初始三角化匹配点的时候的阈值
    bool colorize_structure = true;             // 是否对三角化的点上色
    /***************************************
     *  LiDAR-LiDAR残差的设置，适用于lidarOdometry和LiDAR-Camera联合优化
    ****************************************/
    bool angle_residual = true;                 // 是否使用角度作为残差，如果不用角度就用米
    bool point_to_line_residual = true;
    float point_to_line_dis_threshold = 0.3;    // 进行点到线的特征匹配的时候，距离的阈值，超过这个阈值就不匹配了
    bool line_to_line_residual = true;
    bool point_to_plane_residual = true;
    float point_to_plane_dis_threshold = 1.0;   // 进行点到面的特征匹配的时候的距离阈值，超过这个阈值就认为不匹配了
    float lidar_plane_tolerance = 0.03;         // 判断LiDAR点是否形成平面时，对平面的判断条件，越低平面越平整，提取的点到平面约束越少
    bool normalize_distance = true;             // 在使用角度作为残差时，是否对距离进行归一化，适用于点到线以及点到面
    /***************************************
        LiDAR单独优化时的参数，也就是lidarOdometry
    ****************************************/
    int num_iteration_lidar = 5;
    /***************************************
        LiDAR-Camera联合优化时的参数
    ****************************************/
    int num_iteration_joint = 1;
    int neighbor_size_joint = 3;                // 每个图像的近邻的雷达数
    float camera_weight = 1.0;                  // camera-camera的权重
    float lidar_weight = 1.0;                   // LiDAR-LiDAR的权重
    float camera_lidar_weight = 1.0;            // camera-LiDAR的权重
    /***************************************
        MVS使用的参数
    ****************************************/
    bool mvs_use_lidar = true;                  // 使用雷达作为深度图的初值
    int scale = 0;                              // 图像上采样或降采样，1代表上采样一次，-1代表降采样一次，以此类推
    int ncc_half_window = 11;                   // 计算NCC时半窗口尺寸
    int ncc_step = 2;                           // 计算NCC时窗口采样间隔，间隔为step-1
    int propagate_strategy = 1;                 // 邻域传播的方法，1 - 棋盘格， 2 - 顺序传播
    float depth_diff_threshold = 0.01;          // 判定两个相邻的像素深度不连续的阈值，这个阈值是比例，不是绝对深度
    int min_segment = 50;                       // 去除深度图的独立小区域时的阈值
    bool mvs_use_geometric = true;              // 传播时是否使用几何一致性
    bool keep_lidar_constant = false;           // 传播时固定雷达深度不变
    
    Config()
    {
        LOG(INFO) << "Use default parameter to init config";
    }
    Config(std::string file_path)
    {
        boost::program_options::options_description config("Options");
        config.add_options()
        ("image_path", value<string>(&image_path),"path to load image folder")  
        ("lidar_path", value<string>(&lidar_path),"path to load lidar folder")    
        ("depth_path", value<string>(&depth_path),"path to save/load depth image for SfM") 
        ("frame_path", value<string>(&frame_path),"path to save/load frame data")
        ("match_pair_path", value<string>(&match_pair_path),"path to save/load image pair data for SfM")
        ("match_pair_joint_path", value<string>(&match_pair_joint_path),"path to save/load image pair data for joint optimization")
        ("image_line_path", value<string>(&image_line_path), "path to save/load image lines")
        ("lidar_path_undistort", value<string>(&lidar_path_undistort),"path to save undistort lidar")
        ("mvs_data_path", value<string>(&mvs_data_path), "path to save/load MVS data (depth, normal, confidence)")
        ("result_path", value<string>(&result_path), "result saving folder for sfm result and lidar odometry result")
        ("mask_path", value<string>(&mask_path), "mask path")
        ("gps_path", value<string>(&gps_path), "gps path (optional)")
        ("camera_pose_prior_path", value<string>(&camera_pose_prior_path), "camera pose prior path (optional)")
        ("num_threads", value<int>(&num_threads), "threads number")
        ("data_gap_time", value<float>(&data_gap_time), "time interval between consecutive data frames")
        ("use_cuda", value<bool>(&use_cuda), "set true to use cuda")
        ("max_depth", value<float>(&max_depth), "max depth for depth completion and MVS")
        ("min_depth", value<float>(&min_depth), "min depth for MVS")
        ("max_depth_visual", value<float>(&max_depth_visual), "max depth for a better visualization result")



        ("extraction_method", value<int>(&extraction_method), "extract edge features method")
        ("max_curvature", value<float>(&max_curvature), "maxium curvature in lidar feature extraction")
        ("intersection_angle_threshold", value<float>(&intersection_angle_threshold), "angle threshold in livox loam paper, equation(4)")
        ("ground_segment", value<bool>(&ground_segment), "segment ground points")
        ("lidar_segmentation", value<bool>(&lidar_segmentation), "segment lidar points")

        ("ncc_threshold", value<float>(&ncc_threshold), "ncc threshold")

        ("root_sift", value<bool>(&root_sift), "use root sift or not")
        ("num_sift", value<int>(&num_sift), "number of sift extracted in each image")
        ("sift_match_dist_threshold", value<float>(&sift_match_dist_threshold), "first match and second match distance threshold")
        ("sift_match_num_threshold", value<int>(&sift_match_num_threshold), "image pair with match sift threshold")

        ("keep_pairs_no_scale", value<bool>(&keep_pairs_no_scale), "keep image pairs with no scale")
        ("T_cl", value<string>(&T_cl_string), "T_cl, qw qx qy qz tx ty tz")
        ("time_offset", value<double>(&time_offset), "time offset between LiDAR and camera, in seconds")

        ("rotation_averaging_method", value<int>(&rotation_averaging_method), "rotation averaging method")
        ("translation_averaging_method", value<int>(&translation_averaging_method), "translation averaging method")
        ("use_all_pairs_ra", value<bool>(&use_all_pairs_ra), "use all image pairs in rotation averaging or use pairs with scale")
        ("use_all_pairs_ta", value<bool>(&use_all_pairs_ta), "use all image pairs in translation averaging or use pairs with scale")
        ("init_translation_DLT", value<bool>(&init_translation_DLT), "use DLT method to init a global translation")
        ("init_translation_GPS", value<bool>(&init_translation_GPS), "use GPS to init a global translation")
        ("num_iteration_L2IRLS", value<int>(&num_iteration_L2IRLS), "max iteration in translation averaging L2IRLS")
        ("upper_scale_ratio", value<float>(&upper_scale_ratio), "upper scale = ratio*scale, in translaton averaging L2 L2IRLS SOFTL1")
        ("lower_scale_ratio", value<float>(&lower_scale_ratio), "lower scale = ratio*scale, in translaton averaging L2 L2IRLS SOFTL1")

        ("triangulate_angle_threshold", value<float>(&triangulate_angle_threshold), "angle threshold in triangulate N view")

        ("colorize_structure", value<bool>(&colorize_structure), "colorize structure or not")

        ("angle_residual", value<bool>(&angle_residual), "use angle as a measure of residuals")
        ("point_to_line_residual", value<bool>(&point_to_line_residual), "use point to line residual")
        ("point_to_line_dis_threshold", value<float>(&point_to_line_dis_threshold), "distance threshold for outlier")
        ("line_to_line_residual", value<bool>(&line_to_line_residual), "use line to line residual")
        ("point_to_plane_residual", value<bool>(&point_to_plane_residual), "use point to plane residual")
        ("point_to_plane_dis_threshold", value<float>(&point_to_plane_dis_threshold), "distance threshold for outlier")
        ("lidar_plane_tolerance", value<float>(&lidar_plane_tolerance), "plane tolerance in point to plane ")
        ("normalize_distance", value<bool>(&normalize_distance), "normalize point to line / plane distance when using angle residual")

        ("num_iteration_lidar", value<int>(&num_iteration_lidar), "max iteration in lidar optimize")

        ("num_iteration_joint", value<int>(&num_iteration_joint), "max iteration in joint optimize")
        ("neighbor_size_joint", value<int>(&neighbor_size_joint), "neighbor size in joint optimize")
        ("camera_weight", value<float>(&camera_weight), "camera-camera weight")
        ("lidar_weight", value<float>(&lidar_weight), "lidar-lidar weight")
        ("camera_lidar_weight", value<float>(&camera_lidar_weight), "camera-LiDAR weight")
        
        ("mvs_use_lidar", value<bool>(&mvs_use_lidar), "use lidar depth in MVS depth estimation")
        ("scale", value<int>(&scale), "scale for resize image")
        ("ncc_half_window", value<int>(&ncc_half_window), "half window size for ncc computing")
        ("ncc_step", value<int>(&ncc_step), "step in ncc computing")
        ("propagate_strategy", value<int>(&propagate_strategy), "propagate strategy in MVS, 1 - checker board, 2 - sequential")
        ("depth_diff_threshold", value<float>(&depth_diff_threshold), "max depth difference ")
        ("min_segment", value<int>(&min_segment), "minimum size of a segment")
        ("mvs_use_geometric", value<bool>(&mvs_use_geometric), "use geometric consistency in MVS depth estimation")
        ("keep_lidar_constant", value<bool>(&keep_lidar_constant), "keep lidar depth constant in propagation")
        ;

        boost::program_options::variables_map vm;
        std::ifstream input(file_path);
        if(!input)
        {
            LOG(WARNING) << "Fail to open config file : " << file_path << ", use default config";
            return;
        }
        try
        {
            boost::program_options::store(boost::program_options::parse_config_file(input, config), vm);
            boost::program_options::notify(vm);
        }
        catch(const std::exception& e)
        {
            LOG(ERROR) << e.what() << '\n';
            exit(0);
        }
        sfm_result_path = result_path + "/sfm/";
        odo_result_path = result_path + "/odometry/";
        joint_result_path = result_path + "/joint/";
        calib_result_path = result_path + "/calib/";
        texture_result_path = result_path + "/texture/";
        mvs_result_path = result_path + "/mvs/";
        mvs_depth_path = mvs_data_path + "/depth/";
        mvs_normal_path = mvs_data_path + "/normal/";
        mvs_conf_path = mvs_data_path + "/conf/";
        if(num_threads <= 0 || num_threads > omp_get_max_threads())
            num_threads = omp_get_max_threads();
        
        vector<string> T_cl_split = SplitString(T_cl_string, ' ');
        // qw qx qy qz tx ty tz
        if(T_cl_split.size() == 7)
        {
            Eigen::Quaterniond q_cl(str2num<double>(T_cl_split[0]), str2num<double>(T_cl_split[1]),
                                    str2num<double>(T_cl_split[2]), str2num<double>(T_cl_split[3]));
            T_cl.block<3,3>(0,0) = Eigen::Matrix3d(q_cl);
            T_cl(0,3) = str2num<double>(T_cl_split[4]);
            T_cl(1,3) = str2num<double>(T_cl_split[5]);
            T_cl(2,3) = str2num<double>(T_cl_split[6]);
        }
        // 12个数字就是旋转矩阵+平移向量
        if(T_cl_split.size() == 12)
        {
            T_cl << str2num<double>(T_cl_split[0]), str2num<double>(T_cl_split[1]), str2num<double>(T_cl_split[2]), str2num<double>(T_cl_split[3]),
                    str2num<double>(T_cl_split[4]), str2num<double>(T_cl_split[5]), str2num<double>(T_cl_split[6]), str2num<double>(T_cl_split[7]),
                    str2num<double>(T_cl_split[8]), str2num<double>(T_cl_split[9]), str2num<double>(T_cl_split[10]), str2num<double>(T_cl_split[11]),
                    0,          0,         0,          1;
        }
    }
};


#endif