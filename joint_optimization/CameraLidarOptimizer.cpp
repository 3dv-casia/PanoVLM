/*
 * @Author: Diantao Tu
 * @Date: 2021-11-18 17:24:35
 */

#include "CameraLidarOptimizer.h"

using namespace std;

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4d _T_cl):T_cl_init(_T_cl)
{

}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4f _T_cl)       
{
    T_cl_init = _T_cl.cast<double>();
}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4d _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config):
                        T_cl_init(_T_cl), lidars(lidars), frames(frames), config(_config)
{}

CameraLidarOptimizer::CameraLidarOptimizer(const Eigen::Matrix4f _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config):
                        lidars(lidars), frames(frames), config(_config)
{
    T_cl_init = _T_cl.cast<double>();
}

int CameraLidarOptimizer::Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>&line_pairs, 
                        const Eigen::Matrix4d& T_cl)
{
    ceres::Problem problem;
    ceres::LossFunction *  loss_function = new ceres::HuberLoss(2.0 * M_PI / 180.0);
    Eigen::Matrix3d R = T_cl.block<3,3>(0,0);
    Eigen::Vector3d t = T_cl.block<3,1>(0,3);
    Eigen::Vector3d angle_axis;
    ceres::RotationMatrixToAngleAxis(R.data(), angle_axis.data());
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs.begin();
                it != line_pairs.end(); it++)
    {
        const vector<CameraLidarLinePair>& line_pair = it->second;
        for(const CameraLidarLinePair& pair : line_pair)
        {
            const cv::Vec4f& l = pair.image_line;
            // 把起始点和终止点都变换成单位圆上的XYZ坐标
            cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), float(5.0));
            cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), float(5.0));
            // cout << p1.x * p1.x + p1.y * p1.y + p1.z * p1.z << " " << p2.x * p2.x + p2.y * p2.y + p2.z * p2.z <<endl;
            cv::Point3f p3(0,0,0);
            double a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
            double b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
            double c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );

            ceres::CostFunction *cost_function = Plane2Plane_Relative::Create(Eigen::Vector3d(a,b,c), pair.lidar_line_end, pair.lidar_line_start);
            problem.AddResidualBlock(cost_function, loss_function, angle_axis.data(), t.data());

            ceres::CostFunction *cost_function2 = PlaneRelativeIOUResidual::Create(
                        Eigen::Vector4d(a,b,c,0), (pair.lidar_line_start + pair.lidar_line_end) / 2.0, p1, p2, 2);
            problem.AddResidualBlock(cost_function2, nullptr, angle_axis.data(), t.data());
        }

    }

    LOG(INFO) << "total residual: " <<problem.NumResidualBlocks() << endl;
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.max_linear_solver_iterations = 100;
    options.preconditioner_type = ceres::JACOBI;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 10;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.BriefReport() << '\n';
    // std::cout << summary.IsSolutionUsable() << std::endl;
    ceres::AngleAxisToRotationMatrix(angle_axis.data(), R.data());
    T_cl_optimized = Eigen::Matrix4d::Identity();
    T_cl_optimized.block<3,3>(0,0) = R;
    T_cl_optimized.block<3,1>(0,3) = t;

    return 1;
}

bool CameraLidarOptimizer::ExtractImageLines(string image_line_folder, bool visualization)
{
    const size_t length = frames.size();
    // 先从本地读取图像直线，如果失败了就重新计算
    if(ReadPanoramaLines(image_line_folder, config.image_path, image_lines_all))
        return true;
    image_lines_all.reserve(length);
    LOG(INFO) << "Extract image lines for " << length << " images";
    if(length == 0)
        return false;
    vector<string> depth_image_names = IterateFiles(config.depth_path, ".bin");
    cv::Mat img_mask;
    if(!config.mask_path.empty())
        img_mask = cv::imread(config.mask_path, cv::IMREAD_GRAYSCALE);
    ProcessBar bar(length, 0.2);
    omp_set_num_threads(5);
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < length; i++)
    {
        cv::Mat img_gray = frames[i].GetImageGray();
        PanoramaLine detect(img_gray, i);
        detect.SetName(frames[i].name);
        // 用深度图来指导图像直线提取，效果并不好
        // if(!depth_image_names.empty())
        // {
        //     cv::Mat depth;
        //     ReadOpenCVMat(depth_image_names[i], depth);
        //     depth.convertTo(depth, CV_32F);
        //     depth /= 256.0;
        //     detect.SetDepthImage(depth);
        // }
        if(!img_mask.empty())
            detect.Detect(img_mask);
        else 
            detect.Detect(45, -45);
        detect.Fuse(config.ncc_threshold);
        // 显示提取出的直线，用于debug
        if(visualization)
        {
            vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                        cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                        cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                        cv::Scalar(255,0,255)};                         // 紫
            cv::Mat img_line = DrawLinesOnImage(img_gray, detect.GetLines(), colors, 6, true);
            if(optimization_mode == CALIBRATION)
                cv::imwrite(config.calib_result_path + "/img_line_filtered" + num2str(frames[i].id) + ".jpg", img_line);
            else 
                cv::imwrite(config.joint_result_path + "/img_line_filtered" + num2str(frames[i].id) + ".jpg", img_line);
        }
        #pragma omp critical
        {
            image_lines_all.push_back(detect);
        }
        bar.Add();
    }
    // 对图像直线进行排序
    sort(image_lines_all.begin(), image_lines_all.end(), [this](PanoramaLine& a, PanoramaLine& b){return a.id < b.id;});
    // 保存图像直线，下次可以直接使用
    ExportPanoramaLines(image_line_folder, image_lines_all);
    return true;
}

bool CameraLidarOptimizer::ExtractLidarLines(bool visualization)
{
    const size_t length = lidars.size();

    LOG(INFO) << "Extract lidar lines for " << length << " lidar data";
    omp_set_num_threads(config.num_threads);
    #pragma omp parallel for
    for(size_t i = 0; i < length; i++)
    {
        // 如果某一帧雷达没有特征点，那么大概率是这一帧雷达没有进行特征提取，也有可能是根本就没有读点云
        // 无论那种情况，都是重新搞一遍，可能会慢一点，但是肯定没错误（又不需要实时性）
        if(lidars[i].cornerLessSharp.empty())
        {
            lidars[i].LoadLidar(lidars[i].name);
            lidars[i].ReOrderVLP();
            lidars[i].ExtractFeatures(config.max_curvature, config.intersection_angle_threshold, 
                                    config.extraction_method, config.lidar_segmentation);
        }
        if(visualization)
            lidars[i].SaveFeatures(config.joint_result_path);
        if(lidars[i].IsInWorldCoordinate())
            lidars[i].Transform2Local();    
    }
    return true;
}

bool CameraLidarOptimizer::JointOptimize(bool visualization)
{
    LOG(INFO) << "============== Camera-LiDAR joint optimization begin ======================";
    LOG(INFO) << "Configuration: use angle - " << (config.angle_residual ? "true\r\n" : "false\r\n") << 
            "\t\t use point to line - " << (config.point_to_line_residual ? "true\r\n" : "false\r\n") << 
            "\t\t use line to line - " << (config.line_to_line_residual ? "true\r\n" : "false\r\n") <<
            "\t\t use point to plane - " << (config.point_to_plane_residual ? "true\r\n" : "false\r\n") <<
            "\t\t lidar plane tolerance - " << config.lidar_plane_tolerance << "\r\n" << 
            "\t\t use normalized distance - " << (config.normalize_distance ? "true\r\n" : "false\r\n") << 
            "\t\t camera-camera weight - " << config.camera_weight << "\r\n" << 
            "\t\t LiDAR-LiDAR weight - " << config.lidar_weight << "\r\n" << 
            "\t\t camera-LiDAR weight - " << config.camera_lidar_weight;
            

    // 在进行直线关联的时候需要考虑一下之后要做什么，如果要进行标定任务（calibration），那么只需要
    // 对单帧进行直线关联，因为相当于雷达和相机是同步的，两者只有一个外参
    // 如果是要进行多相机多雷达的整体位姿优化，就要考虑一张图像和多个雷达进行关联，因为这时候相机和雷达
    // 不再同步，可以认为有n个外参（n=相机数 或 n=雷达数）
    if(optimization_mode == CALIBRATION)
    {
        LOG(INFO) << "start camera - LiDAR calibration";
        if(lidars.size() != frames.size())
        {
            const size_t size = min(lidars.size(), frames.size());
            LOG(WARNING) << "In calibration mode, num lidar != num frame, resize them to " << size;
            lidars = vector<Velodyne>(lidars.begin(), lidars.begin() + size);
            frames = vector<Frame>(frames.begin(), frames.begin() + size);
        }
        Eigen::Matrix4d T_cl_last = T_cl_init;
        eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all = AssociateLineSingle(T_cl_last);
        // 显示投影结果，用于debug
        if(visualization)
        {
            Visualize(line_pairs_all, config.calib_result_path + "/init/");
        }
        for(int iter = 0; iter < 35; iter++)
        {
            LOG(INFO) << "iteration: " << iter ;
            
            // 优化外参
            Optimize(line_pairs_all, T_cl_last);
            // 计算外参的变化量
            Eigen::Matrix3d rotation_last = T_cl_last.block<3,3>(0,0);
            Eigen::Vector3d trans_last = T_cl_last.block<3,1>(0,3);
            Eigen::Matrix3d rotation_curr = T_cl_optimized.block<3,3>(0,0);
            float rotation_change = acos(((rotation_last.transpose() * rotation_curr).trace() - 1) / 2.0);
            rotation_change *= 180.0 / M_PI;
            float trans_change = (trans_last - T_cl_optimized.block<3,1>(0,3)).norm();
            // 更新变化之后的外参
            T_cl_last = T_cl_optimized;
            // 特征关联
            line_pairs_all = AssociateLineSingle(T_cl_last);
            LOG(INFO) << "rotation change: " << rotation_change << "  translate change: " << trans_change << endl;
            if(rotation_change < 0.1 && trans_change < 0.01)
                break; 
        }
        // 显示投影结果，用于debug
        if(visualization)
        {
            Visualize(line_pairs_all, config.calib_result_path + "/final/");
        }
    }
    else if(optimization_mode == MAPPING)
    {
        LOG(INFO) << "start camera-LiDAR mapping";
        CameraCenterPCD(config.joint_result_path + "/camera_center_init.pcd", GetCameraTranslation());
        CameraCenterPCD(config.joint_result_path + "/lidar_center_init.pcd", GetLidarTranslation());
        CameraPoseVisualize(config.joint_result_path + "/camera_pose_init.ply", GetCameraRotation(), GetCameraTranslation());
        CameraPoseVisualize(config.joint_result_path + "/lidar_pose_init.ply", GetLidarRotation(), GetLidarTranslation());

        if(false)
        {
            // 对图像特征点重新进行三角化
            vector<MatchPair> image_pairs;
            ReadMatchPair(config.match_pair_joint_path, image_pairs, min(config.num_threads, 4));
            EstimateStructure(image_pairs);
        }
        else 
            ReadPointTracks(config.sfm_result_path + "points.bin", structure);
        

        double last_cost = 0, curr_cost = 0;
        int last_step = INT32_MAX, curr_step = INT32_MAX; 
        eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all = 
                    AssociateLineMulti(config.neighbor_size_joint, true, false);
        for(int iter = 0; iter < config.num_iteration_joint; iter++)
        {
            LOG(INFO) << "iteration: " << iter;
            Optimize(line_pairs_all, structure, true, true, true, true, true, curr_cost, curr_step);
            CameraCenterPCD(config.joint_result_path + "/camera_center_" + num2str(iter) + ".pcd", GetCameraTranslation());
            CameraCenterPCD(config.joint_result_path + "/lidar_center_" + num2str(iter) + ".pcd", GetLidarTranslation());
            CameraPoseVisualize(config.joint_result_path + "/lidar_pose_" + num2str(iter) + ".ply", GetLidarRotation(), GetLidarTranslation());
            CameraPoseVisualize(config.joint_result_path + "/camera_pose_" + num2str(iter) + ".ply", GetCameraRotation(), GetCameraTranslation());
            line_pairs_all.clear();
            line_pairs_all =  AssociateLineMulti(config.neighbor_size_joint, true, false);   
            if(abs(curr_cost - last_cost) / last_cost < 0.01)
            {
                LOG(INFO) << "early termination condition fulfilled. cost change is less than 1%";
                break;
            }
            if(curr_step < 5 && last_step < 5)
            {
                LOG(INFO) << "early termination condition fulfilled. iteration step is less than 5 steps in last two iterations";
                break;
            }
            last_cost = curr_cost;
            last_step = curr_step;
    
        }
        // 输出所有匹配的直线，用于debug
        if(visualization)
            Visualize(line_pairs_all, config.joint_result_path + "/visualization/");

    }
    else 
    {
        LOG(ERROR) << "mode not supported";
        return false;
    }
    LOG(INFO) << "============== Camera-LiDAR joint optimization end ======================";
    return true;
}

// {image id, lidar id} => {line pair}
eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> CameraLidarOptimizer::AssociateLineSingle(Eigen::Matrix4d T_cl)
{
    omp_set_num_threads(config.num_threads);
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all;
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < frames.size(); i++)
    {
        CameraLidarLineAssociate associate(frames[i].GetImageRows(), frames[i].GetImageCols(), frames[i].GetImageGray());
        if(!lidars[i].edge_segmented.empty())
            associate.AssociateByAngle(image_lines_all[i].GetLines(), lidars[i].edge_segmented, lidars[i].segment_coeffs, 
                                lidars[i].cornerLessSharp, lidars[i].point_to_segment, lidars[i].end_points, T_cl);
            // associate.Associate(image_lines_all[i], lidars[i].edge_segmented, T_cl);
        else 
            associate.Associate(image_lines_all[i].GetLines(), lidars[i].cornerLessSharp, T_cl);
        #pragma omp critical 
        {
            line_pairs_all[pair<size_t, size_t>(i,i)] = associate.GetAssociatedPairs();;
        }
    }
    
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs";

    return line_pairs_all;
}

// {image id, lidar id} => {line pair}
eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> CameraLidarOptimizer::AssociateLineMulti(
        const int neighbor_size, const bool temporal, const bool use_lidar_track, const bool use_image_track)
{
    omp_set_num_threads(config.num_threads);
    LOG(INFO) << "Associate lines, neighbor size = " << neighbor_size;
    // 为每个frame找到用于匹配的近邻的LiDAR
    vector<vector<int>> each_frame_neighbor = NeighborEachFrame(neighbor_size, temporal);
    vector<vector<bool>> image_mask_all(frames.size()), lidar_mask_all(lidars.size());
    if(use_lidar_track)
        lidar_mask_all = LidarMaskByTrack();
    if(use_image_track)
        image_mask_all = ImageMaskByTrack();
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs_all;
    
    #pragma omp parallel for schedule(dynamic)
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
                // associate.Associate(image_lines_all[frame_id], lidar.edge_segmented, T_cl);
                associate.AssociateByAngle(image_lines_all[frame_id].GetLines(), lidar.edge_segmented, lidar.segment_coeffs,
                             lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points, T_cl, true, 
                             image_mask_all[frame_id], lidar_mask_all[lidar_id]);
                            
            else 
                associate.Associate(image_lines_all[frame_id].GetLines(), lidar.cornerLessSharp, T_cl);
            vector<CameraLidarLinePair> pairs = associate.GetAssociatedPairs();
            // 设置直线权重
            // SetLineWeight(pairs, frame_id, lidar_id_start);
            #pragma omp critical
            {
                line_pairs_all[pair<size_t,size_t>(frame_id, lidar_id)] = pairs;
            }
        
        }
    }
    size_t num_line_pairs = 0;
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs_all.begin();
        it != line_pairs_all.end(); it++)
        num_line_pairs += it->second.size();
    LOG(INFO) << "Associate " << num_line_pairs << " line pairs";
    return line_pairs_all;
}


int CameraLidarOptimizer::Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, 
                std::vector<PointTrack>& structure, const bool refine_camera_rotation,
                 const bool refine_camera_trans, const bool refine_lidar_rotation, 
                const bool refine_lidar_trans, const bool refine_structure,
                double& cost, int& steps)
{
    // 1. 取出所有的相机位姿和雷达位姿并单独保存下来进行优化，防止影响到原本的位姿
    // 而且取出了雷达的中心用于之后的lidar-lidar约束
    eigen_vector<Eigen::Vector3d> angleAxis_cw_list(frames.size()), angleAxis_lw_list(lidars.size());
    eigen_vector<Eigen::Vector3d> t_cw_list(frames.size()), t_lw_list(lidars.size());
    pcl::PointCloud<pcl::PointXYZI> lidar_center;
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_cw = frames[i].GetPose().inverse();
        Eigen::Matrix3d R_cw = T_cw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_cw.data(), angleAxis_cw_list[i].data());
        t_cw_list[i] = T_cw.block<3,1>(0,3);
    }
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        Eigen::Matrix4d T_wl = lidars[i].GetPose();
        Eigen::Matrix4d T_lw = T_wl.inverse();
        Eigen::Matrix3d R_lw = T_lw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_lw.data(), angleAxis_lw_list[i].data());
        t_lw_list[i] = T_lw.block<3,1>(0,3);
        lidars[i].Transform2LidarWorld();
    }
    
    ceres::Problem problem;
    ceres::LossFunction*  loss_function1 = new ceres::HuberLoss(3 * M_PI / 180.0);     // 夹角为2度
    // ceres::LossFunction *  loss_function = new ceres::HuberLoss(1);         // 点到直线距离为1米

    LOG(INFO) << "Add residuals to problem";
    // 2. 遍历所有的直线匹配关系，进行lidar-camera之间的匹配
    size_t residual_camera_lidar = AddCameraLidarResidual(frames, lidars,
                    angleAxis_cw_list, t_cw_list, angleAxis_lw_list, t_lw_list, 
                    line_pairs, loss_function1, problem, config.camera_lidar_weight);
    LOG(INFO) << "num residual blocks for camera-lidar : " << residual_camera_lidar;

    // 3. 根据三角化的点进行重投影误差的约束，这是camera-camera的约束
    size_t residual_camera = AddCameraResidual(frames, angleAxis_cw_list, t_cw_list, structure, 
                        problem, RESIDUAL_TYPE::ANGLE_RESIDUAL_1, config.camera_weight);
    LOG(INFO) << "num residual blocks for camera-camera: " << residual_camera;

    // 4.根据雷达之间的匹配关系进行约束，这是lidar-lidar的约束
    vector<vector<int>> neighbors = FindNeighbors(lidars, 6);
    size_t residual_lidar = 0;
    if(config.point_to_line_residual)
        residual_lidar += AddLidarPointToLineResidual(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance, config.lidar_weight);
    if(config.line_to_line_residual)
    {
    #if 0 
        residual_lidar += AddLidarLineToLineResidual(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance, config.lidar_weight);
    #endif
        LidarLineMatch matcher(lidars);
        matcher.SetNeighborSize(4);
        matcher.SetMinTrackLength(3);
        matcher.GenerateTracks();
        residual_lidar += AddLidarLineToLineResidual2(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, matcher.GetTracks(),
                                    config.point_to_line_dis_threshold, config.angle_residual, config.normalize_distance);

    }
    if(config.point_to_plane_residual)
        residual_lidar += AddLidarPointToPlaneResidual(neighbors, lidars, angleAxis_lw_list, t_lw_list, problem, 
                                config.point_to_plane_dis_threshold, config.lidar_plane_tolerance, config.angle_residual, 
                                config.normalize_distance, config.lidar_weight);
    LOG(INFO) << "num residual blocks for lidar-lidar: " << residual_lidar;

    // 固定三维空间点位置
    if(refine_structure == false)
    {
        for( PointTrack& track : structure)
            problem.SetParameterBlockConstant(track.point_3d.data());
    }
    // 固定相机位姿
    for(size_t i = 0 ; i < frames.size(); i++)
    {
        if(frames[i].IsPoseValid())
        {
            if(refine_camera_rotation == false)
                problem.SetParameterBlockConstant(angleAxis_cw_list[i].data());
            if(refine_camera_trans == false)
                problem.SetParameterBlockConstant(t_cw_list[i].data());
        }
    }
    // 固定LiDAR位姿
    for(size_t i = 0 ; i < lidars.size(); i++)
    {
        if(lidars[i].IsPoseValid() && lidars[i].valid)
        {
            if(refine_lidar_rotation == false)
                problem.SetParameterBlockConstant(angleAxis_lw_list[i].data());
            if(refine_lidar_trans == false)
                problem.SetParameterBlockConstant(t_lw_list[i].data());
        }
    }    

    problem.SetParameterBlockConstant(angleAxis_cw_list[0].data());
    problem.SetParameterBlockConstant(t_cw_list[0].data());
    // problem.SetParameterBlockConstant(angleAxis_lw_list[0].data());
    // problem.SetParameterBlockConstant(t_lw_list[0].data());

    LOG(INFO) << "total residual blocks : " << problem.NumResidualBlocks();
    ceres::Solver::Options options = SetOptionsSfM(config.num_threads);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(!summary.IsSolutionUsable())
    {
        LOG(INFO) << summary.FullReport();
        ofstream f(config.joint_result_path + "/camera_pose_fail.txt");
        for(size_t i = 0; i < angleAxis_cw_list.size(); i++)
            f << angleAxis_cw_list[i].x() << " " << angleAxis_cw_list[i].y() << " " << angleAxis_cw_list[i].z() << " " 
              << t_cw_list[i].x() << " " << t_cw_list[i].y() << " " << t_cw_list[i].z() << endl;
        f.close();
        f.open(config.joint_result_path + + "/lidar_pose_fail.txt");
        for(size_t i = 0; i < angleAxis_lw_list.size(); i++)
            f << angleAxis_lw_list[i].x() << " " << angleAxis_lw_list[i].y() << " " << angleAxis_lw_list[i].z() << " " 
              << t_lw_list[i].x() << " " << t_lw_list[i].y() << " " << t_lw_list[i].z() << endl;
        f.close();
        LOG(ERROR) << "Camera LiDAR optimization failed";
        return false;
    }
    LOG(INFO) << summary.BriefReport();
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix3d R_cw;
        ceres::AngleAxisToRotationMatrix(angleAxis_cw_list[i].data(), R_cw.data());
        Eigen::Vector3d t_cw = t_cw_list[i];
        Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
        T_cw.block<3,3>(0,0) = R_cw;
        T_cw.block<3,1>(0,3) = t_cw;
        frames[i].SetPose(T_cw.inverse());
    }

    #pragma omp parallel for 
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid())
            continue;
        // 如果雷达在世界坐标系下，那么就一定要先变换到局部坐标系，然后再更新位姿
        if(lidars[i].IsInWorldCoordinate())
            lidars[i].Transform2Local();
        Eigen::Matrix3d R_lw;
        ceres::AngleAxisToRotationMatrix(angleAxis_lw_list[i].data(), R_lw.data());
        Eigen::Vector3d t_lw = t_lw_list[i];
        Eigen::Matrix4d T_lw = Eigen::Matrix4d::Identity();
        T_lw.block<3,3>(0,0) = R_lw;
        T_lw.block<3,1>(0,3) = t_lw;
        lidars[i].SetPose(T_lw.inverse());
    }
    cost = summary.final_cost;
    steps = summary.num_successful_steps;
    return 1;
}


std::vector<std::vector<int>> CameraLidarOptimizer::NeighborEachFrame(const int neighbor_size, const bool temporal)
{
    vector<vector<int>> each_frame_neighbor(frames.size());
    if(temporal)
    {
        for(int frame_id = 0; frame_id < frames.size(); frame_id++)
        {
            // 确定LiDAR的起始和结尾，注意结尾是不包含在近邻范围内的。这里的方法就是先确定LiDAR的起始位置，
            // 根据起始位置和neighbor size确定结尾。然后把结尾限制在lidrs.size()内，接着根据结尾反过来确定起始
            // 这样就能保证图像都有相同数量的近邻了，只有一种是例外，也就是neighbor size > lidars.size
            int lidar_id_start = max(0, frame_id - (neighbor_size / 2));
            int lidar_id_end = min(static_cast<int>(lidars.size()), lidar_id_start + neighbor_size);
            lidar_id_start = max(0, lidar_id_end - neighbor_size);
            for(int lidar_id = lidar_id_start; lidar_id < lidar_id_end; lidar_id++)
                each_frame_neighbor[frame_id].push_back(lidar_id);
        }
    }
    else 
    {
        pcl::PointCloud<PointType> lidar_center;
        for(size_t i = 0; i < lidars.size(); i++)
        {
            if(!lidars[i].IsPoseValid() || !lidars[i].valid)
                continue;
            PointType center;
            Eigen::Vector3d t_wl = lidars[i].GetPose().block<3,1>(0,3);
            center.x = t_wl.x();
            center.y = t_wl.y();
            center.z = t_wl.z();
            // 设置intensity只是为了知道当前的点对应于哪一帧雷达，因为可能有的雷达没有位姿就没被记录下来
            center.intensity = i;   
            lidar_center.push_back(center);
        }
        pcl::KdTreeFLANN<PointType>::Ptr kd_center(new pcl::KdTreeFLANN<PointType>());
        kd_center->setInputCloud(lidar_center.makeShared());
        vector<int> neighbors;
        for(int i = 0; i < frames.size(); i++)
        {
            if(!frames[i].IsPoseValid())
                continue;
            PointType center;
            Eigen::Vector3d t_wl = frames[i].GetPose().block<3,1>(0,3);
            center.x = t_wl.x();
            center.y = t_wl.y();
            center.z = t_wl.z();
            kd_center->nearestKSearch(center, neighbor_size, neighbors, *(new vector<float>()));
            for(int& n_idx : neighbors)
            {
                n_idx = lidar_center[n_idx].intensity;
            }
            set<int> neighbors_set(neighbors.begin(), neighbors.end());
            if(neighbors_set.count(i - 1) == 0 && (i - 1 >= 0))
                neighbors.push_back(i - 1);
            if(neighbors_set.count(i + 1) == 0 && (i + 1 < lidars.size()))
                neighbors.push_back(i + 1);
            each_frame_neighbor[i] = neighbors;
        }
    }
    return each_frame_neighbor;
}

std::vector<std::vector<bool>> CameraLidarOptimizer::LidarMaskByTrack(const int min_track_length, const int neighbor_size)
{
    #pragma omp parallel for
    for(Velodyne& lidar : lidars)
    {
        if(!lidar.IsInWorldCoordinate())
            lidar.Transform2LidarWorld();
    }
    LidarLineMatch lidar_line_matcher(lidars);
    lidar_line_matcher.SetMinTrackLength(min_track_length);
    lidar_line_matcher.SetNeighborSize(neighbor_size);
    lidar_line_matcher.GenerateTracks();

    // 默认所有的LiDAR直线都是被掩模的状态，也就是对应位置为false
    vector<vector<bool>> lidar_mask_all;
    for(size_t i = 0; i < lidars.size(); i++)
        lidar_mask_all.push_back(vector<bool>(lidars[i].edge_segmented.size(), false));
    for(const LineTrack& track : lidar_line_matcher.GetTracks())
    {
        // feature是 {lidar id, line id}
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            lidar_mask_all[feature.first][feature.second] = true;
    }
    #pragma omp parallel for
    for(Velodyne& lidar : lidars)
    {
        if(lidar.IsInWorldCoordinate())
            lidar.Transform2Local();
    }
    return lidar_mask_all;
}

std::vector<std::vector<bool>> CameraLidarOptimizer::ImageMaskByTrack(const int min_track_length, const int neighbor_size)
{
    eigen_vector<Eigen::Matrix3d> R_wc_list;
    eigen_vector<Eigen::Vector3d> t_wc_list;
    for(const Frame& frame : frames)
    {
        R_wc_list.push_back(frame.GetPose().block<3,3>(0,0));
        t_wc_list.push_back(frame.GetPose().block<3,1>(0,3));
    }
    PanoramaLineMatcher image_line_matcher(image_lines_all, R_wc_list, t_wc_list);
    image_line_matcher.SetMinTrackLength(min_track_length);
    image_line_matcher.SetNeighborSize(neighbor_size);
    image_line_matcher.GenerateTracks(BASIC);
    // 除去包含平行直线的track
    image_line_matcher.RemoveParallelLines();
    vector<LineTrack> image_tracks = image_line_matcher.GetTracks();
    for(size_t i = 0; i < image_tracks.size(); i++)
        image_tracks[i].id = i;

    vector<vector<bool>> image_mask_all;
    for(size_t i = 0; i < image_lines_all.size(); i++)
        image_mask_all.push_back(vector<bool>(image_lines_all[i].GetLines().size(), false));
    for(const LineTrack& track : image_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            image_mask_all[feature.first][feature.second] = true;
    }
    return image_mask_all;
}

bool CameraLidarOptimizer::SetLineWeight(vector<CameraLidarLinePair>& line_pair, const size_t frame_idx, const size_t lidar_idx)
{
    
    if(!frames[frame_idx].IsPoseValid())
        return false;
    // 使用当期帧和之后的帧的相对运动作为当前的前进方向
    size_t next_valid_id = frame_idx + 1;
    while(next_valid_id < frames.size() && !frames[next_valid_id].IsPoseValid())
        next_valid_id++;
    if(next_valid_id == frames.size())
        return false;
    Eigen::Matrix4d T_12 = frames[frame_idx].GetPose().inverse() * frames[next_valid_id].GetPose();
    Eigen::Vector3d moving_direction = T_12.block<3,1>(0,3);
    const vector<cv::Vec4f>& lines = image_lines_all[frame_idx].GetLines();
    vector<double> weight;
    Equirectangular eq(frames[frame_idx].GetImageRows(), frames[frame_idx].GetImageCols());
    // 把直线变成单位球面上的点
    for(size_t i = 0; i < line_pair.size(); i++)
    {
        const cv::Vec4f& l = line_pair[i].image_line;
        cv::Point3f start = eq.ImageToCam(cv::Point2f(l[0], l[1]), 1.f);
        cv::Point3f end = eq.ImageToCam(cv::Point2f(l[2],l[3]), 1.f);
        Eigen::Vector3d line_3d(end.x - start.x, end.y - start.y, end.z - start.z);
        // 计算直线和前进方向的夹角，这里是借用了平面夹角的计算，因为两者是一样的，而且都要返回锐角
        double angle = PlaneAngle(moving_direction.data(), line_3d.data());
        // 权重用e的负指数计算，也就是 e^(-angle), angle=0时权重最大为1， angle越大，权重越小
        weight.push_back(exp(-angle));
        line_pair[i].weight = exp(-angle);
    }
    // 可视化各个直线的权重
    // {
    //     cv::Mat img_gray = frames[frame_idx].GetImageGray();
    //     cv::cvtColor(img_gray, img_gray, CV_GRAY2BGR);
    //     double min_weight = exp(- M_PI_2);
    //     double weight_length = 1.0 - min_weight;
    //     for(size_t i = 0; i < lines.size(); i++)
    //     {
    //         uchar relative_weight = static_cast<uchar>((weight[i] - min_weight) / weight_length * 255.0);
    //         cv::Vec3b color = Gray2Color(relative_weight);
    //         DrawLine(img_gray, lines[i], color, 5, true);
    //     }
    //     cv::imwrite("./weighted_lines_" + num2str(int(frames[frame_idx].id)) + ".jpg", img_gray);
    // }
    return true;
}

bool CameraLidarOptimizer::EstimateStructure(const std::vector<MatchPair>& image_pairs)
{
    LOG(INFO) << "==================== Estimate Initial Structure start =================";
    structure = TriangulateTracks(frames, image_pairs);
    if(!FilterTracksToFar(frames, structure, 8))
        return false;
    LOG(INFO) << "Successfully triangulate " << structure.size() << " tracks";
    LOG(INFO) << "==================== Estimate Initial Structure end =================";
    return true;
}


bool CameraLidarOptimizer::GlobalBundleAdjustment(std::vector<PointTrack>& structure ,bool refine_structure, bool refine_rotation, bool refine_translation)
{
    if(!SfMGlobalBA(frames, structure, RESIDUAL_TYPE::ANGLE_RESIDUAL_1, 
                    config.num_threads, refine_structure, refine_rotation, refine_translation))
    {
        LOG(ERROR) << "Global BA failed";
        return false;
    }
    return true;
}

void CameraLidarOptimizer::Visualize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs_all,
                    const std::string path, int line_width, int point_size)
{
    if(!boost::filesystem::exists(path))
        boost::filesystem::create_directories(path);
    LOG(INFO) << "save joint visualization result in " << path;
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

            cv::Mat img_gray = frames[image_idx].GetImageGray();
            cv::Mat img_line = DrawLinePairsOnImage(img_gray, it->second, T_cl, line_width);                
            cv::imwrite(path + "/line_pair_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_line);
            
            cv::Mat img_cloud = ProjectLidar2PanoramaRGB(lidars[lidar_idx].cloud, img_gray,
                        T_cl, config.min_depth, config.max_depth_visual, point_size);
            cv::imwrite(path + "/cloud_project_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_cloud);
        
            cv::Mat img_corner = ProjectLidar2PanoramaRGB(lidars[lidar_idx].cornerLessSharp, img_gray,
                        T_cl, config.min_depth, config.max_depth_visual, point_size);
            cv::imwrite(path + "/corner_project_" + num2str(image_idx) + "_" + num2str(lidar_idx) + ".jpg", img_corner);
        }
    }
}

pcl::PointCloud<PointType> CameraLidarOptimizer::FuseLidar(int skip, double min_range, double max_range)
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

Eigen::Matrix4d CameraLidarOptimizer::GetResult()
{
    return T_cl_optimized;
}

void CameraLidarOptimizer::SetOptimizationMode(int mode)
{
    optimization_mode = mode;
}

eigen_vector<Eigen::Matrix3d> CameraLidarOptimizer::GetCameraRotation(bool with_invalid)
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

eigen_vector<Eigen::Vector3d> CameraLidarOptimizer::GetCameraTranslation(bool with_invalid)
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

std::vector<std::string> CameraLidarOptimizer::GetImageNames(bool with_invalid)
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

eigen_vector<Eigen::Matrix3d> CameraLidarOptimizer::GetLidarRotation(bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> global_rotation;
    for(const Velodyne& l : lidars)
    {
        // 如果位姿不可用，同时也不需要输出不可用位姿，那么就直接跳过
        if((!l.IsPoseValid() || !l.valid) && with_invalid == false)
            continue;
        Eigen::Matrix4d T_wc = l.GetPose();
        global_rotation.push_back(T_wc.block<3,3>(0,0));
    }
    return global_rotation;
}

eigen_vector<Eigen::Vector3d> CameraLidarOptimizer::GetLidarTranslation(bool with_invalid)
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

std::vector<std::string> CameraLidarOptimizer::GetLidarNames(bool with_invalid)
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

const std::vector<Frame>& CameraLidarOptimizer::GetFrames()
{
    return frames;
}

const std::vector<Velodyne>& CameraLidarOptimizer::GetLidars()
{
    return lidars;
}

void CameraLidarOptimizer::SetFrames(const std::vector<Frame>& _frames)
{
    frames = _frames;
}

void CameraLidarOptimizer::SetLidars(const std::vector<Velodyne>& _lidars)
{
    lidars = _lidars;
}

bool CameraLidarOptimizer::ExportStructureBinary(const std::string file_name)
{
    if(structure.empty())
        return false;
    return ExportPointTracks(file_name, structure);
}