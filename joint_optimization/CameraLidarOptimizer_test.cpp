/*
 * @Author: Diantao Tu
 * @Date: 2023-11-28 19:23:53
 */

#include "CameraLidarOptimizer.h"

void CameraLidarOptimizer::TestLineAssociate(const vector<pair<int,int>>& image_lidar_pairs)
{
    for(const pair<int,int>& pair : image_lidar_pairs)
    {
        const int frame_id = pair.first;
        const int lidar_id = pair.second;
        const Frame& frame = frames[frame_id];
        const Velodyne& lidar = lidars[lidar_id];
        // T_cl = T_cw * T_wl
        Eigen::Matrix4d T_cl = frame.GetPose().inverse() * lidar.GetPose();
        // 先输出LiDAR直线，图像直线到文件，用来可视化一下
        cv::Mat img_gray = frame.GetImageGray();
        for(int i = 0; i < lidar.edge_segmented.size(); i++)
        {
            const pcl::PointCloud<PointType>& seg = lidar.edge_segmented[i];
            cv::imwrite(config.joint_result_path + "/lidar_segment_" + num2str(i) + ".jpg",
                    ProjectLidar2PanoramaGray(seg, img_gray, T_cl, 6));
        }
        // 输出图像直线
        DrawEachLine(config.joint_result_path, img_gray, image_lines_all[frame_id].GetLines(), cv::Scalar(0,0,255), 10, true);
        
        CameraLidarLineAssociate associate(frame.GetImageRows(), frame.GetImageCols(), img_gray);
        if(!lidar.edge_segmented.empty())
            // associate.Associate(image_lines_all[frame_id].GetLines(), lidar.edge_segmented, T_cl);
            associate.AssociateByAngle(image_lines_all[frame_id].GetLines(), lidar.edge_segmented, lidar.segment_coeffs,
                            lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points ,T_cl, true);
        else 
            associate.Associate(image_lines_all[frame_id].GetLines(), lidar.cornerLessSharp, T_cl);
        vector<CameraLidarLinePair> line_pairs = associate.GetAssociatedPairs();
        cv::Mat img_line = DrawLinePairsOnImage(img_gray, line_pairs, T_cl, 8);                
        cv::imwrite(config.joint_result_path + "/line_pair_" + num2str(frame_id) + "_" + num2str(lidar_id) + ".jpg", img_line);
        
        cv::Mat img_cloud = ProjectLidar2PanoramaRGB(lidar.cloud_scan, img_gray,
                    T_cl, 0.5, 10, 5);
        cv::imwrite(config.joint_result_path + "/cloud_project_" + num2str(frame_id) + "_" + num2str(lidar_id) + ".jpg", img_cloud);
    
        cv::Mat img_corner = ProjectLidar2PanoramaRGB(lidar.cornerLessSharp, img_gray,
                    T_cl, 0.5, 10, 5);
        cv::imwrite(config.joint_result_path + "/corner_project_" + num2str(frame_id) + "_" + num2str(lidar_id) + ".jpg", img_corner);
    
        LOG(INFO) << "=============================================";
        LOG(INFO) << "image id :" << frame_id << ", lidar id: " << lidar_id;
        Equirectangular eq(frame.GetImageRows(), frame.GetImageCols());
        Eigen::Vector3d angleAxis_cw, angleAxis_lw, t_cw, t_lw;
        Eigen::Matrix3d R_cw = frame.GetPose().block<3,3>(0,0).transpose();
        Eigen::Matrix3d R_lw = lidar.GetPose().block<3,3>(0,0).transpose();
        ceres::RotationMatrixToAngleAxis(R_cw.data(), angleAxis_cw.data());
        ceres::RotationMatrixToAngleAxis(R_lw.data(), angleAxis_lw.data());
        t_cw = frame.GetPose().inverse().block<3,1>(0,3);
        t_lw = lidar.GetPose().inverse().block<3,1>(0,3);
        for(const CameraLidarLinePair& lp : line_pairs)
        {
            // 雷达变换到相机坐标系
            
            Eigen::Vector3d lidar_start = (T_cl * lp.lidar_line_start.homogeneous()).hnormalized();
            Eigen::Vector3d lidar_end = (T_cl * lp.lidar_line_end.homogeneous()).hnormalized();
            Eigen::Vector3d lidar_middle = (lidar_start + lidar_end) / 2.0;
            // 图像直线变换到相机坐标系，并且计算平面
            Eigen::Vector3d p1 = eq.ImageToCam(Eigen::Vector2d(lp.image_line[0], lp.image_line[1]));
            Eigen::Vector3d p2 = eq.ImageToCam(Eigen::Vector2d(lp.image_line[2], lp.image_line[3]));
            Eigen::Vector3d p3 = (p1 + p2) / 2.0;
            Eigen::Vector4d plane = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));
            plane.normalize();
            // 图像直线对应的圆心角
            double image_line_angle = VectorAngle3D(p1.data(), p2.data()) * 180.0 / M_PI;
            // 计算图像直线中点和雷达直线中点在平面上的投影之间的夹角
            Eigen::Vector3d lidar_middle_projected;
            ProjectPointToPlane(lidar_middle.data(), plane.data(), lidar_middle_projected.data());
            double diff_angle = VectorAngle3D(lidar_middle_projected.data(), p3.data()) * 180.0 / M_PI;

            // 用cost fuanction 同样计算这个夹角
            double residual = -1;
            PlaneIOUResidual res(plane, (lp.lidar_line_start + lp.lidar_line_end) / 2.0, p1, p2);
            res(angleAxis_cw.data(), t_cw.data(), angleAxis_lw.data(), t_lw.data(), &residual);
            LOG(INFO) << "image line id: " << lp.image_line_id << ", lidar line id: " << lp.lidar_line_id << 
                    ", image line angle: " << image_line_angle / 2.f << ", diff angle: " << diff_angle << 
                    ", residual: " << residual * 180.0 / M_PI;             
        }
        LOG(INFO) << "=============================================";
    }
}

void CameraLidarOptimizer::TestTrackAssociate()
{
    eigen_vector<Eigen::Matrix3d> R_wc_list, R_wc_list_test;
    eigen_vector<Eigen::Vector3d> t_wc_list, t_wc_list_test;
    vector<string> name_list;
    cv::Mat mask = cv::imread(config.mask_path, CV_LOAD_IMAGE_GRAYSCALE);
    ReadPoseT(config.sfm_result_path + "/camera_pose_final.txt", true, R_wc_list, t_wc_list, name_list);
    // size_t id = 0;
    for(size_t i = 0; i < frames.size(); i++)
    {   
        R_wc_list_test.push_back(R_wc_list[frames[i].id]);
        t_wc_list_test.push_back(t_wc_list[frames[i].id]);
    }
    PanoramaLineMatcher image_line_matcher(image_lines_all, R_wc_list, t_wc_list);
    image_line_matcher.SetMinTrackLength(4);
    image_line_matcher.SetNeighborSize(4);
    image_line_matcher.GenerateTracks(BASIC);
    vector<LineTrack> image_tracks = image_line_matcher.GetTracks();
    // 输出初始的track
    if(!boost::filesystem::exists(config.joint_result_path + "/image_track_init"))
        boost::filesystem::create_directories(config.joint_result_path + "/image_track_init");
    #pragma omp parallel for
    for(size_t i = 0; i < image_tracks.size(); i++)
        PanoramaLineMatcher::VisualizeTrack({image_tracks[i].id, image_tracks[i].feature_pairs}, image_lines_all, config.joint_result_path + "/image_track_init");
    // 除去包含平行直线的track
    image_line_matcher.RemoveParallelLines();
    image_tracks = image_line_matcher.GetTracks();
    for(size_t i = 0; i < image_tracks.size(); i++)
        image_tracks[i].id = i;
    if(!boost::filesystem::exists(config.joint_result_path + "/image_track_final"))
        boost::filesystem::create_directories(config.joint_result_path + "/image_track_final");
    #pragma omp parallel for
    for(size_t i = 0; i < image_tracks.size(); i++)
        PanoramaLineMatcher::VisualizeTrack({image_tracks[i].id, image_tracks[i].feature_pairs}, image_lines_all, config.joint_result_path + "/image_track_final");

    for(Velodyne& lidar : lidars)
        lidar.Transform2LidarWorld();

    LidarLineMatch lidar_line_matcher(lidars);
    lidar_line_matcher.SetMinTrackLength(4);
    lidar_line_matcher.SetNeighborSize(4);
    lidar_line_matcher.GenerateTracks();
    vector<LineTrack> lidar_tracks = lidar_line_matcher.GetTracks();
    for(size_t i = 0; i < lidar_tracks.size(); i++)
        lidar_tracks[i].id = i;

    vector<vector<int>> neighbor_each_frame(frames.size());
    for(int i = 0; i < frames.size(); i++)
        neighbor_each_frame[i].push_back(i);

    for(Velodyne& lidar : lidars)
        lidar.Transform2Local();
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> line_pairs = AssociateTrack(
                image_tracks, lidar_tracks, image_lines_all, frames, lidars, neighbor_each_frame, T_cl_init
    );
    Visualize(line_pairs, config.joint_result_path);
}

void CameraLidarOptimizer::TestRandomAssociate()
{
    for(Velodyne& lidar : lidars)
        lidar.Transform2LidarWorld();

    LidarLineMatch lidar_line_matcher(lidars);
    lidar_line_matcher.SetMinTrackLength(3);
    lidar_line_matcher.SetNeighborSize(3);
    lidar_line_matcher.GenerateTracks();
    vector<LineTrack> lidar_tracks = lidar_line_matcher.GetTracks();
    for(size_t i = 0; i < lidar_tracks.size(); i++)
        lidar_tracks[i].id = i;

    vector<vector<int>> neighbor_each_frame(frames.size());
    for(int i = 0; i < frames.size(); i++)
        neighbor_each_frame[i].push_back(i);

    for(Velodyne& lidar : lidars)
        lidar.Transform2Local();
    // 用一个mask来表示哪些直线（图像直线、雷达直线）是被包含于track中的，哪些是没被包含的
    // 在进行特征匹配的时候，只选择被包含于track中的直线进行特征匹配
    vector<vector<bool>> image_mask_all, lidar_mask_all;
    for(size_t i = 0; i < image_lines_all.size(); i++)
        image_mask_all.push_back(vector<bool>(image_lines_all[i].GetLines().size(), false));
    for(size_t i = 0; i < lidars.size(); i++)
        lidar_mask_all.push_back(vector<bool>(lidars[i].edge_segmented.size(), false));
    for(const LineTrack& track : lidar_tracks)
    {
        for(const pair<uint32_t, uint32_t>& feature : track.feature_pairs)
            lidar_mask_all[feature.first][feature.second] = true;
    }
    for(size_t frame_id = 2; frame_id < frames.size(); frame_id++)
    {
        const Frame& frame = frames[frame_id];
        const Velodyne& lidar = lidars[frame_id];
        lidar.SaveFeatures("./");
        cv::Mat img_gray = frame.GetImageGray();
        vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                        cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                        cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                        cv::Scalar(255,0,255)};                         // 紫
        cv::imwrite("image_line_" + num2str(frame.id) + ".jpg", DrawLinesOnImage(img_gray, image_lines_all[frame_id].GetLines(), colors, 7, true));
        CameraLidarLineAssociate associate(frame.GetImageRows(), frame.GetImageCols());

        Eigen::Matrix4d T_wc = frame.GetPose();
        Eigen::Matrix4d T_wl = lidar.GetPose();
        Eigen::Matrix4d T_cl = T_wc.inverse() * T_wl;

        associate.AssociateByAngle(image_lines_all[frame_id].GetLines(), 
                lidar.edge_segmented, lidar.segment_coeffs, lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points, T_cl,
                false, vector<bool>(), lidar_mask_all[frame_id]);
        vector<CameraLidarLinePair> associations = associate.GetAssociatedPairs();
        LOG(INFO) << "iter 0 ================";
        LOG(INFO) << "line pairs : " << associations.size();
        float average_score = 0;
        for(const CameraLidarLinePair& p : associations)
            average_score += p.angle;
        LOG(INFO) << "average angle: " << average_score / associations.size();
        cv::Mat img_line = DrawLinePairsOnImage(img_gray, associations, T_cl, 7, true);                
        cv::imwrite("./line_pair_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(0) + ".jpg", img_line);
        
        cv::Mat img_cloud = ProjectLidar2PanoramaRGB(lidar.cloud, img_gray,
                    T_cl, 1.5, 20, 5);
        cv::imwrite("./cloud_project_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(0) + ".jpg", img_cloud);
    
        Eigen::Matrix4d delta_T_cl = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d Ry(Eigen::AngleAxisd(0 * M_PI / 180.0, Eigen::Vector3d(0,1,0))); 
        Eigen::Matrix3d Rx(Eigen::AngleAxisd(0 * M_PI / 180.0, Eigen::Vector3d(1,0,0)));
        Eigen::Matrix3d Rz(Eigen::AngleAxisd(0 * M_PI / 180.0, Eigen::Vector3d(0,0,1)));
        delta_T_cl.block<3,3>(0,0) = Rz * Ry * Rx;
        Eigen::Vector3d delta_t_cl(0, 0, 0);
        delta_T_cl.block<3,1>(0,3) = delta_t_cl;
        T_cl = delta_T_cl * T_cl;

        associate.AssociateByAngle(image_lines_all[frame_id].GetLines(),
                lidar.edge_segmented, lidar.segment_coeffs, lidar.cornerLessSharp, lidar.point_to_segment, lidar.end_points, T_cl,
                false, vector<bool>(), lidar_mask_all[frame_id]);
        associations = associate.GetAssociatedPairs();
        LOG(INFO) << "iter 1 ================";
        LOG(INFO) << "line pairs : " << associations.size();
        average_score = 0;
        for(const CameraLidarLinePair& p : associations)
            average_score += p.angle;
        LOG(INFO) << "average angle: " << average_score / associations.size();

        img_line = DrawLinePairsOnImage(img_gray, associations, T_cl, 7, true);                
        cv::imwrite("./line_pair_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(1) + ".jpg", img_line);
        
        img_cloud = ProjectLidar2PanoramaRGB(lidar.cloud, img_gray,
                    T_cl, 1.5, 20, 5);
        cv::imwrite("./cloud_project_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(1) +".jpg", img_cloud);
    
        associate.AssociateRandomDisturbance(image_lines_all[frame_id].GetLines(), frame, lidar, T_cl, false, 
            vector<bool>(), lidar_mask_all[frame_id]);

        return;


    }
}
