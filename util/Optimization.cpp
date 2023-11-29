/*
 * @Author: Diantao Tu
 * @Date: 2022-07-15 13:29:33
 */

#include "Optimization.h"

using namespace std;

bool SfMGlobalBA(std::vector<Frame>& frames, std::vector<PointTrack>& tracks, int residual_type,
                    int num_threads, bool refine_structure, bool refine_rotation, bool refine_translation)
{
    if(refine_structure == false && refine_rotation == false && refine_translation == false)
    {
        LOG(ERROR) << "Structure, rotation and translation all set constant, can not BA";
        return false;
    }
    eigen_vector<Eigen::Vector3d> point_world;
    eigen_vector<Eigen::Vector3d> angleAxis_cw_list(frames.size(), Eigen::Vector3d::Zero());
    eigen_vector<Eigen::Vector3d> t_cw_list(frames.size(), Eigen::Vector3d::Zero());
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        Eigen::Matrix4d T_cw = frames[i].GetPose().inverse();
        const Eigen::Vector3d t_cw = T_cw.block<3,1>(0,3);
        t_cw_list[i] = t_cw;
        const Eigen::Matrix3d R_cw = T_cw.block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_cw.data(), angleAxis_cw_list[i].data());
    }
        
    ceres::Problem problem;
    // 向带求解问题中加入残差项
    AddCameraResidual(frames, angleAxis_cw_list, t_cw_list, tracks, problem, residual_type);
    // 根据选项固定旋转、平移、空间点
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        if(!refine_rotation)
            problem.SetParameterBlockConstant(angleAxis_cw_list[i].data());
        if(!refine_translation)
            problem.SetParameterBlockConstant(t_cw_list[i].data());
    }
    if(!refine_structure)
        for(size_t i = 0; i < tracks.size(); i++)
            problem.SetParameterBlockConstant(tracks[i].point_3d.data());

    // 找到第一个可用的位姿并把它设置为固定
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;
        problem.SetParameterBlockConstant(angleAxis_cw_list[i].data());
        problem.SetParameterBlockConstant(t_cw_list[i].data());
        break;
    }

    ceres::Solver::Options options = SetOptionsSfM(num_threads);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        LOG(INFO) << summary.BriefReport();
        return false;
    }
    LOG(INFO) << summary.BriefReport();
    // 更新相机位姿
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
    return true;
}

bool SfMLocalBA(const Frame& frame1, const Frame& frame2, int residual_type, MatchPair& image_pair)
{
    assert(image_pair.inlier_idx.size() == image_pair.triangulated.size());
    // 图像1就是世界坐标系，所以旋转和平移都是0
    Eigen::Matrix3d R_1w = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_1w = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R_2w = image_pair.R_21;
    Eigen::Vector3d t_2w = image_pair.t_21;
    Eigen::Vector3d angleAxis_1w = Eigen::Vector3d::Zero();
    Eigen::Vector3d angleAxis_2w = Eigen::Vector3d::Zero();
    ceres::RotationMatrixToAngleAxis(R_1w.data(), angleAxis_1w.data());
    ceres::RotationMatrixToAngleAxis(R_2w.data(), angleAxis_2w.data());
    eigen_vector<Eigen::Vector3d> points_3d = image_pair.triangulated;

    // 得到两张图像上的特征点
    const size_t& idx1 = image_pair.image_pair.first;
    const size_t& idx2 = image_pair.image_pair.second;
    const vector<cv::KeyPoint>& keypoints1 = frame1.GetKeyPoints();
    const vector<cv::KeyPoint>& keypoints2 = frame2.GetKeyPoints();

    ceres::Problem problem;
    ceres::LossFunction* loss_function;
    if(residual_type == RESIDUAL_TYPE::PIXEL_RESIDUAL) 
        loss_function = new ceres::HuberLoss(4.0);
    else if(residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_1 || residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_2)
        loss_function = new ceres::HuberLoss(4.0 * M_PI / 180.0);
    else 
        return false;
    Equirectangular eq(frame1.GetImageRows(), frame1.GetImageCols());
    for(size_t i = 0 ; i < points_3d.size(); i++)
    {
        // 三角化的点和inlier_idx是相同顺序对应的，所以第i个三角化的点就对应于inlier_idx[i]
        const cv::DMatch& match = image_pair.matches[image_pair.inlier_idx[i]];
        const cv::Point2f& pt1 = keypoints1[match.queryIdx].pt;
        const cv::Point2f& pt2 = keypoints2[match.trainIdx].pt;
        if(residual_type == RESIDUAL_TYPE::PIXEL_RESIDUAL)
        {
            ceres::CostFunction* cost_function1 = PanoramaReprojResidual_Pixel::Create(pt1, frame1.GetImageRows(), frame1.GetImageCols());
            problem.AddResidualBlock(cost_function1, loss_function, angleAxis_1w.data(), t_1w.data(), points_3d[i].data());

            ceres::CostFunction* cost_function2 = PanoramaReprojResidual_Pixel::Create(pt2, frame2.GetImageRows(), frame2.GetImageCols());
            problem.AddResidualBlock(cost_function2, loss_function, angleAxis_2w.data(), t_2w.data(), points_3d[i].data());
        }
        else
        {
            // 计算图像1上特征点在球坐标系下的坐标
            Eigen::Vector2d point_image(pt1.x, pt1.y);
            Eigen::Vector3d point_sphere = eq.ImageToCam(point_image);
            ceres::CostFunction* cost_function = (residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_1) ? 
                                                    PanoramaReprojResidual_1Angle::Create(eq.ImageToCam(point_image)) :
                                                    PanoramaReprojResidual_2Angle::Create(eq.ImageToSphere(point_image));
            problem.AddResidualBlock(cost_function, loss_function, angleAxis_1w.data(), t_1w.data(), points_3d[i].data());

            // 计算图像2上特征点在球坐标系下的坐标
            point_image.x() = pt2.x;
            point_image.y() = pt2.y;
            ceres::CostFunction* cost_function2 = (residual_type == RESIDUAL_TYPE::ANGLE_RESIDUAL_1) ? 
                                                    PanoramaReprojResidual_1Angle::Create(eq.ImageToCam(point_image)) :
                                                    PanoramaReprojResidual_2Angle::Create(eq.ImageToSphere(point_image));
            problem.AddResidualBlock(cost_function2, loss_function, angleAxis_2w.data(), t_2w.data(), points_3d[i].data());
        }
    }
    // 把第一个位姿设置为固定值
    problem.SetParameterBlockConstant(angleAxis_1w.data());
    problem.SetParameterBlockConstant(t_1w.data());

    ceres::Solver::Options options = SetOptionsSfM(1);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
    {
        LOG(ERROR) << "Refine relative pose fail";
        return false;
    }
    ceres::AngleAxisToRotationMatrix(angleAxis_1w.data(), R_1w.data());
    ceres::AngleAxisToRotationMatrix(angleAxis_2w.data(), R_2w.data());
    image_pair.R_21 = R_2w * R_1w.transpose();
    image_pair.t_21 = t_2w - image_pair.R_21 * t_1w;
    // 经过优化后，平移就不是单位向量了，因此要把它变回单位向量
    double scale = image_pair.t_21.norm();
    image_pair.t_21 /= scale;
    // 所有点从世界坐标系变换到图像1的坐标下, 还要变换尺度
    image_pair.triangulated.clear();
    for(size_t i = 0; i < points_3d.size() ; i++)
        image_pair.triangulated.push_back((R_1w * points_3d[i] + t_1w) / scale);
    return true;
}

size_t AddCameraResidual( const std::vector<Frame>& frames, eigen_vector<Eigen::Vector3d>& angleAxis_cw_list, 
                    eigen_vector<Eigen::Vector3d>& t_cw_list, std::vector<PointTrack>& structure,
                     ceres::Problem& problem, int residual_type, double weight)          
{
    // 这里假设所有图像的尺寸都是相同的，为了提升鲁棒性，可以根据不同的frame设置不同的全景模型
    // 但是没必要，因为实际不可能出现图像尺寸不同的情况
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    ceres::LossFunction* loss_function;
    if(residual_type == ANGLE_RESIDUAL_1 || residual_type == ANGLE_RESIDUAL_2)
        loss_function = new ceres::HuberLoss(4.0 * M_PI / 180.0);
    else if(residual_type == PIXEL_RESIDUAL)
        loss_function = new ceres::HuberLoss(4.0);

    size_t num_residual = 0;
    for(size_t i = 0; i < structure.size(); i++)
    {
        PointTrack& track = structure[i];
        if(isnan(track.point_3d(0)) || isnan(track.point_3d(1)) || isnan(track.point_3d(2)) )
            LOG(ERROR) << "nan";
        for(const pair<uint32_t, uint32_t>& pair : track.feature_pairs)
        {
            const uint32_t frame_idx = pair.first;
            if(!frames[frame_idx].IsPoseValid())
                continue;

            if(residual_type == ANGLE_RESIDUAL_2)
            {
                cv::Point2f pt_sphere = eq.ImageToSphere(frames[frame_idx].GetKeyPoints()[pair.second].pt);
                ceres::CostFunction* cost_function = PanoramaReprojResidual_2Angle::Create(Eigen::Vector2d(pt_sphere.x, pt_sphere.y), weight);
                problem.AddResidualBlock(cost_function, loss_function, 
                            angleAxis_cw_list[frame_idx].data(), t_cw_list[frame_idx].data(), track.point_3d.data());
            }
            else if(residual_type == ANGLE_RESIDUAL_1)
            {
                cv::Point3f pt_sphere = eq.ImageToCam(frames[frame_idx].GetKeyPoints()[pair.second].pt);
                ceres::CostFunction* cost_function = PanoramaReprojResidual_1Angle::Create(Eigen::Vector3d(pt_sphere.x, pt_sphere.y, pt_sphere.z), weight);
                problem.AddResidualBlock(cost_function, loss_function, 
                            angleAxis_cw_list[frame_idx].data(), t_cw_list[frame_idx].data(), track.point_3d.data());
            }
            else if(residual_type == PIXEL_RESIDUAL)
            {
                ceres::CostFunction* cost_function = PanoramaReprojResidual_Pixel::Create(frames[frame_idx].GetKeyPoints()[pair.second].pt, 
                                                        frames[frame_idx].GetImageRows(), frames[frame_idx].GetImageCols(), weight);
                problem.AddResidualBlock(cost_function, loss_function, 
                            angleAxis_cw_list[frame_idx].data(), t_cw_list[frame_idx].data(), track.point_3d.data());
            }
            num_residual++;
        }
    }
    return num_residual;
}

size_t AddLidarLineToLineResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_line_dis_threshold, bool angle_residual,
                        bool normalized_distance, double weight)
{
    ceres::LossFunction* loss_function;
    if(angle_residual)
        loss_function = new ceres::HuberLoss(2 * M_PI / 180.0);
    else 
        loss_function = new ceres::HuberLoss(0.2);
    size_t num_residual = 0;

    // 找到最近邻的几个数据，并进行相互关联
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 如果雷达位姿不可用，那么就先舍弃这一帧，日后可以加入其他改进，应用舍弃的数据
        // to do: 增加一些操作，利用上没有位姿的数据
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        double* angleAxis_rw = angleAxis_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        double* t_rw = t_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        // 遍历所有的邻居LiDAR
        for(int n_idx : neighbors[i])
        {
            if(n_idx < 0 || n_idx == i || n_idx >= lidars.size())
                continue;
            if(!lidars[n_idx].IsPoseValid() || !lidars[n_idx].valid)
                continue;
            
            // 直线到直线的残差
            // if(abs(n_idx - static_cast<int>(i)) <= 1)
            {
                double* t_nw = t_lw_list[lidars[n_idx].id].data();
                double* angleAxis_nw = angleAxis_lw_list[lidars[n_idx].id].data();
                // vector<Line2Line> associations = AssociateLine2LineKNN(lidars[i], lidars[n_idx], point_to_line_dis_threshold);
                vector<Line2Line> associations = AssociateLine2Line(lidars[i], lidars[n_idx], point_to_line_dis_threshold);

                for(const Line2Line& ass : associations)
                {
                    // 以角度为单位的距离误差，共包含三项，第一项是两个直线分别和球心形成的平面的夹角
                    // 第二项是两条直线的IOU
                    // 第三项是两条直线的方向误差
                    if(angle_residual && false)
                    {
                        // 计算reference坐标系下雷达直线形成的平面的夹角
                        const Eigen::Vector3d& p1 = lidars[i].end_points[ass.ref_line_idx * 2];
                        const Eigen::Vector3d& p2 = lidars[i].end_points[ass.ref_line_idx * 2 + 1];
                        Eigen::Vector3d p3(0,0,0);
                        double a = ( (p2.y()-p1.y())*(p3.z()-p1.z())-(p2.z()-p1.z())*(p3.y()-p1.y()) );
                        double b = ( (p2.z()-p1.z())*(p3.x()-p1.x())-(p2.x()-p1.x())*(p3.z()-p1.z()) );
                        double c = ( (p2.x()-p1.x())*(p3.y()-p1.y())-(p2.y()-p1.y())*(p3.x()-p1.x()) );
                        ceres::CostFunction* cost_function = Plane2Plane_Global::Create(
                            Eigen::Vector3d(a,b,c), 
                            lidars[n_idx].end_points[ass.neighbor_line_idx * 2], 
                            lidars[n_idx].end_points[ass.neighbor_line_idx * 2 + 1], weight);
                        problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, t_rw, angleAxis_nw, t_nw);
                        
                        double ref_line_angle = VectorAngle3D(p1.data(), p2.data());
                        const Eigen::Vector3d ref_middle = (p1 + p2) / 2.0;
                        const Eigen::Vector3d neighbor_middle = 
                                (lidars[n_idx].end_points[ass.neighbor_line_idx * 2] + lidars[n_idx].end_points[ass.neighbor_line_idx * 2 + 1]) / 2.0;
                        ceres::CostFunction* cost_function2 = PlaneIOUResidual::Create(Eigen::Vector4d(a,b,c,0), 
                                neighbor_middle, ref_middle, ref_line_angle, 2 * weight);
                        problem.AddResidualBlock(cost_function2, loss_function, angleAxis_rw, t_rw, angleAxis_nw, t_nw);

                        Eigen::Vector3d direction_ref = lidars[i].segment_coeffs[ass.ref_line_idx].block<3,1>(3,0);
                        Eigen::Vector3d direction_nei = lidars[i].segment_coeffs[ass.neighbor_line_idx].block<3,1>(3,0);

                        // ceres::CostFunction* cost_function3 = Line2Line_Angle::Create(direction_ref, direction_nei);
                        // problem.AddResidualBlock(cost_function3, loss_function, angleAxis_rw, angleAxis_nw);
                        // num_residual += 3;
                    }
                    if(angle_residual)
                    {
                        for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                        {
                            Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                            ceres::CostFunction *cost_function = Point2Line_Angle::Create(
                                lidar_point, ass.line_point1, ass.line_point2, normalized_distance, weight);
                            problem.AddResidualBlock(cost_function, nullptr, angleAxis_rw, 
                                                        t_rw, angleAxis_nw, t_nw);
                            num_residual++;
                        }
                    }
                    /* 以米为单位的距离误差 */
                    else
                    {
                        for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                        {
                            Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                            ceres::CostFunction *cost_function = Point2Line_Meter::Create(
                                lidar_point, ass.line_point1, ass.line_point2, weight);
                            problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, 
                                                        t_rw, angleAxis_nw, t_nw);
                            num_residual++;
                        }
                    } 
                }
            }
        }
    }
    return num_residual;
}

size_t AddLidarLineToLineResidual2(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        const vector<LineTrack>& lidar_line_tracks, 
                        double point_to_line_dis_threshold, bool angle_residual,
                        bool normalized_distance, double weight)
{
    ceres::LossFunction* loss_function;
    if(angle_residual)
        loss_function = new ceres::HuberLoss(2 * M_PI / 180.0);
    else 
        loss_function = new ceres::HuberLoss(0.2);
    
    // 记录每个雷达的每个直线对应的track的id
    // key={lidar_id, line_id}   value={track_id}
    map<pair<uint32_t, uint32_t>, vector<uint32_t>> lines_to_track;

    for(const LineTrack& track : lidar_line_tracks)
    {
        for(const pair<uint32_t, uint32_t>& pair : track.feature_pairs)
            lines_to_track[pair].push_back(track.id);
    }
    
    size_t num_residual = 0;
    // 找到最近邻的几个数据，并进行相互关联
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 如果雷达位姿不可用，那么就先舍弃这一帧，日后可以加入其他改进，应用舍弃的数据
        // to do: 增加一些操作，利用上没有位姿的数据
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        double* angleAxis_rw = angleAxis_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        double* t_rw = t_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        // 遍历所有的邻居LiDAR
        for(int n_idx : neighbors[i])
        {
            if(n_idx < 0 || n_idx == i || n_idx >= lidars.size())
                continue;
            if(!lidars[n_idx].IsPoseValid() || !lidars[n_idx].valid)
                continue;
            
            // 直线到直线的残差
            // if(abs(n_idx - static_cast<int>(i)) <= 1)
            {
                double* t_nw = t_lw_list[lidars[n_idx].id].data();
                double* angleAxis_nw = angleAxis_lw_list[lidars[n_idx].id].data();
                // TODO: 其实这里不需要再进行线到线的匹配了，因为已经生成了雷达直线track，那么根据track就能知道当前两帧雷达间哪些直线
                // 是互相匹配的，所以应该删去这个雷达帧间的直线匹配过程
                // 但我懒得弄了，就先这样凑合着把
                // vector<Line2Line> associations = AssociateLine2LineKNN(lidars[i], lidars[n_idx], point_to_line_dis_threshold);
                vector<Line2Line> associations = AssociateLine2Line(lidars[i], lidars[n_idx], point_to_line_dis_threshold);

                for(const Line2Line& ass : associations)
                {
                    pair<uint32_t, uint32_t> ref_feature_pair(i, ass.ref_line_idx);
                    map<pair<uint32_t, uint32_t>, vector<uint32_t>>::const_iterator it = lines_to_track.find(ref_feature_pair);
                    // 当前参考雷达的直线没能形成track，那么直接跳过
                    if(it == lines_to_track.end())
                        continue;
                    bool valid = false;
                    pair<uint32_t, uint32_t> nei_feature_pair(n_idx, ass.neighbor_line_idx);
                    for(const uint32_t& track_id : it->second)
                    {
                        if(lidar_line_tracks[track_id].IsInside(nei_feature_pair))
                        {
                            valid = true;
                            break;
                        }
                    }
                    // 当前参考雷达的直线和近邻雷达的直线没能形成track，跳过
                    if(!valid)
                        continue;
                    // for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                    // {
                    //     Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                    //     ceres::CostFunction *cost_function = Point2Line_Meter::Create(
                    //         lidar_point, ass.line_point1, ass.line_point2, weight);
                    //     problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, 
                    //                                 t_rw, angleAxis_nw, t_nw);
                    //     num_residual++;
                    // }
                    if(angle_residual)
                    {
                        for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                        {
                            Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                            ceres::CostFunction *cost_function = Point2Line_Angle::Create(
                                lidar_point, ass.line_point1, ass.line_point2, normalized_distance, weight);
                            problem.AddResidualBlock(cost_function, nullptr, angleAxis_rw, 
                                                        t_rw, angleAxis_nw, t_nw);
                            num_residual++;
                        }
                    }
                    /* 以米为单位的距离误差 */
                    else
                    {
                        for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                        {
                            Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                            ceres::CostFunction *cost_function = Point2Line_Meter::Create(
                                lidar_point, ass.line_point1, ass.line_point2, weight);
                            problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, 
                                                        t_rw, angleAxis_nw, t_nw);
                            num_residual++;
                        }
                    } 
                    
                }
            }
        }
    }
    return num_residual;
}

size_t AddLidarPointToLineResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_line_dis_threshold, bool use_segment,
                        bool angle_residual, bool normalized_distance, double weight)
                        
{
    ceres::LossFunction* loss_function;
    if(angle_residual)
        loss_function = new ceres::HuberLoss(2 * M_PI / 180.0);
    else 
        loss_function = new ceres::HuberLoss(0.2);

    size_t num_residual = 0;
    // 找到最近邻的几个数据，并进行相互关联
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 如果雷达位姿不可用，那么就先舍弃这一帧，日后可以加入其他改进，应用舍弃的数据
        // to do: 增加一些操作，利用上没有位姿的数据
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        double* angleAxis_rw = angleAxis_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        double* t_rw = t_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        // 遍历所有的邻居LiDAR
        for(int n_idx : neighbors[i])
        {
            if(n_idx < 0 || n_idx == i || n_idx >= lidars.size())
                continue;
            if(!lidars[n_idx].IsPoseValid())
                continue;
            
            // 点到直线的残差
            if(abs(n_idx - static_cast<int>(i)) <= 1)
            {
                double* t_nw = t_lw_list[lidars[n_idx].id].data();
                double* angleAxis_nw = angleAxis_lw_list[lidars[n_idx].id].data();
                vector<Point2Line> associations;
                if(use_segment)
                    associations = AssociatePoint2LineSegmentKNN(lidars[i], lidars[n_idx], point_to_line_dis_threshold);
                else 
                    associations = AssociatePoint2Line(lidars[i], lidars[n_idx], point_to_line_dis_threshold);
                for(const Point2Line& ass : associations)
                {
                    if(angle_residual)
                    {
                        ceres::CostFunction* cost_function = Point2Line_Angle::Create(
                                ass.point, ass.line_point1, ass.line_point2, normalized_distance, weight);
                        problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, t_rw, angleAxis_nw, t_nw );
                    }
                    else
                    {
                        ceres::CostFunction *cost_function = Point2Line_Meter::Create(ass.point, ass.line_point1, ass.line_point2, weight);
                        problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, 
                                                        t_rw, angleAxis_nw, t_nw);
                    }  
                    num_residual++;
                }
            }
        }
    }
    return num_residual;
}

size_t AddLidarPointToPlaneResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_plane_dis_threshold, double plane_tolerance,
                        bool angle_residual,
                        bool normalized_distance, double weight)
{
    ceres::LossFunction* loss_function;
    if(angle_residual)
        loss_function = new ceres::HuberLoss(2 * M_PI / 180.0);
    else 
        loss_function = new ceres::HuberLoss(0.2);
    bool use_segment = !lidars[0].edge_segmented.empty();
    size_t num_residual = 0;
    // 找到最近邻的几个数据，并进行相互关联
    for(size_t i = 0; i < lidars.size(); i++)
    {
        // 如果雷达位姿不可用，那么就先舍弃这一帧，日后可以加入其他改进，应用舍弃的数据
        // to do: 增加一些操作，利用上没有位姿的数据
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        double* angleAxis_rw = angleAxis_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        double* t_rw = t_lw_list[lidars[i].id].data();  // 这里使用下标_rw只是为了和优化的代价函数保持一致
        // 遍历所有的邻居LiDAR
        for(int n_idx : neighbors[i])
        {
            if(n_idx < 0 || n_idx == i || n_idx >= lidars.size())
                continue;
            if(!lidars[n_idx].IsPoseValid())
                continue;
            vector<Point2Plane> associations = AssociatePoint2Plane(lidars[i], lidars[n_idx], plane_tolerance, 
                                point_to_plane_dis_threshold);
            for(const Point2Plane& ass : associations)
            {
                // 同样为了保持和代价函数相同的定义，point在哪个坐标系，哪个坐标系就是neighbor坐标系，另一个坐标系就是reference
                double* t_nw = t_lw_list[lidars[n_idx].id].data();
                double* angleAxis_nw = angleAxis_lw_list[lidars[n_idx].id].data();
                const Eigen::Vector3d& lidar_point = ass.point;
                const Eigen::Vector4d& plane = ass.plane_coeff;
                if(angle_residual)
                {
                    ceres::CostFunction *cost_function = Point2Plane_Angle::Create(
                        lidar_point, plane, normalized_distance, weight);
                    problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, t_rw, angleAxis_nw, t_nw);               
                }
                else 
                {
                    ceres::CostFunction *cost_function = Point2Plane_Meter::Create(lidar_point, plane, weight);
                    problem.AddResidualBlock(cost_function, loss_function,angleAxis_rw, 
                                                t_rw, angleAxis_nw, t_nw);
                } 
                num_residual++;
            }
        }
    }
    return num_residual;
}

size_t AddCameraLidarResidual(const std::vector<Frame>& frames, const std::vector<Velodyne>& lidars,
                    eigen_vector<Eigen::Vector3d>& angleAxis_cw_list, eigen_vector<Eigen::Vector3d>& t_cw_list,
                    eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, eigen_vector<Eigen::Vector3d>& t_lw_list,
                    const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, 
                    ceres::LossFunction* loss_function, ceres::Problem& problem, double weight)
{
    size_t num_residual = 0;
    Equirectangular eq(frames[0].GetImageRows(), frames[0].GetImageCols());
    for(eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>::const_iterator it = line_pairs.begin();
        it != line_pairs.end(); it++)
    {
        const size_t frame_id = it->first.first;
        const size_t lidar_id = it->first.second;
        // if(lidar_id > 1489 && lidar_id < 1494)
        //     continue;
        // if(lidar_id > 1193 && lidar_id < 1198)
        //     continue;
        if(!lidars[lidar_id].IsPoseValid() || !frames[frame_id].IsPoseValid())
            continue;
        for(const CameraLidarLinePair& line_pair : it->second)
        {
            cv::Vec4f l = line_pair.image_line;
            // 把起始点和终止点都变换成单位圆上的XYZ坐标
            Eigen::Vector3d p1 = eq.ImageToCam(Eigen::Vector2d(l[0], l[1]));
            Eigen::Vector3d p2 = eq.ImageToCam(Eigen::Vector2d(l[2], l[3]));
            Eigen::Vector4d plane = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));

            ceres::CostFunction* cost_function = Plane2Plane_Global::Create(
                                plane.head(3), line_pair.lidar_line_end, line_pair.lidar_line_start, line_pair.weight * weight);
            problem.AddResidualBlock(cost_function, loss_function, angleAxis_cw_list[frame_id].data(), t_cw_list[frame_id].data(),
                                    angleAxis_lw_list[lidar_id].data(), t_lw_list[lidar_id].data());

            double angle = VectorAngle3D(p1.data(), p2.data(), true);
            Eigen::Vector3d image_line_middle = (p1 + p2) / 2.0;
            ceres::CostFunction* cost_function2 = PlaneIOUResidual::Create(plane, 
                        (line_pair.lidar_line_end + line_pair.lidar_line_start) / 2.0,
                        image_line_middle, angle, 2.0 * weight);
            problem.AddResidualBlock(cost_function2, loss_function, angleAxis_cw_list[frame_id].data(), t_cw_list[frame_id].data(),
                                    angleAxis_lw_list[lidar_id].data(), t_lw_list[lidar_id].data());
            num_residual += 2;
        }
    }
    return num_residual;
}



ceres::Solver::Options SetOptionsSfM(const int num_thread)
{
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = num_thread;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.preconditioner_type = ceres::JACOBI;
    // If Sparse linear solver are available
    // Descending priority order by efficiency (SUITE_SPARSE > CX_SPARSE > EIGEN_SPARSE)
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    return options;
}

ceres::Solver::Options SetOptionsLidar(const int num_threads, const int lidar_size)
{
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.update_state_every_iteration = false;
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 2000;
    if (lidar_size <= kMaxNumImagesDirectDenseSolver)
    {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    else if (lidar_size <= kMaxNumImagesDirectSparseSolver)
    {
        options.preconditioner_type = ceres::JACOBI;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else
    {
        // Indirect sparse (preconditioned CG) solver.
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    options.num_threads = num_threads;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.max_num_iterations = 20;
    options.max_linear_solver_iterations = 100;
    return options;
}

