/*
 * @Author: Diantao Tu
 * @Date: 2022-08-24 14:39:58
 */
#include "LidarOdometry.h"

bool LidarOdometry::test(const std::vector<std::pair<int, int>>& lidar_pairs)
{
    vector<vector<int>> colors = {{255, 0, 0}, {255, 134, 52},   // 红 橙
                                {255, 230, 20}, {0, 255, 0},   // 黄 绿
                                {51, 255, 255}, {0, 0, 255},   // 蓝 蓝
                                {255, 0, 255}};                         // 紫
    set<int> ids;
    for(const pair<int,int>& p : lidar_pairs)
    {
        ids.insert(p.first);
        ids.insert(p.second);
    }

    for(const int& id : ids)
    {
        lidars[id].LoadLidar();
    }
    // UndistortLidars();
    for(const int id : ids)
    {
        lidars[id].ReOrderVLP2();
        lidars[id].ExtractFeatures(config.max_curvature, config.intersection_angle_threshold, config.extraction_method);
        lidars[id].Transform2LidarWorld();
        if(!lidars[id].valid)
            lidars[id].SetPose(Eigen::Matrix4d::Zero());
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
    ceres::Problem problem;
    for(const pair<int,int>& p : lidar_pairs)
    {
        const int& ref_idx = p.first;
        const int& n_idx = p.second;
        lidars[ref_idx].SaveFeatures(config.odo_result_path);
        lidars[n_idx].SaveFeatures(config.odo_result_path);

        Eigen::Vector3d angleAxis_rw_vec;
        Eigen::Matrix3d R_rw = lidars[ref_idx].GetPose().inverse().block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_rw.data(), angleAxis_rw_vec.data());
        Eigen::Vector3d t_rw_vec(lidars[ref_idx].GetPose().inverse().block<3,1>(0,3));
        Eigen::Vector3d angleAxis_nw_vec;
        Eigen::Matrix3d R_nw = lidars[n_idx].GetPose().inverse().block<3,3>(0,0);
        ceres::RotationMatrixToAngleAxis(R_nw.data(), angleAxis_nw_vec.data());
        Eigen::Vector3d t_nw_vec(lidars[n_idx].GetPose().inverse().block<3,1>(0,3));

        double* angleAxis_rw = angleAxis_rw_vec.data();
        double* t_rw = t_rw_vec.data();
        double* angleAxis_nw = angleAxis_nw_vec.data();
        double* t_nw = t_nw_vec.data();

        bool segmented = !lidars[ref_idx].edge_segmented.empty();
        if(config.point_to_line_residual)
        {
            vector<Point2Line> associations;
            if(segmented)
                associations = AssociatePoint2LineSegment(lidars[ref_idx], lidars[n_idx], config.point_to_line_dis_threshold, true);
            else 
                associations = AssociatePoint2Line(lidars[ref_idx], lidars[n_idx], config.point_to_line_dis_threshold, true);
            for(const Point2Line& ass : associations)
            {
                if(config.angle_residual)
                {
                    ceres::CostFunction* cost_function = Point2Line_Angle::Create(ass.point, ass.line_point1, ass.line_point2, config.normalize_distance);
                    problem.AddResidualBlock(cost_function, nullptr, angleAxis_rw, t_rw, 
                                                    angleAxis_nw, t_nw );
                }
                else
                {
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.2); 
                    ceres::CostFunction *cost_function = Point2Line_Meter::Create(ass.point, ass.line_point1, ass.line_point2);
                    problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, t_rw,
                                                     angleAxis_nw, t_nw);
                }  
            }
            LOG(INFO) << "point to line redisual : " << associations.size() << " for lidar " << ref_idx << " and " << n_idx; 
        }
        if(config.line_to_line_residual)
        {
            vector<Line2Line> associations = AssociateLine2Line(lidars[ref_idx], lidars[n_idx], config.point_to_line_dis_threshold, true);
            for(const Line2Line& ass : associations)
            {
                // 以角度为单位的距离误差，共包含三项，第一项是两个直线分别和球心形成的平面的夹角
                // 第二项是两条直线的IOU
                // 第三项是两条直线的方向误差
                if(config.angle_residual)
                {
                    // 计算reference坐标系下雷达直线形成的平面的夹角
                    const Eigen::Vector3d& p1 = lidars[ref_idx].end_points[ass.ref_line_idx * 2];
                    const Eigen::Vector3d& p2 = lidars[ref_idx].end_points[ass.ref_line_idx * 2 + 1];
                    Eigen::Vector3d p3(0,0,0);
                    double a = ( (p2.y()-p1.y())*(p3.z()-p1.z())-(p2.z()-p1.z())*(p3.y()-p1.y()) );
                    double b = ( (p2.z()-p1.z())*(p3.x()-p1.x())-(p2.x()-p1.x())*(p3.z()-p1.z()) );
                    double c = ( (p2.x()-p1.x())*(p3.y()-p1.y())-(p2.y()-p1.y())*(p3.x()-p1.x()) );
                    ceres::CostFunction* cost_function = Plane2Plane_Global::Create(
                        Eigen::Vector3d(a,b,c), 
                        lidars[n_idx].end_points[ass.neighbor_line_idx * 2], 
                        lidars[n_idx].end_points[ass.neighbor_line_idx * 2 + 1]);
                    problem.AddResidualBlock(cost_function, nullptr, angleAxis_rw, t_rw, angleAxis_nw, t_nw);
                    
                    double ref_line_angle = VectorAngle3D(p1.data(), p2.data());
                    const Eigen::Vector3d ref_middle = (p1 + p2) / 2.0;
                    const Eigen::Vector3d neighbor_middle = 
                            (lidars[n_idx].end_points[ass.neighbor_line_idx * 2] + lidars[n_idx].end_points[ass.neighbor_line_idx * 2 + 1]) / 2.0;
                    ceres::CostFunction* cost_function2 = PlaneIOUResidual::Create(Eigen::Vector4d(a,b,c,0), 
                            neighbor_middle, ref_middle, ref_line_angle, 2);
                    problem.AddResidualBlock(cost_function2, nullptr, angleAxis_rw, t_rw, angleAxis_nw, t_nw);

                    Eigen::Vector3d direction_ref = lidars[ref_idx].segment_coeffs[ass.ref_line_idx].block<3,1>(3,0);
                    Eigen::Vector3d direction_nei = lidars[n_idx].segment_coeffs[ass.neighbor_line_idx].block<3,1>(3,0);

                    ceres::CostFunction* cost_function3 = Line2Line_Angle::Create(direction_ref, direction_nei);
                    problem.AddResidualBlock(cost_function3, nullptr, angleAxis_rw, angleAxis_nw);
                }
                /* 以米为单位的距离误差 */
                else
                {
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.2);
                    for(const PointType& p : lidars[n_idx].edge_segmented[ass.neighbor_line_idx].points)
                    {
                        Eigen::Vector3d lidar_point = lidars[n_idx].World2Local(PclPonit2EigenVecd(p));
                        ceres::CostFunction *cost_function = Point2Line_Meter::Create(
                            lidar_point, ass.line_point1, ass.line_point2);
                        problem.AddResidualBlock(cost_function, loss_function, angleAxis_rw, 
                                                    t_rw, angleAxis_nw, t_nw);
                    }
                } 
            }
            LOG(INFO) << "line to line redisual : " << associations.size() << " for lidar " << ref_idx << " and " << n_idx; 
        }
        if(config.point_to_plane_residual)
        {
            vector<Point2Plane> associations = AssociatePoint2Plane(lidars[ref_idx], lidars[n_idx],
                         config.lidar_plane_tolerance, config.point_to_plane_dis_threshold, true);
            for(const Point2Plane& ass : associations)
            {
                const Eigen::Vector3d& lidar_point = ass.point;
                const Eigen::Vector4d& plane = ass.plane_coeff;
                if(config.angle_residual)
                {
                    ceres::CostFunction *cost_function = Point2Plane_Angle::Create(
                        lidar_point, plane, 1);
                    problem.AddResidualBlock(cost_function, nullptr, angleAxis_rw, t_rw, angleAxis_nw, t_nw);               
                }
                else 
                {
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.2);
                    ceres::CostFunction *cost_function = Point2Plane_Meter::Create(lidar_point, plane);
                    problem.AddResidualBlock(cost_function, loss_function,angleAxis_rw, 
                                                t_rw, angleAxis_nw, t_nw);
                } 
            }
            LOG(INFO) << "point to plane redisual : " << associations.size() << " for lidar " << ref_idx << " and " << n_idx; 
        }
    }
    
    ceres::Solver::Options solver_options;
    solver_options.minimizer_progress_to_stdout = true;
    solver_options.update_state_every_iteration = true;
    const size_t kMaxNumImagesDirectDenseSolver = 50;
    const size_t kMaxNumImagesDirectSparseSolver = 2000;
    const size_t num_lidar = lidars.size();
    if (num_lidar <= kMaxNumImagesDirectDenseSolver)
    {
        solver_options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    else if (num_lidar <= kMaxNumImagesDirectSparseSolver)
    {
        solver_options.preconditioner_type = ceres::JACOBI;
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    }
    else
    {
        // Indirect sparse (preconditioned CG) solver.
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    }

    solver_options.num_threads = 11;
    solver_options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    solver_options.max_num_iterations = 20;
    solver_options.max_linear_solver_iterations = 100;
    
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if(!summary.IsSolutionUsable())
    {
        LOG(ERROR) << "LiDAR odometry failed";
        LOG(INFO) << summary.FullReport();
        ofstream f(config.odo_result_path + "/lidar_pose_failed.txt");
        for(int i = 0; i < angleAxis_lw_list.size(); i++)
        {
            f << i << " " << angleAxis_lw_list[i].x() << " " << angleAxis_lw_list[i].y() << " " << angleAxis_lw_list[i].z() << " " 
                << t_lw_list[i].x() << " " << t_lw_list[i].y() << " " << t_lw_list[i].z() << endl;
        }
        f.close();
    }
    return false;
}