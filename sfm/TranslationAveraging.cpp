/*
 * @Author: Diantao Tu
 * @Date: 2022-08-26 10:33:32
 */

#include "TranslationAveraging.h"

bool InitGlobalTranslationGPS(const std::vector<Frame>& frames, eigen_vector<Eigen::Vector3d>& global_translations,
                            const map<size_t, size_t>& new_to_old)
{
    Eigen::Matrix3d R;
    R << -0.296045,   0.954821,  0.0259601,
        0.00196404,  0.0277868,  -0.999612,    
        -0.955172,  -0.295879, -0.0101015;    

    assert(global_translations.size() == new_to_old.size());
    // GPS 位置是相机光心位置，也就是 t_wc, global_translation 要保存的是t_cw
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        size_t old_id = new_to_old.find(i)->second;
        Eigen::Vector3d t_wc = frames[old_id].GetGPS();
        t_wc = R * t_wc;
        if(isinf(t_wc.x()) || isinf(t_wc.y()) || isinf(t_wc.z()))
            continue;
        Eigen::Matrix3d R_wc = frames[old_id].GetPose().block<3,3>(0,0);
        // t_cw = -R_cw * t_wc
        global_translations[i] = - R_wc.transpose() * t_wc;
    }
    return true;
}
bool TranslationAveragingDLT(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations)
{
    size_t num_cameras = global_translations.size();
    Eigen::SparseMatrix<double> A(image_pairs.size() * 3, num_cameras * 3);
    Eigen::VectorXd b(image_pairs.size() * 3);
    Eigen::VectorXd x(num_cameras * 3);
    A.reserve(A.rows() * 4);
    Eigen::SparseMatrix<double>::Index i = 0, j = 0;
    // 构建矩阵Ax=b中的A和b
    for(const MatchPair& pair : image_pairs)
    {
        const Eigen::Matrix3d& R_ji = pair.R_21;
        j = pair.image_pair.first * 3;
        // A.block<3,3>(i,j) = -R_ji
        A.insert(i, j) = -R_ji(0, 0);
        A.insert(i + 1, j) = -R_ji(1, 0);
        A.insert(i + 2, j) = -R_ji(2, 0);
        A.insert(i, j + 1) = -R_ji(0, 1);
        A.insert(i + 1, j + 1) = -R_ji(1, 1);
        A.insert(i + 2, j + 1) = -R_ji(2, 1);
        A.insert(i, j + 2) = -R_ji(0, 2);
        A.insert(i + 1, j + 2) = -R_ji(1, 2);
        A.insert(i + 2, j + 2) = -R_ji(2, 2);

        j = pair.image_pair.second * 3;
        // A.block<3,3>(i,j) = identity
        A.insert(i, j) = 1;
        A.insert(i + 1, j + 1) = 1;
        A.insert(i + 2, j + 2) = 1;
        
        b.block<3,1>(i,0) = pair.t_21;
        i+=3;
    }
    A.makeCompressed();
    // 求解方程 Ax=b
    Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> linear_solver;
    linear_solver.compute(A);
    if(linear_solver.info() != Eigen::Success)
    {
        LOG(ERROR) << "Cholesky decomposition failed.";
        return false;
    }
    x = linear_solver.solve(b);
    if (linear_solver.info() != Eigen::Success) 
    {
        LOG(ERROR) << "Failed to solve the least squares system";
        return false;
    }
    for(size_t idx = 0; idx < num_cameras; idx++)
    {
        global_translations[idx] = x.block<3,1>(idx * 3, 0);
    }
    return true;
}


bool TranslationAveragingL2(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, vector<double>& scales,
                                size_t origin_idx, vector<double>& weight, double& costs ,ceres::LossFunction* loss_function,
                                const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads)
{
    size_t num_cameras = global_translations.size();
    assert(global_translations[origin_idx].isZero());
    ceres::Problem problem;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].image_pair.first;
        const size_t idx2 = image_pairs[i].image_pair.second;
        ceres::CostFunction* cost_function = PairWiseTranslationResidual::Create(image_pairs[i].R_21, image_pairs[i].t_21, weight[i]);
        ceres::ResidualBlockId residual_block_id = problem.AddResidualBlock(cost_function, loss_function, 
                    global_translations[idx1].data(), global_translations[idx2].data(), &scales[i]);
        if(scales[i] != 1)
        {
            problem.SetParameterLowerBound(&scales[i], 0, scales[i] * 0.5);
            problem.SetParameterUpperBound(&scales[i], 0, scales[i] * 3.0);

            double upper_scale = scales[i] * upper_scale_ratio;
            double lower_scale = scales[i] * lower_scale_ratio;
            // if(image_pairs[i].upper_scale > 0)
            // {
            //     upper_scale = min(upper_scale, image_pairs[i].upper_scale);
            // }
            // if(upper_scale <= lower_scale)
            // {
            //     upper_scale = scales[i] * 1.6;
            // }
            ceres::CostFunction* cost_function2 = ScaleFactor::Create(upper_scale, lower_scale);   // 适用于1410以及14楼正放
            problem.AddResidualBlock(cost_function2, nullptr, &scales[i]);
        }
        else
        {
            ceres::CostFunction* cost_function2 = ScaleFactor::Create(2, 1);
            problem.AddResidualBlock(cost_function2, nullptr, &scales[i]);
            // problem.SetParameterLowerBound(&scales[i], 0, 1e-2);

        }
    }
    problem.SetParameterBlockConstant(global_translations[origin_idx].data());

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else
    {
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    }
    options.max_num_iterations = 50;
    options.num_threads = num_threads;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(!summary.IsSolutionUsable())
    {
        LOG(INFO) << summary.FullReport();
        return false;
    }
    LOG(INFO) << summary.BriefReport();
    costs = summary.final_cost;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].image_pair.first;
        const size_t idx2 = image_pairs[i].image_pair.second;
        weight[i] = (global_translations[idx2] - image_pairs[i].R_21 * global_translations[idx1] - scales[i]*image_pairs[i].t_21).norm();
        weight[i] = pow(weight[i] + 1e-2, -0.5);
    }
    return true;
}

bool TranslationAveragingSoftL1(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations,
                                    size_t origin_idx, double l1_loss_threshold,
                                    const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads)
{
    assert(global_translations[origin_idx].isZero());
    vector<double> scales(image_pairs.size());
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        double scale = image_pairs[i].t_21.norm();
        scales[i] = scale;
        if(image_pairs[i].lower_scale < 0 || image_pairs[i].upper_scale < 0)
            scales[i] = 1;
    }
    ceres::LossFunction * loss =
            (l1_loss_threshold < 0) ? nullptr : new ceres::SoftLOneLoss(l1_loss_threshold);

    if(!TranslationAveragingL2(image_pairs, global_translations, scales, origin_idx, 
                            *(new vector<double>(image_pairs.size(), 1)), *(new double), loss, upper_scale_ratio, lower_scale_ratio, num_threads))
        return false;
    
    // 用于debug 显示scale变化
    // for(size_t i = 0; i < image_pairs.size(); i++)
    // {
    //     if(image_pairs[i].upper_scale < 0 || image_pairs[i].lower_scale < 0)
    //         continue;
    //     LOG(INFO) << "image pair " << i;
    //     double scale = image_pairs[i].t_21.norm();
    //     // LOG(INFO) << "scale before : " << scale << " scale after: " << scales[i];
    //     LOG(INFO) << "scale after / scale : " << scales[i]/scale;
    //     LOG(INFO) << "upper scale / scale : " << image_pairs[i].upper_scale / scale;
    //     LOG(INFO) << "lower scale / scale : " << image_pairs[i].lower_scale / scale;
    // }
    return true;
}

bool TranslationAveragingL2Chordal(const std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames,
                                    eigen_vector<Eigen::Vector3d>& global_translations,
                                    const map<size_t, size_t>& new_to_old, size_t origin_idx, double l2_loss_threshold,
                                    const int num_threads)
{
    assert(global_translations[origin_idx].isZero());

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(l2_loss_threshold);
    // global_translation里保存的是 t_cw,要把它变成t_wc
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        const Eigen::Matrix3d& R_wc = frames[new_to_old.find(i)->second].GetPose().block<3,3>(0,0);
        global_translations[i] = -R_wc * global_translations[i];
    }
    for(const MatchPair& pair : image_pairs)
    {
        const size_t idx1 = pair.image_pair.first;
        const size_t idx2 = pair.image_pair.second;
        const Eigen::Matrix3d R_w2 = frames[new_to_old.find(idx2)->second].GetPose().block<3,3>(0,0);
        const Eigen::Vector3d direction = R_w2 * pair.t_21.normalized();
        ceres::CostFunction* cost_function = ChrodalResidual::Create(direction);
        problem.AddResidualBlock(cost_function, loss_function, global_translations[idx1].data(), 
                            global_translations[idx2].data());
    }
    problem.SetParameterBlockConstant(global_translations[origin_idx].data());

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
    {
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    else
    {
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    }
    options.max_num_iterations = 300;
    options.num_threads = num_threads;
    options.function_tolerance = 1e-7;
    options.parameter_tolerance = 1e-8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(!summary.IsSolutionUsable())
    {
        LOG(INFO) << summary.BriefReport();
        LOG(ERROR) << "Translation averaging L2 chordal failed";
        return false;
    }
    // global_translation里现在保存的是 t_wc,要把它变回到t_cw
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        const Eigen::Matrix3d& R_cw = frames[new_to_old.find(i)->second].GetPose().block<3,3>(0,0).transpose();
        global_translations[i] = -R_cw * global_translations[i];
    }

    return true;
}


bool TranslationAveragingL1(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, size_t origin_idx,
                                const map<size_t, size_t>& new_to_old )
{
    size_t num_cameras = global_translations.size();
    assert(global_translations[origin_idx].isZero());

    // 用一个map来记录image_pair 和 它的索引之间的关系，这样就可以快速的通过匹配的图像id找到它在
    // image_pairs 里的索引
    map<pair<size_t, size_t>, size_t> image_pair_to_idx;
    vector<pair<size_t, size_t>> pairs;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        image_pair_to_idx[image_pairs[i].image_pair] = i;
        pairs.emplace_back(image_pairs[i].image_pair);
    }
    vector<Triplet> triplets = PoseGraph::FindTriplet(pairs);
    // 每个triplet有3个相对位姿，每个相对位姿提供六个约束
    const int num_constrains = triplets.size() * 3 * 6;
    // 每个相机有xyz三维坐标，每个triplet有一个尺度，还有一个gamma
    const int num_vars = num_cameras * 3 + triplets.size() + 1;
    const size_t trans_start = 0;
    const size_t lambda_start = 3 * num_cameras;
    const size_t gamma_start = lambda_start + triplets.size();

    Eigen::SparseMatrix<double, Eigen::RowMajor> A(num_constrains, num_vars);
    vector<double> C(num_constrains, 0);
    vector<int> sign(num_constrains, 0);
    vector<pair<double, double>> bounds(num_vars, 
                pair<double,double>(numeric_limits<double>::lowest(), numeric_limits<double>::max()));

#define TVAR(i,j) (trans_start + i * 3 + j)
#define LAMBDAVAR(i) (lambda_start + (size_t)(i))

    // 第一个相机的位置的上限和下限都设置为0，也就是把第一个相机固定在原点
    bounds[TVAR(origin_idx,0)].first = 0;
    bounds[TVAR(origin_idx,0)].second = 0;
    bounds[TVAR(origin_idx,1)].first = 0;
    bounds[TVAR(origin_idx,1)].second = 0;
    bounds[TVAR(origin_idx,2)].first = 0;
    bounds[TVAR(origin_idx,2)].second = 0;
    // 让尺度全都大于1，也就是设置尺度的下限为1
    for(size_t i = 0; i < triplets.size(); i++)
        bounds[LAMBDAVAR(i)].first = 1;
    // 设置gamma大于0
    bounds[bounds.size() - 1].first = 0;
    // 最小化gamma
    vector<double> costs(num_vars, 0);
    costs[costs.size() - 1] = 1;

    
    size_t row_idx = 0;
    for(size_t trip_idx = 0; trip_idx < triplets.size(); trip_idx++)
    {
        const Triplet& trip = triplets[trip_idx];
        vector<pair<size_t, size_t>> pairs_in_triplet = {{trip.i, trip.j},{trip.j, trip.k}, {trip.i, trip.k}};
        double trip_scale = 0;
        for(const pair<size_t, size_t>& p : pairs_in_triplet)
        {
            const size_t idx1 = p.first;
            const size_t idx2 = p.second;
            const Eigen::Matrix3d& R_21 = image_pairs[image_pair_to_idx[p]].R_21;
            const Eigen::Vector3d& t_21 = image_pairs[image_pair_to_idx[p]].t_21;
            // |t_jw - R_ji * t_iw - lambda_ji * t_ij| < gamma 这个约束可以变成两个
            // t_jw - R_ji * t_iw - lambda_ji * t_ij < gamma  以及 t_jw - R_ji * t_iw - lambda_ji * t_ij > -gamma
            // 每一个约束都是在xyz三维的，因此变成6个约束
            for(size_t i = 0; i < 3; i++)
            {
                // t_jw
                A.coeffRef(row_idx, TVAR(idx2, i)) = 1;
                //- R_ji t_iw
                A.coeffRef(row_idx, TVAR(idx1, 0)) = - R_21(i, 0);
                A.coeffRef(row_idx, TVAR(idx1, 1)) = - R_21(i, 1);
                A.coeffRef(row_idx, TVAR(idx1, 2)) = - R_21(i, 2);

                // - Lambda_ji t_ji
                A.coeffRef(row_idx, LAMBDAVAR(trip_idx)) = - t_21(i);

                // - gamma
                A.coeffRef(row_idx, gamma_start) = -1;

                // < 0
                sign[row_idx] = LP_LESS_OR_EQUAL;
                C[row_idx] = 0;
                row_idx++;

                // t_jw
                A.coeffRef(row_idx, TVAR(idx2, i)) = 1;
                //- R_ij T_i
                A.coeffRef(row_idx, TVAR(idx1, 0)) = - R_21(i, 0);
                A.coeffRef(row_idx, TVAR(idx1, 1)) = - R_21(i, 1);
                A.coeffRef(row_idx, TVAR(idx1, 2)) = - R_21(i, 2);

                // - Lambda_ij t_ij
                A.coeffRef(row_idx, LAMBDAVAR(trip_idx)) = - t_21(i);

                // + gamma
                A.coeffRef(row_idx, gamma_start) = 1;

                // < 0
                sign[row_idx] = LP_GREATER_OR_EQUAL;
                C[row_idx] = 0;
                row_idx++;
            }

            trip_scale += t_21.norm();
        }
        // 根据已有的图像对的尺度设置triplet的尺度下限
        trip_scale /= 3.0;
        bounds[LAMBDAVAR(trip_idx)].first = trip_scale * 0.9;
    }
#undef TVAR
#undef LAMBDAVAR

    LPConstrain constrain;
    constrain.A = A;
    constrain.bounds = bounds;
    constrain.C = C;
    constrain.costs = costs;
    constrain.num_vars = num_vars;
    constrain.sign = sign;
    constrain.minimize = true;

    LPSolver solver;
    solver.Setup(constrain);

    if(!solver.Solve())
    {
        LOG(ERROR) << "Solve linear program failed";
        return false;
    }
    vector<double> solution;
    solver.GetSolution(solution);

    // solution里保存的是t_cw
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        global_translations[i] = Eigen::Vector3d(solution[i*3], solution[i*3+1], solution[i*3+2]);
    }    
    LOG(INFO) << "Gamma is " << solution[solution.size() - 1];
    return true;
}

bool TranslationAveragingL2IRLS(std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, size_t origin_idx,
                                const int num_iteration, const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads)
{
    size_t num_cameras = global_translations.size();
    assert(global_translations[origin_idx].isZero());
    vector<double> scales(image_pairs.size());
    vector<double> weight(image_pairs.size(), 1);
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        double scale = image_pairs[i].t_21.norm();
        scales[i] = scale;
        if(image_pairs[i].lower_scale < 0 || image_pairs[i].upper_scale < 0)
            scales[i] = 1;
    }
    
    double curr_cost = 0, last_cost = 0;
    for(int iter = 0; iter < num_iteration; iter++)
    {
        LOG(INFO) << "Iteration : " << iter;
        ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.01);
        if(!TranslationAveragingL2(image_pairs, global_translations, scales, origin_idx, weight, curr_cost, loss_function,
                                    upper_scale_ratio, lower_scale_ratio, num_threads))
            return false;
        // 如果cost的变化量小于5%就认为是收敛了，或者是总的cost特别小
        if(abs(last_cost - curr_cost) / curr_cost < 0.05 || curr_cost < 10)
        {
            LOG(INFO) << "Translation averaging L2 IRLS converged in " << iter + 1 << " iterations";
            LOG(INFO) << "curr cost : " << curr_cost << ", last cost: " << last_cost;
            break;
        }
        last_cost = curr_cost;
        vector<MatchPair> good_matches;
        vector<double> good_scales, good_weight;
        for(size_t i = 0; i < scales.size(); i++)
        {
            if(scales[i] < 0)
                continue;
            good_scales.push_back(scales[i]);
            good_weight.push_back(weight[i]);
            good_matches.emplace_back(image_pairs[i]);
        }
        if(good_scales.size() != scales.size())
        {
            LOG(INFO) << scales.size() - good_scales.size() << " pairs have negative scale, drop them";
            good_scales.swap(scales);
            good_weight.swap(weight);
            good_matches.swap(image_pairs);
        }
        #if 0
        // 不要用这个，除非所有的图像都参与了平移平均，也就是说所有的图像都被包含在同一个graph里
        for(size_t i = 0; i < global_translations.size(); i++)
        {
            // global translation 算出来的是 t_cw, frame里保存的是 t_wc
            const Eigen::Vector3d& t_cw = global_translations[i];
            const Eigen::Matrix3d& R_wc = frames[i].GetPose().block<3,3>(0,0);
            Eigen::Vector3d t_wc = - R_wc * t_cw;
            frames[i].SetTranslation(t_wc);
        }
        CameraCenterPCD(config.sfm_result_path + "/camera_center_L2IRLS-" + num2str(iter) + ".pcd", GetGlobalTranslation(true));
        #endif
    }
    return true;
}

bool TranslationAveragingBATA(const std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames, 
                            eigen_vector<Eigen::Vector3d>& global_translations, 
                            const map<size_t, size_t>& new_to_old, size_t origin_idx,
                            const string& output_path)
{
    size_t num_cameras = global_translations.size();
    assert(global_translations[origin_idx].isZero());
    vector<pair<int,int>> pairs;
    eigen_vector<Eigen::Vector3d> relative_pose;
    for(const MatchPair& pair : image_pairs)
    {
        pairs.push_back(pair.image_pair);
        const Eigen::Matrix3d R_wc = frames[new_to_old.find(pair.image_pair.first)->second].GetPose().block<3,3>(0,0);
        relative_pose.push_back(R_wc * pair.t_21);
    }
    // 把输入的变成matlab版BATA的样式，这样方便对比结果
    Eigen::Matrix2Xi index(2, pairs.size());
    Eigen::Matrix3Xd observe(3, pairs.size());
    for(size_t i = 0; i < pairs.size(); i++)
    {
        index(0, i) = pairs[i].first;
        index(1, i) = pairs[i].second;
        observe.col(i) = relative_pose[i];
    }
    #if 0
    ofstream f_index("tij_index.txt");
    ofstream f_observe("tij_observe.txt");
    f_index << index << endl;
    f_observe << observe << endl;
    f_index.close();
    f_observe.close();
    #endif

    BATAConfig c;
    global_translations = BATA(pairs, relative_pose, c, output_path);
    // global_translation里现在保存的是 t_wc,要把它变回到t_cw
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        const Eigen::Matrix3d& R_cw = frames[new_to_old.find(i)->second].GetPose().block<3,3>(0,0).transpose();
        global_translations[i] = -R_cw * global_translations[i];
    }
    return true;
}

bool TranslationAveragingLUD(std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames,
                            eigen_vector<Eigen::Vector3d>& global_translations, 
                            const map<size_t, size_t>& new_to_old, size_t origin_idx,
                            const int num_iteration, const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads)
{
    size_t num_cameras =global_translations.size();
    assert(global_translations[origin_idx].isZero());
    vector<double> scales(image_pairs.size());
    vector<double> weight(image_pairs.size(), 1);
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        double scale = image_pairs[i].t_21.norm();
        scales[i] = scale;
        if(image_pairs[i].lower_scale < 0 || image_pairs[i].upper_scale < 0)
            scales[i] = 1;
    }
    // global_translation里保存的是 t_cw,要把它变成t_wc
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        const Eigen::Matrix3d& R_wc = frames[new_to_old.find(i)->second].GetPose().block<3,3>(0,0);
        global_translations[i] = -R_wc * global_translations[i];
    }
    double curr_cost = 0, last_cost = 0;
    for(int iter = 0; iter < num_iteration; iter++)
    {
        LOG(INFO) << "Iteration : " << iter;
        ceres::Problem problem;
        for(size_t i = 0; i < image_pairs.size(); i++)
        {
            const size_t idx1 = image_pairs[i].image_pair.first;
            const size_t idx2 = image_pairs[i].image_pair.second;
            const Eigen::Matrix3d& R_w2 = frames[new_to_old.find(idx2)->second].GetPose().block<3,3>(0,0);
            const Eigen::Vector3d t_12 = - image_pairs[i].R_21.transpose() * image_pairs[i].t_21;
            ceres::CostFunction* cost_function = LUDResidual::Create(R_w2, image_pairs[i].t_21, weight[i]);
            ceres::ResidualBlockId residual_block_id = problem.AddResidualBlock(cost_function, nullptr, 
                        global_translations[idx1].data(), global_translations[idx2].data(), &scales[i]);
            if(scales[i] != 1)
            {
                problem.SetParameterLowerBound(&scales[i], 0, scales[i] * 0.5);
                problem.SetParameterUpperBound(&scales[i], 0, scales[i] * 3.0);

                double upper_scale = scales[i] * upper_scale_ratio;
                double lower_scale = scales[i] * lower_scale_ratio;
               
                ceres::CostFunction* cost_function2 = ScaleFactor::Create(upper_scale, lower_scale);   // 适用于1410以及14楼正放
                problem.AddResidualBlock(cost_function2, nullptr, &scales[i]);
            }
            else
            {
                ceres::CostFunction* cost_function2 = ScaleFactor::Create(2, 1);
                problem.AddResidualBlock(cost_function2, nullptr, &scales[i]);
                problem.SetParameterLowerBound(&scales[i], 0, 1);

            }
        }
        problem.SetParameterBlockConstant(global_translations[origin_idx].data());

        ceres::Solver::Options options;
        options.minimizer_progress_to_stdout = false;
        if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE))
        {
            options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
        else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CX_SPARSE))
        {
            options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
        else if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE))
        {
            options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
        else
        {
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        }
        options.max_num_iterations = 50;
        options.num_threads = num_threads;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if(!summary.IsSolutionUsable())
        {
            LOG(INFO) << summary.FullReport();
            return false;
        }
        LOG(INFO) << summary.BriefReport();
        curr_cost = 0;
        for(size_t i = 0; i < image_pairs.size(); i++)
        {
            const size_t idx1 = image_pairs[i].image_pair.first;
            const size_t idx2 = image_pairs[i].image_pair.second;
            const Eigen::Matrix3d& R_w2 = frames[new_to_old.find(idx2)->second].GetPose().block<3,3>(0,0);
            const Eigen::Vector3d t_12 = - (image_pairs[i].R_21.transpose() * image_pairs[i].t_21).normalized();
            // weight[i] = (global_translations[idx1] - global_translations[idx2] - scales[i] * R_w1 * t_12).norm();
            weight[i] = (global_translations[idx1] - global_translations[idx2] - scales[i] * R_w2 * image_pairs[i].t_21.normalized()).norm();
            curr_cost += weight[i];
            weight[i] = pow(weight[i] + 1e-2, -0.5);
        }
        // 如果cost的变化量小于5%就认为是收敛了，或者是总的cost特别小
        if(abs(last_cost - curr_cost) / curr_cost < 0.05 || curr_cost < 10)
        {
            LOG(INFO) << "Translation averaging LUD converged in " << iter + 1 << " iterations";
            LOG(INFO) << "curr cost : " << curr_cost << ", last cost: " << last_cost;
            break;
        }
        last_cost = curr_cost;
        vector<MatchPair> good_matches;
        vector<double> good_scales, good_weight;
        for(size_t i = 0; i < scales.size(); i++)
        {
            if(scales[i] < 0)
                continue;
            good_scales.push_back(scales[i]);
            good_weight.push_back(weight[i]);
            good_matches.emplace_back(image_pairs[i]);
        }
        if(good_scales.size() != scales.size())
        {
            LOG(INFO) << scales.size() - good_scales.size() << " pairs have negative scale, drop them";
            good_scales.swap(scales);
            good_weight.swap(weight);
            good_matches.swap(image_pairs);
        }
    }
    // global_translation里现在保存的是 t_wc,要把它变回到t_cw
    for(size_t i = 0; i < global_translations.size(); i++)
    {
        const Eigen::Matrix3d& R_cw = frames[new_to_old.find(i)->second].GetPose().block<3,3>(0,0).transpose();
        global_translations[i] = -R_cw * global_translations[i];
    }
    return true;
}

