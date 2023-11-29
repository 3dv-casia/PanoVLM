/*
 * @Author: Diantao Tu
 * @Date: 2022-08-25 20:45:09
 */

#include "RotationAveraging.h"

using namespace std;


vector<MatchPair> FilterPairs(const vector<MatchPair>& image_pairs, const eigen_vector<Eigen::Matrix3d>& global_rotations, double angle_threshold)
{
    vector<double> errors;
    errors.reserve(image_pairs.size());
    for(const MatchPair& p : image_pairs)
    {
        const Eigen::Matrix3d& R_1w = global_rotations[p.image_pair.first];
        const Eigen::Matrix3d& R_2w = global_rotations[p.image_pair.second];
        const Eigen::Matrix3d error_mat = R_2w.transpose() * p.R_21 * R_1w;
        Eigen::Vector3d error_angleaxis;
        ceres::RotationMatrixToAngleAxis(error_mat.data(), error_angleaxis.data());
        errors.push_back(error_angleaxis.norm());
    }
    double threshold = 0;
    if(angle_threshold > 0)
        threshold = angle_threshold / 180.0 * M_PI;
    // given an array of values, compute the X84 threshold as in:
    // Hampel FR, Rousseeuw PJ, Ronchetti EM, Stahel WA
    // "Robust Statistics: the Approach Based on Influence Functions"
    // Wiley Series in Probability and Mathematical Statistics, John Wiley & Sons, 1986
    // median 是中间值， 5.2 * (*mid)是信任域 trust region
    else 
    {
        vector<double> data(errors);
        vector<double>::iterator mid = data.begin() + errors.size() / 2;
        nth_element(data.begin(), mid, data.end());
        const double median = *mid;
        // threshold = 5.2 * MEDIAN(ABS(values-median));
        for(size_t i = 0; i < errors.size(); i++)
            data[i] = abs(errors[i] - median);
        nth_element(data.begin(), mid, data.end());
        threshold = median + 5.2 * (*mid);
    }
    LOG(INFO) << "angle threshold for inlier relative rotation is " << threshold * 180.0 / M_PI << " degree";
    set<size_t> covered_frames, valid_pair_idx, invalid_pair_idx;
    for(int i = 0; i < errors.size(); i++)
    {
        if(errors[i] < threshold)
        {
            valid_pair_idx.insert(i);
            covered_frames.insert(image_pairs[i].image_pair.first);
            covered_frames.insert(image_pairs[i].image_pair.second);
        }
        else 
            invalid_pair_idx.insert(i);
    }
    // 要求所有的图像至少都要有两个匹配对
    if(false)
    {
        vector<int> num_pairs_each_frame(global_rotations.size(), 0);
        for(const int& idx : valid_pair_idx)
        {
            num_pairs_each_frame[image_pairs[idx].image_pair.first]++;
            num_pairs_each_frame[image_pairs[idx].image_pair.second]++;
        }
        map<int, vector<int>> bad_frames;
        for(int i = 0; i < num_pairs_each_frame.size(); i++)
        {
            if(num_pairs_each_frame[i] < 2)
                bad_frames[i] = vector<int>();
        }
        for(const int& idx : invalid_pair_idx)
        {
            const MatchPair& p = image_pairs[idx];
            if(bad_frames.find(p.image_pair.first) != bad_frames.end())
                bad_frames[p.image_pair.first].push_back(idx);
            if(bad_frames.find(p.image_pair.second) != bad_frames.end())
                bad_frames[p.image_pair.second].push_back(idx);
        }
        for(auto& it : bad_frames)
        {
            // 之所以有这个判断条件，是因为在处理某些不太好的帧之后，可能会影响到其他的帧，让它从近邻数量不足变得足够了
            // 因此要额外检查一下，这样可以避免后面一些无谓的计算，而且还能减少误差。
            if(num_pairs_each_frame[it.first] >= 2)
                continue;
            sort(it.second.begin(), it.second.end(), [&](const int& a, const int& b){return errors[a] < errors[b];});
            for(int count = 0; num_pairs_each_frame[it.first] < 2 && count < it.second.size(); count++)
            {
                valid_pair_idx.insert(it.second[count]);
                const MatchPair& p = image_pairs[it.second[count]];
                num_pairs_each_frame[p.image_pair.first]++;
                num_pairs_each_frame[p.image_pair.second]++;
                covered_frames.insert(p.image_pair.first);
                covered_frames.insert(p.image_pair.second);
                invalid_pair_idx.erase(it.second[count]);
            }
        }
    }
    // 要求所有图像都要至少有一前一后两个匹配对，也就是说当前帧的是的id是 A，那么至少要有两帧(B,C)和A能匹配，要求 B < A < C
    // 这是为了保证后面的平移平均时相机的位姿是连续的。不然位姿可能就会在A处出现断开
    if(true)
    {
        // 第一个数字是位于该图像之前的图像的数量，第二个数字是位于该图像之后的图像的数量
        vector<pair<int,int>> num_neighbor_frames(global_rotations.size(), make_pair(0, 0));
        int range = 10, neighbor_size = 3;
        for(const int& idx : valid_pair_idx)
        {
            const MatchPair& p = image_pairs[idx];
            if((p.image_pair.second - p.image_pair.first) <= range)
            {
                num_neighbor_frames[p.image_pair.first].second++;
                num_neighbor_frames[p.image_pair.second].first++;
            }
        }
        map<int, vector<int>> bad_frames;
        for(int i = 0; i < num_neighbor_frames.size(); i++)
        {
            if(num_neighbor_frames[i].first < neighbor_size || num_neighbor_frames[i].second < neighbor_size)
                bad_frames[i] = vector<int>();
        }
        for(const int& idx : invalid_pair_idx)
        {
            const MatchPair& p = image_pairs[idx];
            if((p.image_pair.second - p.image_pair.first) <= range)
            {
                if(bad_frames.find(p.image_pair.first) != bad_frames.end() && num_neighbor_frames[p.image_pair.first].second < neighbor_size)
                    bad_frames[p.image_pair.first].push_back(idx);
                if(bad_frames.find(p.image_pair.second) != bad_frames.end() && num_neighbor_frames[p.image_pair.second].first < neighbor_size)
                    bad_frames[p.image_pair.second].push_back(idx);
            }
        }
        for(auto& it : bad_frames)
        {
            const int& frame_id = it.first;
            // 之所以有这个判断条件，是因为在处理某些不太好的帧之后，可能会影响到其他的帧，让它从近邻数量不足变得足够了
            // 因此要额外检查一下，这样可以避免后面一些无谓的计算，而且还能减少误差。
            if(num_neighbor_frames[frame_id].first >= neighbor_size && num_neighbor_frames[frame_id].second >= neighbor_size)
                continue;
            sort(it.second.begin(), it.second.end(), [&](const int& a, const int& b){return errors[a] < errors[b];});

            // 这是针对当前图像之前缺少图像
            for(int count = 0; num_neighbor_frames[frame_id].first < neighbor_size && count < it.second.size(); count++)
            {
                const MatchPair& p = image_pairs[it.second[count]];
                if(p.image_pair.second != frame_id)
                    continue;
                valid_pair_idx.insert(it.second[count]);
                num_neighbor_frames[p.image_pair.first].second++;
                num_neighbor_frames[p.image_pair.second].first++;
                invalid_pair_idx.erase(it.second[count]);
            }

            // 这是针对当前图像之后缺少图像
            for(int count = 0; num_neighbor_frames[frame_id].second < neighbor_size && count < it.second.size(); count++)
            {
                const MatchPair& p = image_pairs[it.second[count]];
                if(p.image_pair.first != frame_id)
                    continue;
                valid_pair_idx.insert(it.second[count]);
                num_neighbor_frames[p.image_pair.first].second++;
                num_neighbor_frames[p.image_pair.second].first++;
                invalid_pair_idx.erase(it.second[count]);
            }
        }
    }


    LOG(INFO) << "number of inlier relative rotation : " << valid_pair_idx.size() << ", covered " << covered_frames.size() << " frames";
    if(covered_frames.size() < global_rotations.size())
    {
        vector<int> invalid_frames;
        for(size_t i = 0; i < global_rotations.size(); ++i)
        {
            if(covered_frames.find(i) == covered_frames.end())
                invalid_frames.push_back(i);
        }
        LOG(INFO) << "invalid frames : " << Join(invalid_frames);
    }
    vector<MatchPair> valid_pairs;
    for(const int& idx : valid_pair_idx)
        valid_pairs.push_back(image_pairs[idx]);
    return valid_pairs;
}

bool RotationAveragingLeastSquare(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations)
{
    size_t num_cameras = global_rotations.size();
    if(num_cameras == 0 || image_pairs.empty())
    {
        cout << "not enough data in Rotation Averaging Least Square" << endl;
        return false;
    }
    const size_t num_relative_rotation = image_pairs.size();
    vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(num_relative_rotation * 12);   // 12 = 3 * 3 + 3
    Eigen::SparseMatrix<double>::Index cpt = 0;
    for(const MatchPair& p : image_pairs)
    {
        const Eigen::SparseMatrix<double>::Index i = p.image_pair.first;
        const Eigen::SparseMatrix<double>::Index j = p.image_pair.second;
        // weight * (R_jw - R_ji * R_iw) = 0
        double weight = 1;
        // A.block<3,3>(3 * cpt, 3 * i) = - Rij * weight;
        triplet_list.emplace_back(3 * cpt, 3 * i, - p.R_21(0,0) * weight);
        triplet_list.emplace_back(3 * cpt, 3 * i + 1, - p.R_21(0,1) * weight);
        triplet_list.emplace_back(3 * cpt, 3 * i + 2, - p.R_21(0,2) * weight);
        triplet_list.emplace_back(3 * cpt + 1, 3 * i, - p.R_21(1,0) * weight);
        triplet_list.emplace_back(3 * cpt + 1, 3 * i + 1, - p.R_21(1,1) * weight);
        triplet_list.emplace_back(3 * cpt + 1, 3 * i + 2, - p.R_21(1,2) * weight);
        triplet_list.emplace_back(3 * cpt + 2, 3 * i, - p.R_21(2,0) * weight);
        triplet_list.emplace_back(3 * cpt + 2, 3 * i + 1, - p.R_21(2,1) * weight);
        triplet_list.emplace_back(3 * cpt + 2, 3 * i + 2, - p.R_21(2,2) * weight);

        // A.block<3,3>(3 * cpt, 3 * j) = Identity * weight;
        triplet_list.emplace_back(3 * cpt, 3 * j, weight);
        triplet_list.emplace_back(3 * cpt + 1, 3 * j + 1, weight);
        triplet_list.emplace_back(3 * cpt + 2, 3 * j + 2, weight);
        ++cpt;
    }
    Eigen::MatrixXd AtA(3 * num_cameras, 3 * num_cameras);  // A.transpose() * A
    {
        Eigen::SparseMatrix<double> A(num_relative_rotation * 3, num_cameras * 3);
        A.setFromTriplets(triplet_list.begin(), triplet_list.end());
        triplet_list.clear();
        Eigen::SparseMatrix<double> AtA_sparse = A.transpose() * A;
        AtA = Eigen::MatrixXd(AtA_sparse);
    }
    // 求解方程 Ax=0 
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA, Eigen::ComputeEigenvectors);
    if(es.info() != Eigen::Success)
        return false;

    std::vector<std::pair<double, Eigen::VectorXd>> eigs(AtA.cols());
    for (Eigen::SparseMatrix<double>::Index i = 0; i < AtA.cols(); ++i)
    {
        eigs[i] = {es.eigenvalues()[i], es.eigenvectors().col(i)};
    }
    stable_sort(eigs.begin(), eigs.end(), 
        [](pair<double, Eigen::VectorXd> const& x, pair<double, Eigen::VectorXd> const& y)
        {return abs(x.first) < abs(y.first);});

    const Eigen::VectorXd& NullspaceVector0 = eigs[0].second;
    const Eigen::VectorXd& NullspaceVector1 = eigs[1].second;
    const Eigen::VectorXd& NullspaceVector2 = eigs[2].second;
    //--
    // Search the closest matrix :
    //  - From solution of SVD get back column and reconstruct Rotation matrix
    //  - Enforce the orthogonality constraint
    //     (approximate rotation in the Frobenius norm using SVD).
    //--
    // global rotation 里保存的是 R_cw
    for(size_t i = 0; i < num_cameras; i++)
    {
        Eigen::Matrix3d rotation;
        rotation << NullspaceVector0.segment(3 * i, 3),
                    NullspaceVector1.segment(3 * i, 3),
                    NullspaceVector2.segment(3 * i, 3);
        // 计算和rotation最接近的旋转矩阵，使用SVD分解来计算
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rotation, Eigen::ComputeFullV| Eigen::ComputeFullU);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d rot = U * V.transpose();
        // 旋转矩阵的行列式必须是正的
        if(rot.determinant() < 0)
            rot *= -1.0;

        global_rotations[i] = rot;
    }
    // 把第一个旋转设置为单位阵
    const Eigen::Matrix3d R0T = global_rotations[0].transpose();
    for(size_t i = 0; i < num_cameras; i++)
        global_rotations[i] *= R0T;

    return true;
}


bool RotationAveragingSpanningTree(const vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations, size_t start_idx)
{
    // relative rotation 保存的是从 {image1, image2} => R_21
    eigen_map<pair<size_t, size_t>, Eigen::Matrix3d> relative_rotations;
    vector<pair<size_t, size_t>> pairs;
    for(const MatchPair& p : image_pairs)
    {
        pairs.emplace_back(p.image_pair);
        relative_rotations[p.image_pair] = p.R_21;
        relative_rotations[pair<size_t,size_t>(p.image_pair.second, p.image_pair.first)] = p.R_21.transpose();
    }
    PoseGraph graph(pairs);
    vector<vector<size_t>> spanning_tree = graph.FindMaximumSpanningTree();

    global_rotations[start_idx] = Eigen::Matrix3d::Identity();
    // 用一个队列来遍历最大生成树，因为最大生成树的根节点可以是任意的
    using Link = pair<size_t, size_t>;
    queue<Link> stack;      //{node_id, parent_id}
    stack.push(Link(start_idx, 0));
    while(!stack.empty())
    {
        const Link& link = stack.front();
        const vector<size_t>& linked_nodes = spanning_tree[link.first];
        for(const size_t& node : linked_nodes)
        {
            // 在当前节点的相连节点中找到了当前节点的父节点（这是一定能找到的），就要计算当前节点的旋转
            if(node == link.second)
            {
                Eigen::Matrix3d R_cp = relative_rotations.find(Link(link.second, link.first))->second;
                global_rotations[link.first] = R_cp * global_rotations[link.second];
            }
            else 
                stack.push(Link(node, link.first));
        }
        stack.pop();
    }
    return true;
}

bool RotationAveragingL2(int num_threads , const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations)
{
    if(global_rotations.empty())
    {
        cout << "not enough data in Rotation Averaging L2" << endl;
        return false;
    }
    eigen_vector<Eigen::Vector3d> global_angleAxis(global_rotations.size());
    for(size_t i = 0; i < global_rotations.size(); i++)
        ceres::RotationMatrixToAngleAxis(global_rotations[i].data(), global_angleAxis[i].data());
    
    ceres::Problem problem;
    ceres::LossFunction* loss_function = new ceres::SoftLOneLoss(0.07);     // 0.07 弧度，也就是4度
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        size_t idx1 = image_pairs[i].image_pair.first;
        size_t idx2 = image_pairs[i].image_pair.second;
        const Eigen::Matrix3d& R_21 = image_pairs[i].R_21;
        Eigen::Vector3d angleAxis_21;
        ceres::RotationMatrixToAngleAxis(R_21.data(), angleAxis_21.data());
        ceres::CostFunction* cost_function = PairWiseRotationResidual::Create(angleAxis_21);
        problem.AddResidualBlock(cost_function, loss_function, 
                global_angleAxis[idx1].data(), global_angleAxis[idx2].data());
    }
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    // 由于整个问题是比较稀疏的，因此使用稀疏求解器
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
    options.num_threads = num_threads;
    // 求解问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(summary.IsSolutionUsable())
    {
        for(size_t i = 0; i < global_angleAxis.size(); i++)
            ceres::AngleAxisToRotationMatrix(global_angleAxis[i].data(), global_rotations[i].data());
        return true;
    }
    return false;
}

bool RotationAveragingL1(std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations, 
                            size_t start_idx, double angle_threshold)
{
    size_t num_cameras = global_rotations.size();
    assert(start_idx < num_cameras);
    // relative rotation 保存的是从 {image1, image2} => R_21
    eigen_map<pair<size_t, size_t>, Eigen::Matrix3d> relative_rotations;
    for(const MatchPair& p : image_pairs)
    {
        relative_rotations[p.image_pair] = p.R_21;
        relative_rotations[pair<size_t,size_t>(p.image_pair.second, p.image_pair.first)] = p.R_21.transpose();
    }
    #if 1
    // 使用最大生成树得到初始的全局旋转
    RotationAveragingSpanningTree(image_pairs, global_rotations, start_idx);
    assert(global_rotations[start_idx] == Eigen::Matrix3d::Identity());
    // 使用IRLS方法进行优化，根据已有的相对旋转来优化全局旋转
    bool success = RotationAveragingRefineL1(global_rotations, start_idx, relative_rotations, 1);
    if(!success)
    {
        LOG(ERROR) << "rotation averaging refine L1 failed";
        return false;
    }
    // 输出一下经过L1RA之后的旋转，这样如果之后需要再进行旋转平均的话，就直接读这个结果即可
    ofstream f("rotations_after_L1.txt");
    for(size_t i = 0; i < global_rotations.size(); i++)
        f << global_rotations[i](0,0) << " " << global_rotations[i](0,1) << " " << global_rotations[i](0,2) << " "
          << global_rotations[i](1,0) << " " << global_rotations[i](1,1) << " " << global_rotations[i](1,2) << " "
          << global_rotations[i](2,0) << " " << global_rotations[i](2,1) << " " << global_rotations[i](2,2) << endl;
    f.close();
    #else
    // 读取之前的L1RA的结果，避免再次计算耗时
    // 主要是调参的时候用一下
    int count = 0;
    ifstream f2("rotations_after_L1.txt");
    while(!f2.eof())
    {
        Eigen::Matrix3d R;
        f2 >> R(0,0) >> R(0,1) >> R(0,2) >> R(1,0) >> R(1,1) >> R(1,2) >> R(2,0) >> R(2,1) >> R(2,2);
        if(f2.eof())
            break;
        global_rotations[count] = R;
        count++;
    }
    #endif

    // 对图像对进行过滤
    image_pairs = FilterPairs(image_pairs, global_rotations, angle_threshold);
    return true;

}

bool RotationAveragingRefineL1(eigen_vector<Eigen::Matrix3d>& global_rotations, size_t start_idx,
                                const eigen_map<pair<size_t, size_t>, Eigen::Matrix3d>& relative_rotations,
                                const int weight_function)
{
    const size_t m = relative_rotations.size() * 3;
    const size_t n = (global_rotations.size() - 1) * 3;     // 这里减去1是为了保证start idx所对应的旋转是单位阵
    // 构建 Ax=b 中的A
    Eigen::SparseMatrix<double> A(m,n);
    A.reserve(A.rows() * 2);
    Eigen::SparseMatrix<double>::Index i = 0, j = 0;
    for(eigen_map<pair<size_t, size_t>, Eigen::Matrix3d>::const_iterator it_rel = relative_rotations.begin(); 
        it_rel != relative_rotations.end(); ++it_rel)
    {
        const size_t& idx1 = it_rel->first.first;
        const size_t& idx2 = it_rel->first.second;
        if(idx1 != start_idx)
        {
            j = 3 * (idx1 - (idx1 >= start_idx));
            A.insert(i+0,j+0) = -1.0;
            A.insert(i+1,j+1) = -1.0;
            A.insert(i+2,j+2) = -1.0;
        }
        if(idx2 != start_idx)
        {
            j = 3 * (idx2 - (idx2 >= start_idx));
            A.insert(i+0,j+0) = 1.0;
            A.insert(i+1,j+1) = 1.0;
            A.insert(i+2,j+2) = 1.0;
        }
        i += 3;
    }
    A.makeCompressed();
    // L1 rotation averaging
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd b(m);
    // current error    last error
    double curr_e = numeric_limits<double>::max(), last_e;
    int iter = 0;
    do
    {
        // 计算误差
        size_t count = 0;
        for(eigen_map<pair<size_t, size_t>, Eigen::Matrix3d>::const_iterator it_rel = relative_rotations.begin(); 
            it_rel != relative_rotations.end(); ++it_rel)
        {
            const Eigen::Matrix3d& R_1w = global_rotations[it_rel->first.first];
            const Eigen::Matrix3d& R_2w = global_rotations[it_rel->first.second];
            const Eigen::Matrix3d& R_21 = it_rel->second;
            const Eigen::Matrix3d error_mat = R_2w.transpose() * R_21 * R_1w;
            Eigen::Vector3d error_angleaxis;
            ceres::RotationMatrixToAngleAxis(error_mat.data(), error_angleaxis.data());
                                        
            b.block<3,1>(3 * count, 0) = error_angleaxis;
            count ++;
        }
        L1Solver<Eigen::SparseMatrix<double>>::Options options;
        L1Solver<Eigen::SparseMatrix<double>> l1_solver(options, A);
        l1_solver.Solve(b, &x);
        last_e = curr_e;
        curr_e = x.norm();
        if(last_e < curr_e)
            break;
        // 更新global rotation
        for(size_t r = 0; r < global_rotations.size(); r++)
        {
            if(r == start_idx)
                continue;
            Eigen::Matrix3d& R_cw = global_rotations[r];
            const size_t i = r - (r >= start_idx);
            Eigen::Vector3d update_angleAxis = Eigen::Vector3d(x.block<3,1>(3 * i, 0));
            Eigen::Matrix3d update_rotation;
            ceres::AngleAxisToRotationMatrix(update_angleAxis.data(), update_rotation.data());
            R_cw = R_cw * update_rotation;
        }
    }while(++iter < 32 && curr_e > 1e-5 && (last_e-curr_e)/curr_e > 1e-2);
    LOG(INFO) << "L1 rotation averaging converged in " << iter << " iterations";
    // Iteratively Reweighted Least Squares (IRLS) implementation
    x = Eigen::VectorXd::Zero(n);       // x设置为全0就相当于信任之前的所有global rotation
    b = Eigen::VectorXd(m);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> linear_solver;
    linear_solver.analyzePattern(A.transpose() * A);
    if(linear_solver.info() != Eigen::Success)
    {
        LOG(ERROR) << "Cholesky decomposition failed.";
        return false;
    }
    const double squre_sigma = (5.0 * M_PI / 180.0) * (5.0 * M_PI / 180.0);     // 5度
    Eigen::ArrayXd errors, weights;
    Eigen::VectorXd last_x(n);
    curr_e = numeric_limits<double>::max();
    last_e = -1;
    iter = 0;
    do
    {
        last_x = x;
        // 计算误差
        size_t count = 0;
        for(eigen_map<pair<size_t, size_t>, Eigen::Matrix3d>::const_iterator it_rel = relative_rotations.begin(); 
            it_rel != relative_rotations.end(); ++it_rel)
        {
            const Eigen::Matrix3d& R_1w = global_rotations[it_rel->first.first];
            const Eigen::Matrix3d& R_2w = global_rotations[it_rel->first.second];
            const Eigen::Matrix3d& R_21 = it_rel->second;
            const Eigen::Matrix3d error_mat(R_2w.transpose() * R_21 * R_1w);
            Eigen::Vector3d error_angleaxis;
            ceres::RotationMatrixToAngleAxis(error_mat.data(), error_angleaxis.data());       
            b.block<3,1>(3 * count, 0) = error_angleaxis;
            count ++;
        }
        // 计算每一项误差的权重
        if(weight_function == 1)
        {
            errors = (A * x - b).array();
            weights = errors.abs().pow(-1.5);
        }
        else if(weight_function == 2)
        {
            errors = (A * x - b).array();
            weights = squre_sigma / (errors.square() + squre_sigma).square();
        }
        

        // Update the factorization for the weighted values
        const Eigen::SparseMatrix<double> at_weight = A.transpose() * weights.matrix().asDiagonal();
        linear_solver.factorize(at_weight * A);
        if(linear_solver.info() != Eigen::Success)
        {
            LOG(ERROR) << "Failed to factorize the least squares system";
            return false;
        }
        // Solve the least squares problem
        x = linear_solver.solve(at_weight * b);
        if (linear_solver.info() != Eigen::Success) 
        {
            LOG(ERROR) << "Failed to solve the least squares system";
            return false;
        }
        // 更新global rotation
        for(size_t r = 0; r < global_rotations.size(); r++)
        {
            if(r == start_idx)
                continue;
            Eigen::Matrix3d& R_cw = global_rotations[r];
            const size_t i = r - (r >= start_idx);
            const Eigen::Vector3d update_angleAxis = Eigen::Vector3d(x.block<3,1>(3 * i, 0));
            Eigen::Matrix3d update_rotation;
            ceres::AngleAxisToRotationMatrix(update_angleAxis.data(), update_rotation.data());
            R_cw = R_cw * update_rotation;
        }
        last_e = curr_e;
        curr_e = (last_x - x).norm();
    }while(++iter < 32 && curr_e > 1e-5 && (last_e-curr_e)/curr_e > 1e-2);
    LOG(INFO) << "Iteratively Reweighted Least Squares converged in " << iter << " iterations";
    return true;
}