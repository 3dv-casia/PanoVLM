/*
 * @Author: Diantao Tu
 * @Date: 2022-04-30 16:18:24
 */
#include "BATA.h"

using namespace std;

// 对稀疏矩阵进行拼接，把四个矩阵 ABCD 拼接为一个矩阵 [A B ; C D] ，也就是左上角为A，右上角为B，左下角为C，右下角为D
// 对ABCD的要求是：1. 矩阵必须都是 colMajor的 2. A和B行数相等，A和C列数相等
// 由于矩阵都是列优先的，所以只能按列拼接，也就是水平拼接，所以整体步骤为： 
// 1. 先把ABCD都转置得到At Bt Ct Dt 
// 2. 把 At Ct 进行水平拼接，得到 [At Ct]，把Bt Dt进行水平拼接，得到 [Bt Dt]
// 3. 把[At Ct] 转置，得到 [A; C] ，把[Bt Dt]转置得到 [B;D]
// 4. 把 [A; C]和[B;D]水平拼接得到 [A B ; C D]
Eigen::SparseMatrix<double> SpliceMatrix(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double>& B, 
                                        const Eigen::SparseMatrix<double>& C, const Eigen::SparseMatrix<double>& D)
{
    assert(!A.IsRowMajor && !B.IsRowMajor && !C.IsRowMajor && !D.IsRowMajor);
    Eigen::SparseMatrix<double> At = A.transpose();
    Eigen::SparseMatrix<double> Bt = B.transpose();
    Eigen::SparseMatrix<double> Ct = C.transpose();
    Eigen::SparseMatrix<double> Dt = D.transpose();
    Eigen::SparseMatrix<double> AtCt(At.rows(), At.cols() + Ct.cols());
    Eigen::SparseMatrix<double> BtDt(Bt.rows(), Bt.cols() + Dt.cols());
    AtCt.leftCols(At.cols()) = At;
    AtCt.rightCols(Ct.cols()) = Ct;
    BtDt.leftCols(Bt.cols()) = Bt;
    BtDt.rightCols(Dt.cols()) = Dt;
    Eigen::SparseMatrix<double> spliced(A.rows() + C.rows(), A.cols() + B.cols());
    spliced.leftCols(AtCt.rows()) = AtCt.transpose();
    spliced.rightCols(BtDt.rows()) = BtDt.transpose();
    return spliced;
}

Eigen::MatrixXd LUDRevised(const BATAConfig& config, const int num_camera,
                            const Eigen::MatrixXd& tij_all, const Eigen::SparseMatrix<double>& At0_full,
                            Eigen::VectorXd& S)
{
    int num_relative_pose = tij_all.cols();
    Eigen::VectorXd tij_sq_sum = tij_all.array().square().colwise().sum();
    Eigen::SparseMatrix<double> Aeq1(num_relative_pose, 3 * num_relative_pose);
    Aeq1.reserve(num_relative_pose * 3);
    // 这个插入是按照列增加的顺序进行的，而且Aeq1本身就是colMajor的，因此直接插入速度会比较快，就不需要使用三元组了
    for(size_t i = 0; i < num_relative_pose; i++)
    {
        Aeq1.insert(i, 3 * i) =  tij_all(0,i);
        Aeq1.insert(i, 3 * i + 1) =  tij_all(1,i);
        Aeq1.insert(i, 3 * i + 2) =  tij_all(2,i);
    }
    Aeq1 = Aeq1 * At0_full;
    // Aeq 每一列只有两个非零元素，其中Aeq的第一行是Aeq1的每一列之和，后三行是单位阵水平堆叠
    Eigen::SparseMatrix<double> Aeq(4, Aeq1.cols());
    Aeq.reserve(2 * Aeq1.cols());  
    for(size_t i = 0; i < Aeq.cols(); i++)
    {
        Aeq.insert(0, i) = Aeq1.col(i).sum();
        Aeq.insert(i % 3 + 1, i) = 1;
    }
    Eigen::SparseMatrix<double> Aeq_transpose = Aeq.transpose();
    Eigen::SparseMatrix<double> beq(4,1);
    beq.insert(0,0) = num_relative_pose;
    // Initialization with LUDRevised
    Eigen::VectorXd Svec = Eigen::VectorXd::Random(num_relative_pose).array().abs();
    // 根据相对位姿的尺度设置初始尺度,初始尺度保存在S中
    // 如果S对应的位置是-1，就代表这一对匹配没有尺度信息，那么就用随机的尺度代替
    if(S.rows() == num_relative_pose)
        Svec = (S.array() < 0).select(Svec, S);
    ofstream f("Svec.txt");
    f << Svec.transpose() << endl;
    f.close();
    Svec *= num_relative_pose / Svec.sum();
    S = Svec.replicate(1,3).reshaped<Eigen::RowMajor>();
    Eigen::VectorXd W = Eigen::VectorXd::Ones(3 * num_relative_pose);
    W = W.array().sqrt();
    Eigen::VectorXd t;
    for(int iter = 0; iter < config.init_iteration; iter ++)
    {
        Eigen::SparseMatrix<double> A(3 * num_relative_pose, 3 * num_relative_pose);
        A.setIdentity();
        A = A * W.asDiagonal() * At0_full;
        Eigen::SparseMatrix<double> At = A.adjoint();
        Eigen::MatrixXd B = W.array() * S.array() * tij_all.reshaped<Eigen::ColMajor>().array();
        Eigen::SparseMatrix<double> A_full(A.cols() + 4, A.cols() + 4);
        // A_full 可以分为四块，左上角是2 * A^T * A
        // 右上角是 Aeq.transpose()
        // 左下角是 Aeq
        // 右下角是 零矩阵
        A_full = SpliceMatrix(2 * At * A, Aeq_transpose, Aeq, Eigen::SparseMatrix<double>(4,4));

        Eigen::SparseMatrix<double> b = SpliceMatrix(2 * At * Eigen::SparseMatrix<double>(B.sparseView()), Eigen::SparseMatrix<double>(0,0), 
                                                    beq, Eigen::SparseMatrix<double>(0,0));

        // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A_full);
        if(solver.info() != Eigen::Success)
            LOG(ERROR) << "decomposition failed" << endl;
        Eigen::VectorXd X;
        X = solver.solve(b);
        if(solver.info() != Eigen::Success)
            LOG(ERROR) << "solving failed" << endl;

        t = X.head(3 * num_camera);
        Eigen::MatrixXd Aij = (At0_full * t).reshaped(3, num_relative_pose);

        Svec = (Aij.array() * tij_all.array()).colwise().sum();
        Svec = Svec.array() / tij_sq_sum.array();  

        S = Svec.replicate(1,3).reshaped<Eigen::RowMajor>();

        Eigen::MatrixXd tmp = At0_full * t;
        tmp = (tmp.array() - S.array() * tij_all.reshaped<Eigen::ColMajor>().array()).reshaped(3, num_relative_pose);
        tmp = tmp.array().square();
        Eigen::MatrixXd tmp_col_sum = tmp.colwise().sum();
        Eigen::MatrixXd Wvec = (tmp_col_sum.array() + config.delta).pow(-0.5);
        W = Wvec.replicate(3,1).reshaped<Eigen::ColMajor>();
        W = W.array().sqrt();
    }
    Eigen::MatrixXd pose = t.reshaped(3, num_camera);
    for(size_t i = 1; i < num_camera; i++)
        pose.col(i) -= pose.col(0);
    pose.col(0).setZero();
    return pose;
}

Eigen::MatrixXd IRLS(const BATAConfig& config, const Eigen::MatrixXd& tij_all,
                    const Eigen::SparseMatrix<double>& At0 , const Eigen::MatrixXd& init_pose,
                    Eigen::VectorXd& S )
{
    int num_relative_pose = tij_all.cols();
    int num_camera = init_pose.cols();
    Eigen::VectorXd t = init_pose.rightCols(num_camera - 1).reshaped<Eigen::ColMajor>();
    ProcessBar bar((config.outer_iteration - 1) * config.inner_iteration, 0.1);
    for(int i = 0; i < config.outer_iteration - 1; i++)
    {
        Eigen::SparseMatrix<double> A(3 * num_relative_pose, 3 * num_relative_pose);
        A.setIdentity();
        A = S.cwiseInverse().asDiagonal() * At0;
        Eigen::VectorXd B = tij_all.reshaped<Eigen::ColMajor>();
        Eigen::MatrixXd tmp = ((A*t - B).reshaped(3, num_relative_pose)).array().square();
        Eigen::VectorXd tmp_col_sum = tmp.colwise().sum().cwiseSqrt();
        
        Eigen::VectorXd Wvec = Eigen::VectorXd::Zero(tmp_col_sum.rows());
        Wvec = (tmp_col_sum.array() < config.robust_threshold).select(1, Wvec);
        Wvec = (tmp_col_sum.array() > config.robust_threshold).select(config.robust_threshold / tmp_col_sum.array() , Wvec);
        
        Eigen::VectorXd W = Wvec.replicate(1,3).reshaped<Eigen::RowMajor>();
        W = W.cwiseSqrt();
        for(int j = 0; j < config.inner_iteration; j++)
        {
            Eigen::MatrixXd Aij = (At0 * t).reshaped(3, num_relative_pose);
            Eigen::VectorXd Svec = Aij.cwiseAbs2().colwise().sum();
            tmp_col_sum = (Aij.array() * tij_all.array()).colwise().sum();
            Svec = Svec.array() / tmp_col_sum.array();
            Svec = (Svec.array() < 0).select(numeric_limits<double>::infinity(), Svec);
            S = Svec.replicate(1,3).reshaped<Eigen::RowMajor>();
            A = Eigen::VectorXd(W.array() / S.array()).asDiagonal() * At0;
            B = W.array() * tij_all.reshaped<Eigen::ColMajor>().array();
            // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.compute(A.transpose() * A);
            if(solver.info() != Eigen::Success)
                LOG(ERROR) << "decomposition failed" << endl;
            t = solver.solve(A.transpose() * B);
            if(solver.info() != Eigen::Success)
                LOG(ERROR) << "solving failed" << endl;
            bar.Add();
        }
    }
    Eigen::MatrixXd pose(3, num_camera);
    pose << Eigen::Vector3d::Zero(), t.reshaped(3, num_camera-1);
    return pose;
}

eigen_vector<Eigen::Vector3d> BATA(const vector<pair<int,int>>& pairs, const eigen_vector<Eigen::Vector3d>& relative_pose, const BATAConfig& config, const string& output_path)
{
    assert(pairs.size() == relative_pose.size());
    const int num_relative_pose = relative_pose.size();
    Eigen::VectorXd S(num_relative_pose);      // 尺度
    Eigen::MatrixXd tij_all(3, num_relative_pose);
    int num_camera, max_camera_id = -1, min_camera_id = INT16_MAX;
    Eigen::MatrixXd tij_index(2, num_relative_pose);
    for(int i = 0; i < num_relative_pose; i++)
    {
        const pair<int,int>& p = pairs[i];
        max_camera_id = max(max_camera_id, max(p.first, p.second));
        min_camera_id = min(min_camera_id, min(p.first, p.second));
        tij_index(0, i) = p.first;
        tij_index(1, i) = p.second;
        double scale = relative_pose[i].norm();
        if(scale > 0.99 && scale < 1.01)
            scale = -1;
        S[i] = scale;
        tij_all.col(i) = relative_pose[i].normalized();
    }
    num_camera = max_camera_id - min_camera_id + 1;
    Eigen::VectorXd tij_sq_sum = tij_all.array().square().colwise().sum();
    
    // 设置矩阵A，是一个分块矩阵，共 num_relative_pose 行，num_camera 列，里面的每个分块是单位阵，负单位阵或者零矩阵
    // 对于第k对相对位姿，是关于图像i和j的，那么A矩阵的第k行的第i列是单位阵，第k行第j列是负单位阵，其他为零
    vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(num_relative_pose * 6);
    for(size_t i = 0; i < num_relative_pose; i++)
    {
        // 这里减去了min camera id是因为可能初始的索引不是从0开始的，所以要减去这个，让索引从零开始
        const int idx1 = pairs[i].first - min_camera_id;
        const int idx2 = pairs[i].second - min_camera_id;
        triplet_list.emplace_back(3 * i, 3 * idx1, 1);
        triplet_list.emplace_back(3 * i + 1, 3 * idx1 + 1, 1);
        triplet_list.emplace_back(3 * i + 2, 3 * idx1 + 2, 1);
        triplet_list.emplace_back(3 * i, 3 * idx2, -1);
        triplet_list.emplace_back(3 * i + 1, 3 * idx2 + 1, -1);
        triplet_list.emplace_back(3 * i + 2, 3 * idx2 + 2, -1);
    }
    Eigen::SparseMatrix<double> At0_full(3 * num_relative_pose, 3 * num_camera);
    At0_full.setFromTriplets(triplet_list.begin(), triplet_list.end());
    triplet_list.clear();
    Eigen::SparseMatrix<double> At0 = At0_full.rightCols(3*num_camera -3);  // 舍弃前三列，也就是第一个相机对应的矩阵
    
    Eigen::MatrixXd pose_LUD = LUDRevised(config, num_camera, tij_all, At0_full, S);
    eigen_vector<Eigen::Vector3d> init_pose;
    ofstream f(output_path + "/LUD.txt");
    for(size_t i = 0; i < pose_LUD.cols(); i++)
    {
        f << pose_LUD.col(i).transpose() << endl;
        init_pose.push_back(pose_LUD.col(i));
    }
    f.close();
    CameraCenterPCD(output_path + "/LUD.pcd", init_pose);
    LOG(INFO) << "Finish init with LUD";
    Eigen::MatrixXd pose_BATA = IRLS(config, tij_all, At0, pose_LUD, S);
    eigen_vector<Eigen::Vector3d> global_pose;
    for(size_t i = 0; i < pose_BATA.cols(); i++)
        global_pose.push_back(pose_BATA.col(i));
    return global_pose;
}