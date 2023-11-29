/*
 * @Author: Diantao Tu
 * @Date: 2021-12-15 09:58:23
 */

#include "Triangulate.h"

Eigen::Vector3d Triangulate2View(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    Eigen::Vector3d point1(p1.x, p1.y, p1.z);
    Eigen::Vector3d point2(p2.x, p2.y, p2.z);
    const Eigen::Vector3d trans_12 = -R_21.transpose() * t_21;
    const Eigen::Vector3d bearing_2_in_1 = R_21.transpose() * point2;

    Eigen::Matrix2d A;
    A(0, 0) = point1.dot(point1);
    A(1, 0) = bearing_2_in_1.dot(point1);
    A(0, 1) = -point1.dot(bearing_2_in_1);
    A(1, 1) = -bearing_2_in_1.dot(bearing_2_in_1);

    const Eigen::Vector2d b{point1.dot(trans_12), bearing_2_in_1.dot(trans_12)};

    const Eigen::Vector2d lambda = A.inverse() * b;
    const Eigen::Vector3d pt_1 = lambda(0) * point1;
    const Eigen::Vector3d pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
    return (pt_1 + pt_2) / 2.0;
}

// 这个不太准，不知道为啥, 所以不要用
Eigen::Vector3d Triangulate2View_2(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    Eigen::Vector3d point1(p1.x, p1.y, p1.z);
    Eigen::Vector3d point2(p2.x, p2.y, p2.z);

    const Eigen::Vector3d trans_12 = -R_21.transpose() * t_21;
    const Eigen::Vector3d bearing_2_in_1 = R_21.transpose() * point2 + trans_12 - trans_12;
    Eigen::Matrix<double, 3, 2> A;
    A(0,0) = point1.x();
    A(1,0) = point1.y();
    A(2,0) = point1.z();
    A(0,1) = -bearing_2_in_1.x();
    A(1,1) = -bearing_2_in_1.y();
    A(2,1) = -bearing_2_in_1.z();
    Eigen::Vector3d b = bearing_2_in_1 + trans_12 - point1;
    // 进行svd分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(A,
                                                 Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    // 构建SVD分解结果
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();

    // 构建S矩阵
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();

    for (unsigned int i = 0; i < D.size(); ++i) {

        if (D(i, 0) > 1e-6) {
            S(i, i) = 1 / D(i, 0);
        } else {
            S(i, i) = 0;
        }
    }

    // pinv_matrix = V * S * U^T
    Eigen::MatrixXd A_inv = V * S * U.transpose();

    // Eigen::Vector2d lambda = A.inverse() * b;
    // Eigen::Vector2d lambda = A_inv * b;
    Eigen::Vector2d lambda = (A.transpose() * A).inverse() * A.transpose() * b;
    // cout << "inverse A:" << endl;
    // cout << A_inv << endl << endl;
    // cout << (A.transpose() * A).inverse() * A.transpose() << endl << endl;

    const Eigen::Vector3d pt_1 = lambda(0) * point1;
    const Eigen::Vector3d pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
    return (pt_1 + pt_2) / 2.0;
}

Eigen::Vector3d Triangulate2ViewIDWM(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2)
{
    const Eigen::Vector3d point1(p1.x, p1.y, p1.z);
    const Eigen::Vector3d point2(p2.x, p2.y, p2.z);
    Eigen::Vector3d Rp1 = R_21 * point1;
    const double p_norm = Rp1.cross(point2).norm();
    const double q_norm = Rp1.cross(t_21).norm();
    const double r_norm = point2.cross(t_21).norm();
    // Eq. (10)     这是在相机2的坐标系下，还要变换回相机1下
    Eigen::Vector3d triangulated = ( q_norm / (q_norm + r_norm) )
        * ( t_21 + (r_norm / p_norm) * (Rp1 + point2) );

    // Eq. (7)
    const Eigen::Vector3d lambda0_Rp1 = (r_norm / p_norm) * Rp1;
    const Eigen::Vector3d lambda1_p2 = (q_norm / p_norm) * point2;

    // Eq. (9) - test adequation
    if((t_21 + lambda0_Rp1 - lambda1_p2).squaredNorm() <  
        std::min(std::min(
        (t_21 + lambda0_Rp1 + lambda1_p2).squaredNorm(),
        (t_21 - lambda0_Rp1 - lambda1_p2).squaredNorm()),
        (t_21 - lambda0_Rp1 + lambda1_p2).squaredNorm()))
    {
        return R_21.transpose() * (triangulated - t_21);
    }
    else 
        return Eigen::Vector3d::Ones() * std::numeric_limits<double>::infinity();
}

// 最小化代数误差的三角化过程，从openMVG里抄来的，
// 当特征点进行了各向同性的正则化(isotropic normalization)之后，三角化的效果会更好
// 经过测试，效果确实好了一点
Eigen::Vector3d TriangulateNViewAlgebraic(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    assert(R_cw_list.size() > 2);
    Eigen::Matrix4d AtA = Eigen::Matrix4d::Zero();
    for(size_t i = 0; i < points.size(); i++)
    {
        // const Eigen::Vector3d point = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z).normalized();
        // 把旋转平移组成3x4的矩阵 [R|t]
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        Eigen::Matrix<double, 3, 4> cost = pose - point_norm * point_norm.transpose() * pose;
        AtA += cost.transpose() * cost;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
    Eigen::Vector4d p = eigen_solver.eigenvectors().col(0);
    Eigen::Vector3d point_world = p.hnormalized();
    if(eigen_solver.info() == Eigen::Success)
        return point_world;
    else 
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
}

// 这是最常规的多视图三角化的方法，把输入的向量的z坐标都变成1，这样每个特征点可以提供两组约束
//             | x_1 * P_1^3 - P_1^1 |
//             | y_1 * P_1^3 - P_1^2 |      | X_1 |
//             | x_2 * P_2^3 - P_2^1 |      | X_2 |
//             | y_2 * P_2^3 - P_2^2 |   *  | X_3 |  =  0
//             | x_3 * P_3^3 - P_3^1 |      | X_4 |
//             | y_3 * P_3^3 - P_3^1 |
// 这里的 x_1 y_1 x_2 y_2 就代表每个特征点的第一维和第二维（因为第三维是1）   
// P_1^1 P_1^2 代表位姿矩阵的第一行，第二行。 P_1 = [R_1|t_1] 是第1张图像的位姿
Eigen::Vector3d TriangulateNView1(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    const size_t num_row = 2 * R_cw_list.size();
    Eigen::MatrixXd A(num_row, 4) ;
    for(size_t i = 0; i < points.size(); i++)
    {
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
        point_norm /= point_norm(2);
        // 把旋转平移组成3x4的矩阵 [R|t]
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        A.row(2 * i) = point_norm(0) * pose.row(2) - pose.row(0);
        A.row(2 * i + 1) = point_norm(1) * pose.row(2) - pose.row(1);
    }
    Eigen::Matrix<double, 4, 4> AtA = A.transpose() * A;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
    Eigen::Vector4d p = eigen_solver.eigenvectors().col(0);
    Eigen::Vector3d point_world = p.hnormalized();
    if(eigen_solver.info() == Eigen::Success)
        return point_world;
    else 
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
}

// 这是另一种三角化的方法，和上面的方法的区别就是没有把输入特征的z变为1，那么每一个特征可以多提供一个约束（每个特征3个约束）
// 但是相应的，需要求解的未知量也多了一个（三角化的三维点在特征点处投影的深度）
// 具体的推导过程见 公式推导.md
Eigen::Vector3d TriangulateNView2(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    const size_t num_views = points.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * num_views, 4 + num_views);
    for(size_t i = 0; i < points.size(); i++)
    {
        Eigen::Vector3d point_norm = Eigen::Vector3d(points[i].x, points[i].y, points[i].z);
        // 把旋转平移组成3x4的矩阵 [R|t]
        Eigen::Matrix<double, 3, 4> pose = (Eigen::Matrix<double, 3, 4>() << R_cw_list[i], t_cw_list[i]).finished();
        A.block<3, 4>(3 * i, 0) = -pose;
        A.block<3, 1>(3 * i, 4 + i) = point_norm;
    }
    Eigen::VectorXd X(num_views + 4);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    X = svd.matrixV().col(A.cols() - 1);
    return X.head(4).hnormalized();
}

Eigen::Vector3d TriangulateNView(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points)
{
    assert(R_cw_list.size() == t_cw_list.size());
    assert(R_cw_list.size() == points.size());
    if(R_cw_list.size() == 2)
    {
        Eigen::Matrix3d R_21 = R_cw_list[1] * R_cw_list[0].transpose();     // R_21 = R_2w * R_w1
        Eigen::Vector3d t_21 = t_cw_list[1] - R_21 * t_cw_list[0];           // t_21 = t_2w - R_21 * t_1w
        Eigen::Vector3d point_triangulated = Triangulate2View(R_21, t_21, points[0], points[1]);
        Eigen::Vector3d t_w1 = - R_cw_list[0].transpose() * t_cw_list[0];   // t_w1 = - R_w1 * t_1w
        Eigen::Vector3d point_world = R_cw_list[0].transpose() * point_triangulated + t_w1;     // P_w = R_wc * P_c + t_wc
        return point_world;
    }
    else if(R_cw_list.size() > 2)
    {
        // 这三种方法效果依次变差
        Eigen::Vector3d point_triangulated = TriangulateNViewAlgebraic(R_cw_list, t_cw_list, points);
        // Eigen::Vector3d point_triangulated = TriangulateNView2(R_cw_list, t_cw_list, points);
        // Eigen::Vector3d point_triangulated = TriangulateNView1(R_cw_list, t_cw_list, points);
        return point_triangulated;
    }
    else 
    {
        LOG(ERROR) << "Invalid number in triangulate";
        return numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
    }
}