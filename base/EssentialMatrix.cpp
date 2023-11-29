/*
 * @Author: Diantao Tu
 * @Date: 2021-11-23 14:38:22
 */

#include "EssentialMatrix.h"

using namespace std;

Eigen::Matrix3d ComputeEssential(const std::vector<cv::Point3f>& points1, const std::vector<cv::Point3f>& points2)
{
    assert(points2.size() == points1.size());
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(points1.size(), 9);
    for(size_t i = 0; i < points1.size(); i++)
    {
        A(i,0) = points2[i].x * points1[i].x;
        A(i,1) = points2[i].x * points1[i].y;
        A(i,2) = points2[i].x * points1[i].z;
        A(i,3) = points2[i].y * points1[i].x;
        A(i,4) = points2[i].y * points1[i].y;
        A(i,5) = points2[i].y * points1[i].z;
        A(i,6) = points2[i].z * points1[i].x;
        A(i,7) = points2[i].z * points1[i].y;
        A(i,8) = points2[i].z * points1[i].z;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(A.transpose() * A);
    Eigen::Matrix<double, 9, 1> e = solver.eigenvectors().leftCols<1>();
    Eigen::Matrix3d init_E_21 = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(e.data());
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(init_E_21, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix3d& U = svd.matrixU();
    Eigen::Vector3d lambda = svd.singularValues();
    const Eigen::Matrix3d& V = svd.matrixV();

    lambda(2) = 0.0;

    const Eigen::Matrix3d E_21 = U * lambda.asDiagonal() * V.transpose();

    return E_21;
}

Eigen::Matrix3d FindEssentialRANSAC(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point3f>& points1
                                ,const std::vector<cv::Point3f>& points2, const int iterations, const float inlier_threshold,
                                std::vector<size_t>& inlier_idx)
{
    if(matches.size() < 8)
        return Eigen::Matrix3d::Zero();
    Eigen::Matrix3d best_essential = Eigen::Matrix3d::Zero();
    int state = 0;
    std::vector<cv::Point3f> min_point_set1, min_point_set2;
    double best_score = 0;
    vector<bool> best_inlier;
    for(int iter = 0; iter < iterations; iter++)
    {
        const vector<size_t> indices = CreateRandomArray(size_t(8), size_t(0), matches.size() - 1);
        for(const size_t& ind : indices)
        {
            min_point_set1.push_back(points1[matches[ind].queryIdx]);
            min_point_set2.push_back(points2[matches[ind].trainIdx]);
        }
        Eigen::Matrix3d E_21 = ComputeEssential(min_point_set1, min_point_set2);
        vector<bool> inlier;
        int num_inlier;
        double score = ScoreEssential(E_21, matches, points1, points2, inlier_threshold, inlier, num_inlier);
        if (score > best_score) 
        {
            best_score = score;
            best_essential = E_21;
            best_inlier = inlier;
        }
        // 如果已经过了90%的迭代次数，但还是没有找到一个很好的模型，那就不再继续了
        if(iter > 0.9 * iterations && state == 0)
            return Eigen::Matrix3d::Zero();
        // 如果某一次找到了大量的内点，就认为这个模型已经足够好了，也停止运算
        if(num_inlier > 80)
            break;
        else if(num_inlier > 24)
            state = 1;
    }

    int num_inlier = count(best_inlier.begin(), best_inlier.end(), true);
    if(num_inlier < 8 || best_score == 0)
        return Eigen::Matrix3d::Zero();
    inlier_idx.clear();
    vector<cv::Point3f> inlier_1, inlier_2;
    for(size_t i = 0; i < matches.size(); i++)
    {
        if(best_inlier[i] == false)
            continue;
        inlier_1.push_back(points1[matches[i].queryIdx]);
        inlier_2.push_back(points2[matches[i].trainIdx]);
        inlier_idx.push_back(i);
    }
    best_essential = ComputeEssential(inlier_1, inlier_2);
    
    return best_essential;
}

double ScoreEssential(const Eigen::Matrix3d& E_21, const std::vector<cv::DMatch>& matches, 
                        const std::vector<cv::Point3f>& points1, const std::vector<cv::Point3f>& points2,
                        const float inlier_threshold,
                        std::vector<bool>& is_inlier, int& num_inlier)
{
    const size_t num_matches = matches.size();
    is_inlier.resize(num_matches);
    Eigen::Matrix3d E_12 = E_21.transpose();
    double score = 0;
    num_inlier = 0;
    // outlier threshold as cosine value between a bearing vector and a normal vector of the epipolar plane
    double cos_threshold = cos( (90.0 - inlier_threshold) / 180.0 * M_PI);

    for(size_t i = 0; i < num_matches; i++)
    {
        const cv::Point3f& p1_cv = points1[matches[i].queryIdx];
        const cv::Point3f& p2_cv = points2[matches[i].trainIdx];
        Eigen::Vector3d p1(p1_cv.x, p1_cv.y, p1_cv.z);
        Eigen::Vector3d p2(p2_cv.x, p2_cv.y, p2_cv.z);
        // 把图1的点变换到图2的坐标系下，得到对极面，然后计算图2的对应点和对极面的法向量的夹角
        Eigen::Vector3d epiplane2 = E_21 * p1;
        double error = abs(epiplane2.dot(p2) / p2.norm() / epiplane2.norm());
        // 注意这里用的是cos，所以角度越大，对应的余弦越小。我们所期望的是夹角接近90度，因此如果error很大，
        // 就说明角度很小，那么就是外点
        if(error > cos_threshold)
        {
            is_inlier[i] = false;
            continue;
        }
        else 
        {
            is_inlier[i] = true;
            score += error;     // 改成 score += 1- error 会不会更好
        }

        Eigen::Vector3d epiplane1 = E_12 * p2;
        error = abs(epiplane1.dot(p1) / p1.norm()/ epiplane1.norm());
        if(error > cos_threshold)
        {
            is_inlier[i] = false;
            continue;
        }
        else 
        {
            is_inlier[i] = true;
            score += error;
            num_inlier++;
        }
    }
    return score;
}

bool DecomposeEssential(const Eigen::Matrix3d& E_21, 
                        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotations,
                        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& translations )
{
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_21, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d trans = svd.matrixU().col(2);
    trans.normalize();

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    W(0, 1) = -1;
    W(1, 0) = 1;
    W(2, 2) = 1;

    Eigen::Matrix3d rot_1 = svd.matrixU() * W * svd.matrixV().transpose();
    if (rot_1.determinant() < 0) {
        rot_1 *= -1;
    }

    Eigen::Matrix3d rot_2 = svd.matrixU() * W.transpose() * svd.matrixV().transpose();
    if (rot_2.determinant() < 0) {
        rot_2 *= -1;
    }

    rotations = {rot_1, rot_1, rot_2, rot_2};
    translations = {trans, -trans, trans, -trans};

    return true;
}

Eigen::Matrix3d FindEssentialACRANSAC(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point3f>& points1
                                ,const std::vector<cv::Point3f>& points2, const int max_iterations, const double precision,
                                ACRansac_NFA& nfa_estimator, std::vector<size_t>& inlier_idx)
                                
{
    if(matches.size() < 8)
        return Eigen::Matrix3d::Zero();

    const double max_threshold = Square(precision);
    double minNFA = std::numeric_limits<double>::infinity();
    double errorMax = std::numeric_limits<double>::infinity();
    Eigen::Matrix3d best_essential = Eigen::Matrix3d::Zero();
    int iter_reserved = max_iterations / 10;
    int iter_limit = max_iterations - iter_reserved;
    std::vector<cv::Point3f> min_point_set1, min_point_set2;
    // 初始化为所有匹配都是内点,也就是说所有点都在inlier idx的范围内
    inlier_idx.resize(matches.size());
    iota(inlier_idx.begin(), inlier_idx.end(), 0);
    nfa_estimator.SetSampleSize(matches.size());

    // AC-RANSAC 在最初是不进行的，首先要在较少的迭代次数的ransac中找到一个合适的解，然后才能进行AC-RANSAC
    // 这是因为AC-RANSAC速度比较慢，如果绝大多数的匹配都是误匹配，那么很有可能根本找不到合适的解，因此先用比较快的RANSAC
    // 进行一定次数的拟合，找到了合适的解再用AC-RANSAC进行更精确的解。
    // 注意，RANSAC找到的解和AC-RANSAC找到的解并没有关联，RANSAC只是用来证明确实能找到一个解，仅此而已
    // 如果经过了多次的ransac都没找到合适的解，那么就直接退出
    // ac_ransac_mode=true代表使用AC-RANSAC方法
    bool ac_ransac_mode = (precision == numeric_limits<double>::infinity());
    ac_ransac_mode = true;
    for(int iter = 0; iter < iter_limit && iter < max_iterations; iter++)
    {
        vector<size_t> indices;
        indices = CreateRandomArray(size_t(8), inlier_idx);
        for(const size_t& ind : indices)
        {
            min_point_set1.push_back(points1[matches[ind].queryIdx]);
            min_point_set2.push_back(points2[matches[ind].trainIdx]);
        }
        Eigen::Matrix3d E_21 = ComputeEssential(min_point_set1, min_point_set2);

        vector<double> residuals;
        for(size_t i = 0; i < matches.size(); i++)
        {
            const cv::Point3f& p1_cv = points1[matches[i].queryIdx];
            const cv::Point3f& p2_cv = points2[matches[i].trainIdx];
            Eigen::Vector3d p1(p1_cv.x, p1_cv.y, p1_cv.z);
            Eigen::Vector3d p2(p2_cv.x, p2_cv.y, p2_cv.z);
            double angle = p2.transpose() * (E_21 * p1).normalized();
            // 如果这里使用acos()，那么结果应该是越接近90度越好
            // 如果使用asin(),那么结果应该是越接近0度越好
            // 而且这里理论上应该是 Square (abs (asin(angle))) 省略了abs，因为平方之后一定是正数
            residuals.push_back(Square(asin(angle)));
        }
        if(!ac_ransac_mode)
        {
            int num_inlier = 0;
            for(const double& res : residuals)
                num_inlier += (res <= max_threshold);
            if(num_inlier > 2.5 * 8)
                ac_ransac_mode = true;
        }

        bool better = false;    // 是否找到了一个更好的解
        vector<size_t> curr_inlier_idx;     
        if(ac_ransac_mode)
        {
            pair<double, double> nfa_threshold(minNFA, 0);  // NFA - error
            pair<double, double> best_nfa(minNFA, 0.0);
            // 要注意这里传递的是一定是所有样本对应的误差，而不能仅仅是内点的误差，
            // 这是因为计算NFA后输出的inliner idx是基于所有样本的，如果仅仅传入内点误差，就会导致这个inlier idx出问题
            // 还有就是当前的inlier是A，之后的inlier可能就是B，而且B中可能还包含A中不含有的一些点，也就是说，并不是绝对的A包含B
            nfa_estimator.SetResidual(residuals);
            bool find_good_model = nfa_estimator.ComputeNFA(curr_inlier_idx, best_nfa);
            if(find_good_model && best_nfa.first < minNFA)
            {
                minNFA = best_nfa.first;
                errorMax = best_nfa.second;
                best_essential = E_21;
                better = true;
            }
        }
        // 如果经过了 20% 的迭代次数还没能从ransac里找到一个合适的解，那么就退出
        if(!ac_ransac_mode && iter > iter_reserved * 2)
            break;
        
        if(ac_ransac_mode && ((better && minNFA < 0) || (iter + 1 == iter_limit && iter_reserved > 0)))
        {
            if(curr_inlier_idx.size() == 0)
            {
                iter_limit++;
                iter_reserved--;
            }
            else 
            {
                // 一旦进行到这里了，就开始专注于优化当前的结果，也就是之后所有的点只能在这些点里选择，
                // 期望能得到一个更好的结果
                inlier_idx.swap(curr_inlier_idx);
                if(iter_reserved > 0)
                {
                    iter_limit = iter + 1 + iter_reserved;
                    iter_reserved = 0;
                }
            }
        }
    }
    if(minNFA >= 0)
        return Eigen::Matrix3d::Zero();

    return best_essential;
}


