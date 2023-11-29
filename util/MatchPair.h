/*
 * @Author: Diantao Tu
 * @Date: 2022-01-05 16:03:57
 */

#ifndef _MATCH_PAIR_H_
#define _MATCH_PAIR_H_
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "../base/common.h"
#include "../base/Serialization.h"

struct MatchPair{
    std::pair<size_t, size_t> image_pair;   // 图像对的id，小的id在前，大的id在后
    std::vector<cv::DMatch> matches;        // 二维特征点的匹配
    std::vector<size_t> inlier_idx;         // 在计算本质矩阵的时候所有内点的索引，以及三角化时使用的内点，这个索引是基于matches的
    Eigen::Matrix3d R_21;                   // 相对旋转 
    Eigen::Vector3d t_21;                   // 相对平移
    Eigen::Matrix3d E_21;                   // 本质矩阵
    eigen_vector<Eigen::Vector3d> triangulated;     // 三角化的点的坐标，在图像1的坐标系下
    int points_with_depth;                  // 三角化后有真实深度的三维点数量，真实深度是指在深度图上有深度
    double upper_scale;                     // 图像对的相对平移的尺度的上限, 当这个值为0的时候表示当前的尺度不是特别可靠
    double lower_scale;                     // 图像对的相对平移的尺度的下限, 当这个值为0的时候表示当前的尺度不是特别可靠
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MatchPair(size_t i, size_t j, const std::vector<cv::DMatch>& _matches):matches(_matches),upper_scale(-1),lower_scale(-1)
    {
        image_pair = std::pair<size_t, size_t>(i,j);
        R_21 = Eigen::Matrix3d::Zero();
        t_21 = Eigen::Vector3d::Zero();
        E_21 = Eigen::Matrix3d::Zero();
        points_with_depth = 0;
    }
    MatchPair(size_t i, size_t j):upper_scale(-1),lower_scale(-1)
    {
        image_pair = std::pair<size_t, size_t>(i,j);
        matches = std::vector<cv::DMatch>();
        R_21 = Eigen::Matrix3d::Zero();
        t_21 = Eigen::Vector3d::Zero();
        E_21 = Eigen::Matrix3d::Zero();
        points_with_depth = 0;
    }
    MatchPair():upper_scale(-1),lower_scale(-1)
    {
        image_pair = std::pair<size_t, size_t>(0,0);
        matches = std::vector<cv::DMatch>();
        R_21 = Eigen::Matrix3d::Zero();
        t_21 = Eigen::Vector3d::Zero();
        E_21 = Eigen::Matrix3d::Zero();
        points_with_depth = 0;
    }
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & image_pair.first;
        ar & image_pair.second;
        ar & matches;
        ar & inlier_idx;
        ar & E_21;
        ar & R_21;
        ar & t_21;
        ar & triangulated;
        ar & points_with_depth;
        ar & upper_scale;
        ar & lower_scale;
    }
};





#endif