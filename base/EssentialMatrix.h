/*
 * @Author: Diantao Tu
 * @Date: 2021-11-22 19:33:39
 */

#ifndef _ESSENTIAL_MATRIX_H_
#define _ESSENTIAL_MATRIX_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include "ACRansac_NFA.h"
#include "common.h"
#include "Random.hpp"
#include "Math.h"

/**
 * @description: 根据两组对应的三维点计算本质矩阵
 * @param points1 第一组三维点
 * @param points2 第二组三维点
 * @return {Eigen::Matrix3d} 本质矩阵
 */
Eigen::Matrix3d ComputeEssential(const std::vector<cv::Point3f>& points1, const std::vector<cv::Point3f>& points2);

/**
 * @description: 使用RANSAC方法计算两个图像之间的本质矩阵
 * @param matches 三维点之间的匹配关系
 * @param points1 第一张图像对应的三维点
 * @param points2 第二张图像对应的三维点
 * @param iterations RANSAC迭代的次数
 * @param inlier_threshold 内点对应角度阈值
 * @param inlier_idx 指示matches中inlier的索引
 * @return {Eigen::Matrix3d} 本质矩阵，如果计算出现问题则返回全0
 */
Eigen::Matrix3d FindEssentialRANSAC(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point3f>& points1
                                ,const std::vector<cv::Point3f>& points2, const int iterations, 
                                const float inlier_threshold, std::vector<size_t>& inlier_idx);

/**
 * @description: 对本质矩阵进行评分，也就是检查有多少个inlier
 * @param E_21 本质矩阵
 * @param matches 三维点之间的匹配关系
 * @param points1 第一张图像对应的三维点
 * @param points2 第二张图像对应的三维点
 * @param is_inlier 某个match是否为inlier
 * @param num_inlier 内点个数
 * @return 评分
 */
double ScoreEssential(const Eigen::Matrix3d& E_21, const std::vector<cv::DMatch>& matches, 
                        const std::vector<cv::Point3f>& points1, const std::vector<cv::Point3f>& points2,
                        const float inlier_threshold,
                        std::vector<bool>& is_inlier, int& num_inlier);

/**
 * @description: 使用ACRANSAC方法计算本质矩阵，adaptive Structure From Motion with a contrario model estimation - ACCV 2012
 ** @param matches 三维点之间的匹配关系
 * @param points1 第一张图像对应的三维点
 * @param points2 第二张图像对应的三维点
 * @param iterations RANSAC迭代的次数
 * @param precision 2.5*2.5  
 * @param inlier_idx matches中inlier的索引
 * @return {*}
 */
Eigen::Matrix3d FindEssentialACRANSAC(const std::vector<cv::DMatch>& matches, const std::vector<cv::Point3f>& points1
                                ,const std::vector<cv::Point3f>& points2, const int iterations, const double precision,
                                ACRansac_NFA& nfa_estimator, std::vector<size_t>& inlier_idx);


bool DecomposeEssential(const Eigen::Matrix3d& E_21, 
                        std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& rotations,
                        std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& translations );

// 根据可能的位姿和匹配关系找到最好的位姿
bool BestPose(const eigen_vector<Eigen::Matrix3d>& rotations, const eigen_vector<Eigen::Vector3d>& translations,
                const std::vector<cv::DMatch>& matches, std::vector<bool>& is_inlier,
                const std::vector<cv::Point3f>& points1, const std::vector<cv::Point3f>& points2,
                Eigen::Matrix3d& best_rotation, Eigen::Vector3d& best_trans);

int CheckRT(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, const std::vector<bool>& is_inlier, 
            const bool depth_is_posivite, eigen_vector<Eigen::Vector3d>& triangulated_points, 
            std::vector<bool> is_triangulated, float& parallax);
#endif