/*
 * @Author: Diantao Tu
 * @Date: 2022-08-25 20:20:18
 */
#ifndef _ROTATION_AVERAGING_H_
#define _ROTATION_AVERAGING_H_

#include "../base/common.h"
#include <Eigen/Sparse>
#include "../util/MatchPair.h"
#include "../base/CostFunction.h"
#include "l1_solver_admm.hpp"
#include "PoseGraph.h"

using namespace std;

// 使用线性方程组求解一个初始旋转，方程组为 weight * (R_jw - R_ji * R_iw) = 0，其实就是一个最小二乘解
// 从openMVG抄的
bool RotationAveragingLeastSquare(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations);

// 使用最大生成树得到一个初始的全局旋转
bool RotationAveragingSpanningTree(const vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations, size_t start_idx);

// 基于L2模的旋转平均的全局优化，也是从openMVG抄的
bool RotationAveragingL2(int num_threads , const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations);

// 基于L1模的旋转平均，从openMVG抄的
// Efficient and Robust Large-Scale Rotation Averaging - ICCV 2013
// Robust Relative Rotation Averaging - TPAMI 2018
// 整体思想是根据pose graph得到一颗最大生成树，把生成树的根认为是世界坐标系，旋转设置为单位阵，
// 依次根据生成树上的关系算出其他各个节点的全局旋转, 然后计算全局旋转在L1模下的误差并使用IRLS方法最小化误差
// 之后会对图像对进行过滤，如果图像对的相对旋转和计算得到的绝对旋转之间的差异太大，就过滤掉，
// angle threshold就是过滤的角度阈值，大于这个角度就过滤掉
// 注意：这一步会对image pair进行过滤，可能会导致image pair变少
bool RotationAveragingL1(std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Matrix3d>& global_rotations, 
                            size_t start_idx, double angle_threshold);

// 使用IRLS方法对计算得到的全局旋转进行近一步优化
// weight_function = 1 => 使用L-1/2 范数作为权重
// weight_function = 2 => 使用 Geman-McClure 作为权重
bool RotationAveragingRefineL1(eigen_vector<Eigen::Matrix3d>& global_rotations, size_t start_idx,
                                const eigen_map<pair<size_t, size_t>, Eigen::Matrix3d>& relative_rotations,
                                const int weight_function);


#endif