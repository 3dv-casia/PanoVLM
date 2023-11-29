/*
 * @Author: Diantao Tu
 * @Date: 2022-08-26 09:48:59
 */
#ifndef _TRANSLATION_AVERAGING_H_
#define _TRANSLATION_AVERAGING_H_

#include "../base/common.h"
#include "../util/MatchPair.h"
#include "../sensors/Frame.h"
#include "../base/CostFunction.h"
#include "PoseGraph.h"
#include "LinearProgramming.h"
#include "BATA.h"
#include <Eigen/Sparse>
#include <glog/logging.h>

bool InitGlobalTranslationGPS(const std::vector<Frame>& frames, eigen_vector<Eigen::Vector3d>& global_translations,
                            const map<size_t, size_t>& new_to_old);

/**
 * @description: 使用直接线性变换(Direct Linear Transform)方法求解全局平移 t_jw - R_ji*t_iw - t_ji=0, 这里的t_ji是有绝对尺度的
 * @param pairs 具有绝对尺度的图像对，要求图像对里相机的id一定要是从0到n的连续数
 * @param global_translation 求解得到的全局平移
 * @return DLT是否成功
 */ 
bool TranslationAveragingDLT(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations);

/**
 * @description: 最小化目标函数 t_jw - R_ji * t_iw - scale * t_ji 的L2范数来求解全局平移
 * @param num_cameras 要求解的相机数量
 * @param global_translations 求解得到的全局平移
 * @param scales 每个图像对的尺度（会被改变）
 * @param origin_idx 固定平移不动的相机的索引
 * @param weight 每个图像对的权重（会被改变）
 * @param costs 本次最终的误差
 * @param loss_function 损失函数
 * @return 是否成功
 */  
bool TranslationAveragingL2(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, vector<double>& scales,
                                size_t origin_idx, vector<double>& weight, double& costs ,ceres::LossFunction* loss_function,
                                const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads);

// 根据相对平移计算绝对平移，目标函数是最小化 t_2w - R_21 * t_1w - scale * t_21 的 L2模,使用了softL1作为损失函数
bool TranslationAveragingSoftL1(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations,
                                    size_t origin_idx, double l1_loss_threshold,
                                    const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads);

// 根据相对平移和绝对旋转计算绝对平移，使用的方法是 Robust Global Translations with 1DSfM - ECCV 2014
bool TranslationAveragingL2Chordal(const std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames,
                                    eigen_vector<Eigen::Vector3d>& global_translations,
                                    const map<size_t, size_t>& new_to_old, size_t origin_idx, double l2_loss_threshold,
                                    const int num_threads);

// 使用线性规划的方法计算绝对平移，使用的方法是 Global Fusion of Relative Motion - ICCV 2013 
// 对应其中的公式(9),最小化了 t_2w - R_21 * t_1w - scale * t_21 的无穷范数
bool TranslationAveragingL1(const std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, size_t origin_idx,
                                const map<size_t, size_t>& new_to_old );

// 最小化 t_2w - R_21 * t_1w - scale * t_21 的L2模，使用了一种迭代的加权最小二乘法，来自于
// Robust Camera Location Estimation by Convex Programming - CVPR 2015
// 原论文中作者使用了QP（二次规划）来求解，我这里使用了ceres
bool TranslationAveragingL2IRLS(std::vector<MatchPair>& image_pairs, eigen_vector<Eigen::Vector3d>& global_translations, size_t origin_idx,
                                const int num_iteration, const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads);

// 根据绝对旋转和相对平移计算绝对平移，使用的方法是 Baseline Desensitizing In Translation Averaging - CVPR 2018
bool TranslationAveragingBATA(const std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames, 
                            eigen_vector<Eigen::Vector3d>& global_translations, 
                            const map<size_t, size_t>& new_to_old, size_t origin_idx,
                            const std::string& output_path);

// 平移平均方法，按照论文实现的结果
// Robust Camera Location Estimation by Convex Programming - CVPR 2015
// 效果非常不好，大概率是我实现的有问题，不过也不用它了
// BATA中也有LUD的实现，效果更好，速度更快，用那个即可
bool TranslationAveragingLUD(std::vector<MatchPair>& image_pairs, const std::vector<Frame>& frames,
                            eigen_vector<Eigen::Vector3d>& global_translations, 
                            const map<size_t, size_t>& new_to_old, size_t origin_idx,
                            const int num_iteration, const float upper_scale_ratio, const float lower_scale_ratio, const int num_threads);

#endif