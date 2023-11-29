/*
 * @Author: Diantao Tu
 * @Date: 2022-07-15 13:22:10
 */

#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

#include <glog/logging.h>

#include "../base/common.h"
#include "../base/CostFunction.h"
#include "../sensors/Velodyne.h"
#include "../sensors/Frame.h"
#include "../sensors/Equirectangular.h"
#include "../joint_optimization/CameraLidarLinePair.h"
#include "../lidar_mapping/LidarFeatureAssociate.h"

#include "Tracks.h"
#include "MatchPair.h"
enum RESIDUAL_TYPE
{
    ANGLE_RESIDUAL_1 = 1,       // 以角度为单位的误差，一个角度，也就是两条射线的夹角
    ANGLE_RESIDUAL_2 = 2,       // 以角度为单位的误差，两个角度，分别是 longitude 和 latitude
    PIXEL_RESIDUAL = 3          // 以像素距离为单位的误差
};

/**
 * @description: 进行SfM的全局BA，优化相机位姿和三维点坐标
 * @param frames 所有图像信息
 * @param tracks 三维点的信息，也就是 structure
 * @param residual_type 残差类型，角度残差或者是像素残差
 * @param num_threads 线程数
 * @param refine_structure 是否优化三维点位置
 * @param refine_rotation 是否优化相机的旋转
 * @param refine_translation 是否优化相机的平移
 * @return BA是否成功
 */
bool SfMGlobalBA(std::vector<Frame>& frames, std::vector<PointTrack>& tracks, int residual_type, int num_threads,
                    bool refine_structure=true, bool refine_rotation=true, bool refine_translation=true);

/**
 * @description: 进行SfM的局部BA，优化两张图像之间的相对位姿
 * @param frame1 第一张图像
 * @param frame2 第二张图像
 * @param residual_type 残差类型，角度残差或者是像素残差
 * @param image_pair 图像对的相关信息，包括相对位姿、三角化的空间点
 * @return BA是否成功
 */
bool SfMLocalBA(const Frame& frame1, const Frame& frame2, int residual_type, MatchPair& image_pair);

/**
 * @description: 向problem中添加camera-camera的残差，也就是三维点的投影误差
 * @param frames 图像信息
 * @param angleAxis_cw_list 从世界到相机的旋转，用轴角表示
 * @param t_cw_list 从世界到相机的平移，用平移向量表示
 * @param structure 三角化的空间点
 * @param problem ceres的求解问题
 * @param residual_type 残差类型，可以是角度残差，也可以是像素残差
 * @param weight 权重
 * @return 相关的残差数量
 */    
size_t AddCameraResidual(const std::vector<Frame>& frames, eigen_vector<Eigen::Vector3d>& angleAxis_cw_list, 
                    eigen_vector<Eigen::Vector3d>& t_cw_list, std::vector<PointTrack>& structure,
                     ceres::Problem& problem, int residual_type, double weight = 1.0);

/**
 * @description: 向problem中添加点到直线的雷达残差
 * @param neighbors 每个雷达的近邻雷达id
 * @param lidars 所有雷达数据
 * @param angleAxis_lw_list 从世界到雷达的旋转，用轴角表示
 * @param t_lw_list 从世界到雷达的平移，用平移向量表示
 * @param problem ceres的求解问题
 * @param point_to_line_dis_threshold 点到直线距离阈值，超过这个值就认为没法形成点到直线关联
 * @param use_segment 雷达是否已经经过了直线分割
 * @param angle_residual 是否使用基于角度的残差表示
 * @param normalized_distance 如果使用基于角度的残差，是否要使用归一化距离
 * @param weight 权重
 * @return 相关的残差数量
 */                        
size_t AddLidarPointToLineResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_line_dis_threshold, bool use_segment,
                        bool angle_residual = false,
                        bool normalized_distance = true, double weight = 1.0 );

/**
 * @description: 向problem中添加直线到直线的雷达残差
 * @param neighbors 每个雷达的近邻雷达id
 * @param lidars 所有雷达数据
 * @param angleAxis_lw_list 从世界到雷达的旋转，用轴角表示
 * @param t_lw_list 从世界到雷达的平移，用平移向量表示
 * @param problem ceres的求解问题
 * @param point_to_line_dis_threshold 点到直线距离阈值，超过这个值就认为没法形成点到直线关联
 * @param angle_residual 是否使用基于角度的残差表示
 * @param normalized_distance 如果使用基于角度的残差，是否要使用归一化距离
 * @param weight 权重
 * @return 相关的残差数量
 */  
size_t AddLidarLineToLineResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_line_dis_threshold, bool angle_residual = false,
                        bool normalized_distance = true, double weight = 1.0);

size_t AddLidarLineToLineResidual2(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        const vector<LineTrack>& lidar_line_tracks, 
                        double point_to_line_dis_threshold, bool angle_residual = false,
                        bool normalized_distance = true, double weight = 1.0);

/**
 * @description: 向problem中添加点到平面的雷达残差
 * @param neighbors 每个雷达的近邻雷达id
 * @param lidars 所有雷达数据
 * @param angleAxis_lw_list 从世界到雷达的旋转，用轴角表示
 * @param t_lw_list 从世界到雷达的平移，用平移向量表示
 * @param problem ceres的求解问题
 * @param point_to_plane_dis_threshold 点到平面距离阈值，超过这个值就认为没法形成点到平面关联
 * @param plane_tolerance 平面的不平整程度，超过这个值就认为近邻无法形成平面，也就没有点到平面关联
 * @param angle_residual 是否使用基于角度的残差表示
 * @param normalized_distance 如果使用基于角度的残差，是否要使用归一化距离
 * @param weight 权重
 * @return 相关的残差数量
 */  
size_t AddLidarPointToPlaneResidual(const vector<vector<int>>& neighbors, const std::vector<Velodyne>& lidars,
                        eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, 
                        eigen_vector<Eigen::Vector3d>& t_lw_list, ceres::Problem& problem, 
                        double point_to_plane_dis_threshold, double plane_tolerance,
                        bool angle_residual = false,
                        bool normalized_distance = true, double weight = 1.0);
                        
                     
/**
 * @description: 向problem中添加camera-LiDAR的残差，两条直线的平面夹角，单位是度
 * @param frames 所有图像信息
 * @param lidars 所有雷达信息
 * @param angleAxis_cw_list 从世界到相机的旋转，用轴角表示
 * @param t_cw_list 从世界到相机的平移，用平移向量表示
 * @param angleAxis_lw_list 从世界到雷达的旋转，用轴角表示
 * @param t_lw_list 从世界到雷达的平移，用平移向量表示
 * @param line_pairs 匹配的直线对
 * @param loss_function 用于对外点鲁棒的损失函数
 * @param problem ceres的求解问题
 * @param weight 权重
 * @return 相关的残差数量
 */ 
size_t AddCameraLidarResidual(const std::vector<Frame>& frames, const std::vector<Velodyne>& lidars,
                    eigen_vector<Eigen::Vector3d>& angleAxis_cw_list, eigen_vector<Eigen::Vector3d>& t_cw_list,
                    eigen_vector<Eigen::Vector3d>& angleAxis_lw_list, eigen_vector<Eigen::Vector3d>& t_lw_list,
                    const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, 
                    ceres::LossFunction* loss_function, ceres::Problem& problem, double weight = 1.0);

ceres::Solver::Options SetOptionsSfM(const int num_threads);

ceres::Solver::Options SetOptionsLidar(const int num_threads, const int lidar_size);

#endif