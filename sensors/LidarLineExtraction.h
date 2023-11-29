/*
 * @Author: Diantao Tu
 * @Date: 2022-08-25 10:47:32
 */

#ifndef _LIDAR_LINE_EXTRACTION_H_
#define _LIDAR_LINE_EXTRACTION_H_

#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include "../base/common.h"
#include "../base/Geometry.hpp"
#include "../base/Math.h"

using namespace std;

/**
 * @description: 向外扩张直线，也就是以start_point为中心，把已有直线向外扩张
 * @param kd_tree 用于搜索近邻的三维点
 * @param cloud 所有的三维点的点云
 * @param start_point 扩张的起始点，这个点应该是已有直线的端点
 * @param line_points_idx 构成已有直线的点的索引，这个索引是基于cloud的
 * @return 是否成功向外扩张了直线
 */
bool ExpandLine(const pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree, const pcl::PointCloud<pcl::PointXYZI>& cloud, const pcl::PointXYZI& start_point, 
                set<int>& line_points_idx);

/**
 * @description: 查找当前直线的近邻直线，注意这个近邻是可以传播的，也就是说当前直线和A近邻，A和B近邻，那么当前直线也和B近邻
 * @param fused 用于指示直线是否已经被其他的直线设为了近邻，设为近邻之后就要融合了，所以叫 fused
 * @param line_idx {size_t} 当前直线id
 * @param neighbor_idx 所有图像的近邻
 * @return 与当前直线近邻的直线id
 */
std::vector<int> FindNeighbors(vector<bool>& fused, const std::vector<std::vector<int>>& neighbor_idx, size_t line_idx);

/**
 * @description: 把多个直线融合为一条直线
 * @param segments 所有的直线对应的点云
 * @param line_coeffs 所有的直线的参数
 * @param neighbors 要融合的直线的id
 * @param optimize 融合之后是否使用RANSAC再进行一次优化，剔除外点
 * @param cloud_fused 融合之后的直线的点云
 * @param line_coeff 融合之后的直线的参数
 */
void FuseLines(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segments, 
                        const eigen_vector<Vector6d>& line_coeffs,
                        const vector<int>& neighbors, bool optimize,
                        pcl::PointCloud<pcl::PointXYZI>& cloud_fused, Vector6d& line_coeff);

/**
 * @description: 融合距离很近的直线段
 * @param points_each_line 每个直线段对应的点云
 * @param line_coeffs 每条直线参数
 * @return {*}
 */
void FuseLineSegments(std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs);

// 根据长度对直线进行过滤，太短的直线就滤掉
void FilterLineByLength(std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs);

// 根据直线跨越的扫描线数进行过滤，如果直线里有很多点但却都集中在两三条扫描线上，那么也过滤掉
void FilterLineByScan(const std::vector<std::pair<size_t, size_t> >& point_idx_to_image , 
                std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs);

/**
 * @description: 从点云中提取直线
 * @param edge_points 点云
 * @param point_idx_to_image 点云里各个点到range image的映射，这是用来判定每个点所属的扫描线（scan line）
 * @param points_each_line 提取出的直线包含的点
 * @param line_coeffs 提取出的直线的参数
 * @return {*}
 */
void ExtractLineFeatures(const pcl::PointCloud<pcl::PointXYZI>& edge_points, const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
    std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs);
                        


#endif