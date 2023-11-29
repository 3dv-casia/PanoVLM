/*
 * @Author: Diantao Tu
 * @Date: 2021-11-26 16:55:05
 */

#ifndef _DEPTH_COMPLETION_H_
#define _DEPTH_COMPLETION_H_

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <glog/logging.h>

// CGAL: depth-map initialization
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include "Visualization.h"

// 自己实现的 numpy.where, 返回一个矩阵，对应位置是0或1，其中0就是不满足条件的，1是满足条件的
// 生成的mask类型和输入矩阵的类型一致，这样方便之后进行矩阵的逐元素乘法
cv::Mat Where(const cv::Mat& src, std::string condition, float threshold);

/**
 * @description: 找到矩阵最大值的索引，仿照 numpy.argmax
 * @param src 数据矩阵
 * @param axis 方向，axis=0代表找到每一列的最大值，axis=1代表找到每一行的最大值
 * @param max_value 在遍历每一行(列)的过程中，如果找到了target_value对应的值，就立刻终止当前行(列),开始下一行(列)
 * @return 最大值的索引
 */
std::vector<int> ArgMax(const cv::Mat& src, int axis, float target_value = FLT_MAX);

/**
 * @description: 深度图补全算法，
 * In Defense of Classical Image Processing: Fast Depth Completion on the CPU - Conference on Computer and Robot Vision 2018
 * https://github.com/kujason/ip_basic
 * @param sparse_depth {Mat&} 输入的稀疏深度图
 * @param max_depth {float} 最大深度
 * @return {*} 补全后的深度图
 */  
cv::Mat DepthCompletion(const cv::Mat& sparse_depth, const float max_depth);


/**
 * @description: 深度图补全算法，具体做法是把激光雷达点云进行 Delaunay 三角化，然后投影到图像平面
 * 得到每个像素的深度值。该方法受启发自 RPV-SLAM: Range-augumented Panoramic Visual SLAM for Mobile Mapping 
 * Systems with Panoramic Camera and Tilted LiDAR - ICRA 2021
 * 实测下来效果并不好, 故没有使用, 而且速度比较慢。这是整个代码中唯一使用CGAL的部分, 删除函数即可不依赖CGAL
 * @param rows 图像行数
 * @param cols 图像列数
 * @param cloud 点云
 * @param T_cl 激光雷达到相机的变换
 * @return 补全后的深度图
*/
cv::Mat DepthCompletionDelaunay(const int& rows, const int& cols, const pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Matrix4d& T_cl);

cv::Mat GenerateLidarMask(const int& rows, const int& cols, const pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Matrix4d& T_cl);

#endif