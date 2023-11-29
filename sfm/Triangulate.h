/*
 * @Author: Diantao Tu
 * @Date: 2021-12-15 09:16:11
 */

#ifndef _TRIANGULATE_H_
#define _TRIANGULATE_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>
#include "../base/common.h"

/**
 * @description: 对特征点进行两视图三角化
 * @param R_21 旋转矩阵，从图像1到图像2
 * @param t_21 平移向量，从图像1到图像2
 * @param p1 特征点在图像1下的坐标，这里是三维的，假设在单位球上
 * @param p2 特征点在图像2下的坐标，这里是三维的，假设在单位球上
 * @return 三角化后的三维点
 */  
Eigen::Vector3d Triangulate2View(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2);

// 这个不太准，不知道为啥, 所以不要用
Eigen::Vector3d Triangulate2View_2(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2);

/**
 * @description: 对特征点进行两视图三角化,使用的方法是 Inverse Depth Weighted Midpoint 
 *              Triangulation: Why optimize? - BMVC 2019
 * @param R_21 旋转矩阵，从图像1到图像2
 * @param t_21 平移向量，从图像1到图像2
 * @param p1 特征点在图像1下的坐标，这里是三维的，假设在单位球上
 * @param p2 特征点在图像2下的坐标，这里是三维的，假设在单位球上
 * @return 三角化后的三维点,如果三角化失败则返回(inf, inf, inf)
 */
Eigen::Vector3d Triangulate2ViewIDWM(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21,
            const cv::Point3f& p1, const cv::Point3f& p2);

// 最小化代数误差的三角化过程，从openMVG里抄来的，
// 当特征点进行了各向同性的正则化(isotropic normalization)之后，三角化的效果会更好
// 经过测试，效果确实好了一点
Eigen::Vector3d TriangulateNViewAlgebraic(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points);

Eigen::Vector3d TriangulateNView1(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points);

Eigen::Vector3d TriangulateNView2(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points);
/**
 * @description: 多张图像进行三角化
 * @param R_cw_list 全局的旋转
 * @param t_cw_list 全局的平移
 * @param points 在每个相机自己的坐标系下，特征点在单位球上的坐标
 * @return 三角化后的空间点坐标
 */  
Eigen::Vector3d TriangulateNView(const eigen_vector<Eigen::Matrix3d>& R_cw_list, 
                                    const eigen_vector<Eigen::Vector3d>& t_cw_list,
                                    const std::vector<cv::Point3f>& points);

#endif