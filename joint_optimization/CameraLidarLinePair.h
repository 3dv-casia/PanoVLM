/*
 * @Author: Diantao Tu
 * @Date: 2022-01-18 18:57:35
 */

#ifndef _CAMERA_LIDAR_LINE_PAIR_H_
#define _CAMERA_LIDAR_LINE_PAIR_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 图像直线和雷达直线的匹配
struct CameraLidarLinePair{
    size_t image_id;            // 当前图像的id
    size_t lidar_id;            // 当前LiDAR的id
    cv::Vec4f image_line;       // 图像直线的起点和终点
    Eigen::Vector3d lidar_line_start;   // 雷达直线的起点，雷达坐标系下
    Eigen::Vector3d lidar_line_end;     // 雷达直线的终点,雷达坐标系下
    int image_line_id;              // 当前的图像直线是原图的第几条直线，用来debug
    int lidar_line_id;              // 当前的LiDAR直线是原本LiDAR的第几个segment，用来debug
    float angle;                    // 把雷达直线变换到相机坐标系下后,雷达直线和图像直线的夹角
    float weight;                   // 权重
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraLidarLinePair():image_line(cv::Vec4f(0,0,0,0)), lidar_line_start(Eigen::Vector3d(0,0,0)), lidar_line_end(Eigen::Vector3d(0,0,0)),
                image_line_id(-1), lidar_line_id(-1), angle(-1), weight(1)
    {}
};



#endif