/*
 * @author: TuDian tao
 * @Date: 2021-04-25 10:28:53
 * @LastEditTime: 2023-11-29 09:45:23
 */


#ifndef LIDAR_DATA_H
#define LIDAR_DATA_H
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <sstream> 
#include <map>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h> 
#include <pcl/io/ply_io.h> 
#include <pcl/common/transforms.h> 
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <cmath>
#include <stdio.h> 
#include <stdlib.h>
#include <stack>

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <glog/logging.h>

#include "../base/common.h"
#include "../base/Geometry.hpp"
#include "ground_segmentation.h"
#include "LidarLineExtraction.h"
#include "LidarPlaneExtraction.h"

#pragma once

typedef pcl::PointXYZI PointType;

enum FEATURE_EXTRACTION_METHOD
{
    LOAM = 1,               // LOAM的方法，抄自LEGO-LOAM
    DOUBLE_EXTRACTION = 2,  // 两次提取直线特征，以不同的参数提取，然后融合在一起。受启发自图像边缘提取算法
    ADAPTIVE = 3            // 自己琢磨出来的，效果最好
};

enum PointClassification
{
    POINT_NORMAL = 0x01,                        // 普通点，就是最初始的点
    POINT_LESS_SHARP = POINT_NORMAL << 1,       // 属于CornerLessSharp的点
    POINT_SHARP = POINT_NORMAL << 2,            // 属于CornerSharp的点
    POINT_FLAT  = POINT_NORMAL << 3,            // 属于SurfFlat的点
    POINT_GROUND = POINT_NORMAL << 4,           // 地面点
    POINT_DISABLE = POINT_NORMAL << 5,          // 由于周围点是特征点而被“压制”的点
    POINT_OCCLUDED = POINT_NORMAL << 6          // 被遮挡的点
};

class Velodyne
{
public:
    std::string name;   // 文件名称
    int N_SCANS;        // 雷达线数
    int horizon_scans;  // 水平扫描的列数


    int id;
    double scanPeriod;
    bool valid;

    pcl::PointCloud<PointType> cloud;  // 从文件读取的雷达数据    
    pcl::PointCloud<PointType> cloud_scan;   // 把雷达数据按scan排序
    pcl::PointCloud<PointType> cornerSharp;
    pcl::PointCloud<PointType> cornerLessSharp;
    pcl::PointCloud<PointType> surfFlat;
    pcl::PointCloud<PointType> surfLessFlat;

    /* 点云经过直线拟合后才有的一些参数 */
    std::vector<pcl::PointCloud<PointType>> edge_segmented;     // 每条直线对应的点云
    std::vector<std::set<int>> point_to_segment;                // cornerlessSharp里的第i个点分别在哪些segment里
    eigen_vector<Vector6d> segment_coeffs;                      // 每条直线的参数
    eigen_vector<Eigen::Vector3d> end_points;                    // 每条直线的起始点和终止点，这两个点是投影到直线上的，并非原本的点

    // 用于debug的，以后删去
    pcl::PointCloud<PointType> cornerBeforeFilter;

private:
	// 保存每个scan开始和结束的点在laserCloud中的index，而且每个scan的前5个点和后六个点都被剔除了
	std::vector<int> scanStartInd, scanEndInd;
    // 保存在计算曲率的时候每个点的邻居点的起点和终点，用于后面的边缘特征提取使用
    int* left_neighbor;
    int* right_neighbor;
	// world = 1 代表cloud_scan是在雷达世界坐标系下的点  world=2代表在相机世界坐标系下
	int world;

    Eigen::Matrix3d R_wl;
    Eigen::Vector3d t_wl;
    Eigen::Matrix4d T_wc_wl;   // 这是从雷达的世界坐标转换到相机的世界坐标

    float* cloudCurvature;      // 每个点的曲率
    
    int* cloudSortInd;
    int* cloudState;
    // 雷达点的索引到range image的行列之间的对应关系
    std::vector<std::pair<size_t, size_t> > point_idx_to_image;
    std::vector<std::vector<int> > image_to_point_idx;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> range_image;

    pcl::PointCloud<pcl::PointXYZI> removeClosedPointCloud(float threshold);

    /**
     * @description: 根据竖直方向上的角度计算每个点所在的扫描线(scan id)
     * @param vertical_angle {float&} 点的竖直方向上的角度
     * @param max_scan {int&} 最大的扫描线数
     * @return {*} 扫描线的id
     */    
    int VerticalAngleToScanID(const float& vertical_angle, const int& max_scan);

    // 雷达点变换到range image上
    void CloudToRangeImage();


    // 从LiDAR点云里分割出地面点，来自于论文 Fast Segmentation of 3D Point Cloud for Ground Vehicles - IEEE Intelligent Vehicles Symposium 2010
    // 源代码来自 https://github.com/lorenwel/linefit_ground_segmentation
    Eigen::Vector4f SegmentGround(bool compute_coeff = true);

    // 根据range image检测被遮挡的点云，从 Lego LOAM里抄来的
    void MarkOccludedPoints();
    
    /**
     * @description: 提取边缘特征，并把提取出的特征点的位置用一个图像来表示
     * @param picked_image {MatrixXf&} 用来表示提取的特征点的位置，每个像素对应一个特征点，像素值就是深度，和range image 一样
     * @param max_curvature {float} 最大的曲率，超过这个范围的曲率就视为外点
     * @param intersect_angle_threshold {float} 当前点与它所处面片的夹角，具体看livox loam里的公式4，夹角大于这个值就跳过
     * @return {*}
     */    
    void ExtractEdgeFeatures(Eigen::MatrixXf& picked_image ,float max_curvature = 5, float intersect_angle_threshold = 5);

    // 尝试另一种边缘提取的方法
    // 这种方法在小场景下的效果应该会更好一些
    void ExtractEdgeFeatures2(Eigen::MatrixXf& picked_image ,float max_curvature = 5, float intersect_angle_threshold = 5);

    // 提取平面特征
    void ExtractPlaneFeatures();

    // 提取平面特征，是和 ADAPTIVE 方法配合使用的
    void ExtractPlaneFeatures2();

    // 把边缘特征点在空间中连成一条条直线
    void EdgeToLine();
    
    // 把两次不同参数的边缘特征进行联合，得到更好的更稳定的边缘特征
    void CombineEdgeFeatures(const Eigen::MatrixXf& strict_image, const Eigen::MatrixXf& loose_image);

    // 可视化range image，用于debug
    void VisualizeRangeImage(std::string file_name, const Eigen::MatrixXf& _range_image, 
                                const float max_range = 10, const float min_range = 0);
public:

    // scan>0 代表是VLP雷达
    Velodyne(int scan, int _id, int _horizon_scan = 1800);
    Velodyne();
    ~Velodyne();
    
    // 从本地读取雷达
    void LoadLidar(std::string file_path = "");
    // 对输入的雷达点重新按照scan进行排序,只用于VLP雷达
    void ReOrderVLP();	
    // 和上面的方法效果一样,但是会更加鲁棒一些,能处理scan line冲突的情况,但是也会增加耗时 
    void ReOrderVLP2();
          
    // 对点云进行分割，并除去较小的分割块，该方法来自 Fast Range Image Segmentation - IROS 2016 
    // 从Lego LOAM里抄来的               
    int Segmentation();  

    // 从LiDAR点云里提取出地面点，并且使用RANSAC算法拟合出地面的平面
    bool ExtractGroundPointCloud(pcl::PointCloud<PointType>& ground_cloud, pcl::PointCloud<PointType>& other_cloud, Eigen::Vector4f& ground_coeff);

    bool ExtractPlanes();

    /**
     * @description: 提取特征点，
     * @param max_curvature {float} 曲率的上限，超过这值就视为不可靠，这个值最好大一些
     * @param intersect_angle_threshold {float} 雷达和局部平面夹角，小于这个视为不可靠，这个值最好小一些
     * @param method {int} 特征提取的方法
     * @param segment {bool} 是否对激光雷达点云进行分割，去除小的物体
     * @return {*}
     */    
    void ExtractFeatures(float max_curvature = 50, float intersect_angle_threshold = 5, 
                        int method = ADAPTIVE, bool segment = true);	
    /**
     * @description: 对点云进行去畸变
     * @param T_we {Matrix4d&} 点云最后一个点对应的位姿，也就是下一帧点云的起始位姿
     * @return {bool} 去畸变是否成功
     */    
    bool UndistortCloud(const Eigen::Matrix4d& T_we);
    bool UndistortCloud(const Eigen::Matrix3d& R_we, const Eigen::Vector3d& t_we);

    // 把所有变量重置为初始状态，只保留读取的雷达点，也就是 cloud
    void Reset();

    // 把点云变换到雷达世界坐标系下
    void Transform2LidarWorld(); 
    // 点云从世界坐标系变换回原本的坐标系
    void Transform2Local(); 
    // to do
    // void Transform2CameraWorld();   
    // 把单个点从雷达世界坐标系变回雷达坐标系    
    const Eigen::Vector3d World2Local(Eigen::Vector3d point_w) const;   
    // 把单个点从雷达坐标系变换到世界坐标系
    const Eigen::Vector3d Local2World(Eigen::Vector3d point_local) const;
    // 保存特征点到本地，用于debug
    const bool SaveFeatures(std::string path) const;
    // 设置雷达的名字，也就是对应的雷达点云保存的位置
    void SetName(std::string name);
    void SetPose(const Eigen::Matrix3d _R_wl, const Eigen::Vector3d _t_wl);
    void SetPose(const Eigen::Matrix4d T_wl);
    void SetRotation(const Eigen::Matrix3d _R_wl);
    void SetTranslation(const Eigen::Vector3d _t_wl);
    // 获取的是 T_wl
    const Eigen::Matrix4d GetPose() const;
    const bool IsPoseValid() const;
    const bool IsInWorldCoordinate() const;

    void test();

};



#endif