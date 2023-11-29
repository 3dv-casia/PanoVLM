/*
 * @Author: Diantao Tu
 * @Date: 2022-03-30 17:57:34
 */

#ifndef _LIDAR_FEATURE_ASSOCIATE_
#define _LIDAR_FEATURE_ASSOCIATE_

#include <vector>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include "../base/common.h"
#include "../base/Geometry.hpp"
#include "../sensors/Velodyne.h"

// 存储点到直线的匹配关系
struct Point2Line
{
public:
    Eigen::Vector3d point;                          // neighbor 坐标系下的点
    Eigen::Vector3d line_point1, line_point2;       // reference坐标系下直线上的两个点
    Point2Line(const Eigen::Vector3d& _point, const Eigen::Vector3d& _p1, const Eigen::Vector3d& _p2):
        point(_point), line_point1(_p1), line_point2(_p2)
    {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 存储线到线的匹配关系
struct Line2Line
{
public:
    int neighbor_line_idx;                      // neighbor 里直线的索引
    int ref_line_idx;                           // reference 里直线的索引
    Eigen::Vector3d line_point1, line_point2;   // reference 对应的直线经过的两个点
    Line2Line(const int& _neighbor_line_idx, const int& _ref_line_idx, const Eigen::Vector3d& _p1, const Eigen::Vector3d& _p2):
        neighbor_line_idx(_neighbor_line_idx), ref_line_idx(_ref_line_idx),line_point1(_p1), line_point2(_p2)
    {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// 存储点到面的匹配关系
struct Point2Plane
{
public:
    Eigen::Vector3d point;                      // neighbor 坐标系下的点
    Eigen::Vector4d plane_coeff;                // reference 坐标系下的平面参数 a b c d, 平面法向量是经过归一化的
    Point2Plane(const Eigen::Vector3d& _point, const Eigen::Vector4d& _plane_coeff):
        point(_point), plane_coeff(_plane_coeff)
    {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @description: 找到每个雷达的近邻雷达，对于有位姿的雷达，使用K近邻，对于没有位姿的雷达，就按照拍摄顺序来
 * @param lidars 输入的雷达
 * @param neighbor_size 近邻的数量，这个数字不是绝对的，一般实际上近邻数量会额外增加1或2
 * @return 每个雷达的近邻的索引
 */ 
std::vector<std::vector<int>> FindNeighbors(const std::vector<Velodyne>& lidars , const int neighbor_size);

std::vector<std::vector<int>> FindNeighborsConsecutive(const std::vector<Velodyne>& lidars , const int neighbor_size);


/**
 * @description: 对两个雷达进行点到线的特征匹配，要求两个雷达在同一个坐标系下（世界坐标系）,匹配的方法是对每个nei_lidar的点找到
 *          它在ref_lidar里的k个近邻点，要求这k个点要呈直线型排列
 * @param ref_lidar {Velodyne&} 线所在的雷达
 * @param nei_lidar {Velodyne&} 点所在的雷达
 * @param dist_threshold {float} 近邻搜索的距离阈值
 * @param visualization {bool} 是否输出匹配的点云，只用于debug
 * @return 点到线的匹配关系
 */
std::vector<Point2Line> AssociatePoint2Line(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        const float dist_threshold = 0.7, bool visualization = false);

// 和上面一样，区别在于这里要求两个雷达已经进行过分割了， 也就是已经提前知道哪些点是同一条直线，因此在寻找k近邻的时候，要求这
// k个近邻点都要属于同一条直线才行
std::vector<Point2Line> AssociatePoint2LineSegmentKNN(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        const float dist_threshold = 0.7, bool visualization = false);

/**
 * @description: 对两个雷达进行点到线的特征匹配，和上面的区别在于匹配方法。这里用的方法是直接计算nei_lidar到ref_lidar里各条直线的距离，
 *          然后用最近距离来当做当前点到直线的匹配关系
 * @param ref_lidar {Velodyne&} 线所在的雷达
 * @param nei_lidar {Velodyne&} 点所在的雷达
 * @param dist_threshold {float} 点到直线的距离阈值，超出这个阈值就认为点和直线不匹配
 * @param visualization {bool} 是否输出匹配的点云，只用于debug
 * @return 点到线的匹配关系
 */
std::vector<Point2Line> AssociatePoint2LineSegment(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        const float dist_threshold = 0.7, bool visualization = false);

/**
 * @description: 对两个雷达进行线到线的特征匹配，要求两个雷达在同一个坐标系下（世界坐标系）
 * @param ref_lidar {Velodyne&} 参考雷达
 * @param nei_lidar {Velodyne&} 近邻雷达，匹配是在近邻雷达搜索，然后在参考雷达里找到匹配关系
 * @param dist_threshold {float} 近邻搜索的距离阈值
 * @param visualization {bool} 是否输出匹配的点云，只用于debug
 * @return 线到线的匹配关系
 */
std::vector<Line2Line> AssociateLine2LineKNN(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        const float dist_threshold = 0.7, bool visualization = false);

/**
 * @description: 对两个雷达进行线到线的特征匹配，要求两个雷达在同一个坐标系下（世界坐标系）。和上面方法的区别就像点到直线匹配的两种
 *          方法的区别一样。这里直线到直线的匹配关系是依靠直线上每个点到另一条直线的距离来判断的，如果直线A上大多数点都和另一条直线B
 *          距离最近，那么就认为A和B匹配
 * @param ref_lidar {Velodyne&} 参考雷达
 * @param nei_lidar {Velodyne&} 近邻雷达，匹配是在近邻雷达搜索，然后在参考雷达里找到匹配关系
 * @param dist_threshold {float} 点到直线的距离阈值
 * @param visualization {bool} 是否输出匹配的点云，只用于debug
 * @return 线到线的匹配关系
 */
std::vector<Line2Line> AssociateLine2Line(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        const float dist_threshold = 0.7, bool visualization = false);


// 进行面到面的匹配
std::vector<Point2Plane> AssociatePoint2Plane(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                                        double plane_tolerance, const float dist_threshold = 0.7, bool visualization = false);

#endif