/*
 * @Author: Diantao Tu
 * @Date: 2021-12-27 12:29:09
 */

#ifndef _LIDAR_ODOMETRY_H_
#define _LIDAR_ODOMETRY_H_

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <omp.h>
#include "../sensors/Velodyne.h"
#include "../base/Config.h"
#include "../base/CostFunction.h"
#include "../util/Visualization.h"
#include "LidarFeatureAssociate.h"
#include "LidarLineMatch.h"
#include "../util/Optimization.h"

class LidarOdometry
{
private:
    std::vector<Velodyne> lidars;
    const Config& config;
    /**
     * @description: 真正对位姿进行优化的函数
     * @param cost {double&} 当前优化结果的最终残差
     * @param steps {int&} 本次优化进行了多少步
     * @param use_segment {bool} LiDAR是否经过了提前的直线分割
     * @return 优化是否成功
     */    
    bool RefinePose(double& cost, int& steps, bool use_segment);

    
public:
    LidarOdometry(const std::vector<Velodyne>& _lidars, const Config& _config);
    
    /**
     * @description: 根据已有的雷达位姿进行优化，得到更好的雷达位姿，主要就是LOAM的方法
     * @param max_iteration {int} 最多迭代优化的次数
     * @return 优化是否成功
     */    
    bool EstimatePose(const int max_iteration = 10);
    // 对雷达数据进行去畸变，要注意，这是基于lidar.cloud，所以它必须是有畸变的数据才行
    // 如果雷达已经经过畸变矫正了，可以使用LoadLidars读取最初的数据
    bool UndistortLidars(const float gap_time = 0);
    // 把某个雷达帧设置为世界坐标系的原点
    bool SetToOrigin(size_t lidar_idx);
    // 把所有雷达数据的各项变量都重置，除了 lidar.cloud
    void ResetAllLidars();
    // 从指定目录下读取雷达数据
    bool LoadLidars(const std::string path);
    /**
     * @description: 把所有雷达都融合在一起，只融合有位姿的雷达数据
     * @param skip {int} 间隔skip个雷达数据融合一个，skip=0代表融合所有雷达数据
     * @param min_range 最近的距离，低于这个距离的点舍弃
     * @param max_range 最远的距离，高于这个距离的点舍弃
     * @return {*} 融合后的雷达点云
     */    
    pcl::PointCloud<PointType> FuseLidar(int skip = 2, double min_range = 0, double max_range = 100);
    // 获取所有的雷达数据
    const std::vector<Velodyne>& GetLidarData() const;
    // 得到所有LiDAR的全局旋转 R_wc, with_invalid=true代表返回的旋转中包含没有计算位姿的雷达数据
    eigen_vector<Eigen::Matrix3d> GetGlobalRotation(bool with_invalid=false);
    // 得到所有LiDAR的全局旋转 t_wc, with_invalid=true代表返回的平移中包含没有计算位姿的雷达数据
    eigen_vector<Eigen::Vector3d> GetGlobalTranslation(bool with_invalid=false);
    // 得到所有LiDAR的名字，with_invalid=true代表返回的名字中包含没有计算位姿的雷达数据
    std::vector<std::string> GetLidarNames(bool with_invalid=false);
    
    // 用来debug的，用于输出指定的雷达对之间的特征匹配关系，检查有没有误匹配
    bool test(const std::vector<std::pair<int, int>>& lidar_pairs);

    ~LidarOdometry();
};




#endif