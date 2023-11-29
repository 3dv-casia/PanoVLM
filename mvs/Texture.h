/*
 * @Author: Diantao Tu
 * @Date: 2022-03-16 14:14:41
 */

#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include <omp.h>

#include "../base/Config.h"
#include "../base/common.h"

#include "../sensors/Frame.h"
#include "../sensors/Velodyne.h"
#include "../sensors/Equirectangular.h"


class Texture
{
private:
    std::vector<Velodyne> lidars;
    std::vector<Frame> frames;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>> lidar_colored;   // 经过上色的LiDAR点云，这些都是在雷达局部坐标系下的
    const Config config;

public:
    Texture(const std::vector<Velodyne>& _lidars, const std::vector<Frame>& _frames, const Config& _config);
    /**
     * @description: 对一定范围内的点云上色
     * @param min_dist {double} 最小距离，比这个距离小的直接跳过
     * @param max_dist {double} 最大距离，比这个距离大的直接跳过
     * @return 上色是否成功
     */    
    bool ColorizeLidarPointCloud(const double min_dist = 0, const double max_dist = 1000);
    // 融合雷达点，skip代表雷达融合的间隔，因为可能全融合的话太密集了
    // skip=1就是隔一个融合一个，skip=0就是全融合
    pcl::PointCloud<pcl::PointXYZRGB> FuseCloud(int skip = 1);
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>>& GetColoredLidar() const;
};
#endif