/*
 * @Author: Diantao Tu
 * @Date: 2022-06-28 17:32:18
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "../sensors/Frame.h"
#include "../sensors/Velodyne.h"
#include "../base/common.h"
#include "FileIO.h"

/**
 * @description: 根据时间偏差和LiDAR位姿来设置相机位姿
 * @param frames 图像
 * @param lidars 雷达，要有位姿
 * @param T_cl 时间偏差为零时相机和雷达的外参
 * @param time_offset 时间偏差，正数代表雷达比图像晚，负数代表雷达比图像早， frame_time = lidar_time + time_offset 
 * @param time_gap 连续两帧之间的时间间隔，如果是连续采样的，那么间隔就是0，如果是隔一帧采样一帧，那么就应该是0.1(因为每一帧雷达持续0.1秒)
 * @return {*}
 */
bool SetFramePose(std::vector<Frame>& frames, const std::vector<Velodyne>& lidars, const Eigen::Matrix4d& T_cl, 
                    const double time_offset, const double time_gap = 0);

// 和上面的一样，只是用图像位姿来设置LiDAR位姿
bool SetLidarPose(const std::vector<Frame>& frames, std::vector<Velodyne>& lidars, const Eigen::Matrix4d& T_cl, 
                const double time_offset, const double time_gap = 0);

/**
 * @description: 从文件中读取图像位姿并赋予frame
 * @param frame
 * @param pose_file 保存位姿的文件
 * @return 是否成功
 */
bool LoadFramePose(std::vector<Frame>& frames, const std::string pose_file);

/**
 * @description: 从文件中读取图像位姿并赋予lidar
 * @param lidars
 * @param pose_file 保存位姿的文件
 * @param with_invalid 设置为true时有无效位姿的雷达也会保留下来，否则只保留有效位姿的雷达
 * @return 是否成功
 */
bool LoadLidarPose(std::vector<Velodyne>& lidars, const std::string pose_file, const bool with_invalid);

#endif