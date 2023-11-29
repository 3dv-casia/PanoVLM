/*
 * @Author: Diantao Tu
 * @Date: 2022-04-24 10:57:52
 */

#ifndef _CAMERA_LIDAR_TRACK_ASSOCIATE_H_
#define _CAMERA_LIDAR_TRACK_ASSOCIATE_H_

#include <Eigen/Sparse>
#include <omp.h>
#include "../sensors/Frame.h"
#include "../util/Tracks.h"
#include "CameraLidarLinePair.h"
#include "../sensors/Velodyne.h"
#include "CameraLidarLineAssociate.h"
#include "../util/PanoramaLine.h"

/**
 * @description: 通过图像track和雷达track之间的匹配进行单张图像上的直线匹配。也就是让图像直线和LiDAR直线组成track，然后
 *              根据初始的图像-LiDAR直线匹配关系得到track级别的匹配。如果两条track是相互匹配的，那么这两条track上的所有
 *              直线特征之间都是相互匹配的。
 * @param image_tracks 图像直线组成的track，注意track id 必须从0开始
 * @param lidar_tracks LiDAR直线组成的track，注意 track id必须从0开始
 * @param image_lines_all 所有的图像直线
 * @param frames 所有的frame，主要是用这个相机的位姿
 * @param lidars 所有的LiDAR
 * @param each_frame_neighbor 每张图像需要匹配的LiDAR
 * @param T_cl_init 初始的相机-LiDAR外参
 * @return 相机和LiDAR的匹配关系
 */
eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> AssociateTrack(
                                                const std::vector<LineTrack>& image_tracks, const std::vector<LineTrack>& lidar_tracks, 
                                                const std::vector<PanoramaLine>& image_lines_all, const std::vector<Frame>& frames,
                                                const std::vector<Velodyne>& lidars, const std::vector<std::vector<int>>& each_frame_neighbor,
                                                const Eigen::Matrix4d& T_cl_init
                                                );


#endif