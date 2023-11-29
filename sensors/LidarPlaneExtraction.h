/*
 * @Author: Diantao Tu
 * @Date: 2022-09-12 14:54:53
 */

#ifndef _LIDAR_PLANE_EXTRACTION_H_
#define _LIDAR_PLANE_EXTRACTION_H_

#include <vector>
#include <map>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include "../base/Math.h"

using namespace std;

std::vector<std::vector<int>> PlaneSegmentation(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
                                            const std::vector<std::vector<int> >& image_to_point_idx,
                                            int min_cluster_size, int max_cluster_size);

std::vector<std::vector<int>> PlaneSegmentation2(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, 
                                            const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& range_image,
                                            const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
                                            const std::vector<std::vector<int> >& image_to_point_idx,
                                            int min_cluster_size, int max_cluster_size);

pcl::PointCloud<pcl::Normal> ComputeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int k,
                                            const std::vector<std::pair<size_t, size_t> >& point_idx_to_image, 
                                            const std::vector<std::vector<int> >& image_to_point_idx
                                            );

pcl::PointCloud<pcl::Normal> ComputeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int k);

#endif 