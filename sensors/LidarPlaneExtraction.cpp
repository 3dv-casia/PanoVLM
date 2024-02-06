/*
 * @Author: Diantao Tu
 * @Date: 2022-09-12 15:43:45
 */
#include "LidarPlaneExtraction.h"
#include "../base/common.h"
#include "../base/Geometry.hpp"

set<int> debug_ids = {6462};

std::vector<std::vector<int>> PlaneSegmentation(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
                                            const std::vector<std::vector<int> >& image_to_point_idx,
                                            int min_cluster_size, int max_cluster_size)
{
    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    #if 1
    *normals = ComputeNormals(cloud, 8, point_idx_to_image, image_to_point_idx);
    #else 
    *normals = ComputeNormals(cloud, 30);
    #endif


    // 对点云进行重新的排序 
    pcl::PointCloud<pcl::PointXYZI> cloud_reordered(*cloud);
    for(size_t i = 0; i < cloud_reordered.size(); i++)
    {
        cloud_reordered.points[i].intensity = 0;
        cloud_reordered.points[i].x = point_idx_to_image[i].first ;
        cloud_reordered.points[i].y = point_idx_to_image[i].second ;
        cloud_reordered.points[i].z = 1.f;
    }

    // 使用法向量进行区域生长分割
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = std::shared_ptr<pcl::search::Search<pcl::PointXYZI> >(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::RegionGrowing<pcl::PointXYZI, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(max_cluster_size);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(20);
    reg.setInputCloud(cloud_reordered.makeShared());
    reg.setInputNormals(normals);

    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    
    vector<pcl::PointIndices> cluster;
    reg.extract(cluster);
    vector<vector<int>> cluster_filtered;
    for(size_t i = 0; i < cluster.size(); i++)
    {
        set<int> scan_ids;
        for(const int point_id : cluster[i].indices)
        {
            int scan_id = point_idx_to_image[point_id].first;
            scan_ids.insert(scan_id);
        }
        if(scan_ids.size() > 1)
        {
            cluster_filtered.push_back(cluster[i].indices);
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloud_segmented;
    for(size_t i = 0; i < cluster_filtered.size(); i++)
    {
        // 生成随机颜色
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        for(const int point_id : cluster_filtered[i])
        {
            pcl::PointXYZRGB pt;
            pt.x = cloud->points[point_id].x;
            pt.y = cloud->points[point_id].y;
            pt.z = cloud->points[point_id].z;
            pt.r = r;
            pt.g = g;
            pt.b = b;
            cloud_segmented.push_back(pt);
        }
    }
    pcl::io::savePCDFileBinary("cloud.pcd", *cloud);
    pcl::io::savePCDFileBinary("cloud_segmented.pcd", cloud_segmented);

    return cluster_filtered;

}

pcl::PointCloud<pcl::Normal> ComputeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int k,
                                            const std::vector<std::pair<size_t, size_t> >& point_idx_to_image, 
                                            const std::vector<std::vector<int> >& image_to_point_idx
                                            )
{
    // 假设是16线激光雷达，每线的点数为1800
    int image_width = 1800, image_height = 16;
    pcl::PointCloud<pcl::Normal> normals;
    for(size_t i = 0; i < cloud->size(); i++)
    {
        int num_points_current_line = k * 0.7;
        int num_points_neighbor_line = max(k - num_points_current_line, 1);
        std::vector<int> neighbor_ids_upper, neighbor_ids_lower;
        int row = point_idx_to_image[i].first, col = point_idx_to_image[i].second;
        int upper_row = row + 1, lower_row = row - 1;
        bool use_upper_row = (upper_row < image_height), use_lower_low = (lower_row >= 0);
        for(int curr_col = col - num_points_current_line / 2; 
            (neighbor_ids_upper.size() < num_points_current_line && use_upper_row) || (neighbor_ids_lower.size() < num_points_current_line && use_lower_low); 
            curr_col++)
        {
            if(curr_col == col)
                continue;
            if(curr_col < 0)
                curr_col += image_width;
            if(curr_col >= image_width)
                curr_col -= image_width;
            int point_idx = image_to_point_idx[row][curr_col];
            if(point_idx >= 0)
            {
                if(use_upper_row)
                    neighbor_ids_upper.push_back(point_idx);
                if(use_lower_low)
                    neighbor_ids_lower.push_back(point_idx);
            }
        }
        for(int curr_col = col - num_points_neighbor_line / 2; 
                (neighbor_ids_upper.size() < k && use_upper_row) || (neighbor_ids_lower.size() < k && use_lower_low); 
                curr_col++)
        {
            if(curr_col == col)
                continue;
            if(curr_col < 0)
                curr_col += image_width;
            if(curr_col >= image_width)
                curr_col -= image_width;
            if(use_upper_row)
            {
                int point_idx = image_to_point_idx[upper_row][curr_col];
                if(point_idx >= 0)
                    neighbor_ids_upper.push_back(point_idx);
            }
            if(use_lower_low)
            {
                int point_idx = image_to_point_idx[lower_row][curr_col];
                if(point_idx >= 0)
                    neighbor_ids_lower.push_back(point_idx);
            }
        }
        
        pcl::Normal normal, normal1, normal2;
        Eigen::Matrix3f covariance_matrix;
        Eigen::Vector4f xyz_centroid;
        if(use_upper_row)
        {
            pcl::computeMeanAndCovarianceMatrix(*cloud, neighbor_ids_upper, covariance_matrix, xyz_centroid);
            pcl::solvePlaneParameters(covariance_matrix, normal1.normal_x, normal1.normal_y, normal1.normal_z, normal1.curvature);
            pcl::flipNormalTowardsViewpoint(cloud->points[i], 0, 0, 0, normal1.normal_x, normal1.normal_y, normal1.normal_z);

            normal = normal1;
        }
        if(use_lower_low)
        {
            pcl::computeMeanAndCovarianceMatrix(*cloud, neighbor_ids_lower, covariance_matrix, xyz_centroid);
            pcl::solvePlaneParameters(covariance_matrix, normal2.normal_x, normal2.normal_y, normal2.normal_z, normal2.curvature);
            pcl::flipNormalTowardsViewpoint(cloud->points[i], 0, 0, 0, normal2.normal_x, normal2.normal_y, normal2.normal_z);
            normal = normal2;
        }
        // 如果当前点的上下两行都有点，那么取两个法向量的平均值
        if(use_upper_row && use_lower_low)
        {
            float diff_angle = acos(normal1.normal_x * normal2.normal_x + normal1.normal_y * normal2.normal_y + normal1.normal_z * normal2.normal_z);
            // cross product of two normals
            float cross_x = normal1.normal_y * normal2.normal_z - normal1.normal_z * normal2.normal_y;
            float cross_y = normal1.normal_z * normal2.normal_x - normal1.normal_x * normal2.normal_z;
            float cross_z = normal1.normal_x * normal2.normal_y - normal1.normal_y * normal2.normal_x;
            Eigen::Matrix3f rot(Eigen::AngleAxisf(diff_angle / 2, Eigen::Vector3f(cross_x, cross_y, cross_z)));
            normal.normal_x = rot(0, 0) * normal1.normal_x + rot(0, 1) * normal1.normal_y + rot(0, 2) * normal1.normal_z;
            normal.normal_y = rot(1, 0) * normal1.normal_x + rot(1, 1) * normal1.normal_y + rot(1, 2) * normal1.normal_z;
            normal.normal_z = rot(2, 0) * normal1.normal_x + rot(2, 1) * normal1.normal_y + rot(2, 2) * normal1.normal_z;
            normal.curvature = (normal1.curvature + normal2.curvature) / 2.f;
        }
        
        normals.push_back(normal);
        if(debug_ids.count(i) > 0)
        {
            pcl::PointCloud<pcl::PointXYZI> neighbor_points;
            for(const int point_id : neighbor_ids_upper)
            {
                neighbor_points.push_back(cloud->points[point_id]);
            }
            pcl::io::savePCDFileASCII("2d_neighbor_" + num2str(i) + ".pcd", neighbor_points);

        }
    }
    return normals;
}

pcl::PointCloud<pcl::Normal> ComputeNormals(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int k)
{
    pcl::search::Search<pcl::PointXYZI>::Ptr tree = std::shared_ptr<pcl::search::Search<pcl::PointXYZI> >(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::PointCloud <pcl::Normal> normals;
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(k);
    normal_estimator.compute(normals);
    pcl::search::KdTree<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(cloud);
    for(int i : debug_ids)
    {
        std::vector<int> pointIdxNKNSearch;
        std::vector<float> pointNKNSquaredDistance;
        if(kdtree.nearestKSearch(cloud->points[i], k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            pcl::PointCloud<pcl::PointXYZI> neighbor_points;
            for(const int point_id : pointIdxNKNSearch)
            {
                neighbor_points.push_back(cloud->points[point_id]);
            }
            pcl::io::savePCDFileASCII("3d_neighbor_" + num2str(i) + ".pcd", neighbor_points);
        }
    }
    return normals;
}

// Fast Plane Detection and Polygonalization in Noisy 3D Range Images - IROS 2008
std::vector<std::vector<int>> PlaneSegmentation2(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, 
                                            const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& range_image,
                                            const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
                                            const std::vector<std::vector<int> >& image_to_point_idx,
                                            int min_cluster_size, int max_cluster_size)
{
    // 统计range_image 中大于0的元素的数量
    int num_points = 0;
    for(int i = 0; i < range_image.rows(); i++)
    {
        for(int j = 0; j < range_image.cols(); j++)
        {
            if(range_image(i, j) > 0)
            {
                num_points++;
            }
        }
    }

    // cout << "non zero :" << num_points << " cloud size: " << cloud->size() << endl;
    // cout << "point_idx_to_image size: " << point_idx_to_image.size() << " image_to_point_idx size: " << image_to_point_idx.size() << endl;
    std::vector<std::vector<int>> clusters;
    std::vector<bool> visited(cloud->size(), false);
    for(int i = 0; i < cloud->size(); ++i)
    {
        if(visited[i])
            continue;
        Eigen::Vector4f plane;      // 平面参数 ax + by + cz + d = 0
        std::vector<int> cluster;
        std::queue<int> q;
        q.push(i);
        while(!q.empty())
        {
            int point_id = q.front();
            q.pop();
            if(visited[point_id])
                continue;
            
            visited[point_id] = true;
            // 判断当前点能否和已有的点构成平面
            cluster.push_back(point_id);
            if(cluster.size() >= 3)
            {
                if(cluster.size() > 3)
                {
                    float d = abs(plane[0] * cloud->points[point_id].x + plane[1] * cloud->points[point_id].y + plane[2] * cloud->points[point_id].z + plane[3]);
                    if(d > 0.1)
                    {
                        cluster.pop_back();
                        visited[point_id] = false;
                        continue;
                    }
                }
                Eigen::Matrix3f covariance_matrix;
                Eigen::Vector4f xyz_centroid;
                float curvature;

                pcl::computeMeanAndCovarianceMatrix(*cloud, cluster, covariance_matrix, xyz_centroid);
                pcl::solvePlaneParameters(covariance_matrix, xyz_centroid, plane, curvature);
                pcl::flipNormalTowardsViewpoint(cloud->points[i], 0, 0, 0, plane);
                float mse = 0, new_point_distance = 0;
                for(const int& id : cluster)
                {
                    new_point_distance = cloud->points[id].x * plane[0] + cloud->points[id].y * plane[1] + cloud->points[id].z * plane[2] + plane[3];
                    mse += Square(new_point_distance);
                }
                mse /= cluster.size();
                if(mse > Square(0.03) || abs(new_point_distance) > 0.03)
                {
                    cluster.pop_back();
                    visited[point_id] = false;
                    continue;
                }
                

            }
            // image_idx.first 是行数，image_idx.second 是列数
            std::pair<size_t, size_t> image_idx = point_idx_to_image[point_id];
            const float& depth1 = range_image(image_idx.first, image_idx.second);
            // 遍历当前点的上下左右四个邻居，如果邻居点的深度差小于阈值，则认为可能是处于同一平面上的点
            // 那么就把邻居点加入当前的队列中
            for(int dx = -1; dx <= 1; ++dx)
            {
                for(int dy = -1; dy <= 1; ++dy)
                {
                    if(dx == 0 && dy == 0)
                        continue;
                    int neighbor_x = image_idx.second + dx;
                    int neighbor_y = image_idx.first + dy;
                    if(neighbor_y > range_image.rows() - 1 || neighbor_y < 0)
                        continue;
                    neighbor_x = (neighbor_x > range_image.cols() - 1 ? neighbor_x - range_image.cols() : neighbor_x);
                    neighbor_x = (neighbor_x < 0 ? neighbor_x + range_image.cols() : neighbor_x);
                    int neighbor_id = image_to_point_idx[neighbor_y][neighbor_x];
                    if(neighbor_id >= 0)
                    {
                        // const float& depth2 = range_image(neighbor_image_idx.first, neighbor_image_idx.second);
                        // if(abs(depth1 - depth2) / depth1 > 0.02)
                        //     continue;
                        q.push(neighbor_id);
                    }
                }
            }
        }
        bool is_valid = true;
        if(cluster.size() < min_cluster_size || cluster.size() > max_cluster_size)
            is_valid = false;
        if(is_valid)
        {
            set<int> scan_in_cluster;
            for(const int& id : cluster)
                scan_in_cluster.insert(point_idx_to_image[id].first);
            if(scan_in_cluster.size() < 2)
                is_valid = false;
        }
        if(is_valid)
            clusters.push_back(cluster);
        else
        {
            for(const int& id : cluster)
                visited[id] = false;
        }
    }


    vector<vector<int>> cluster_filtered;
    for(size_t i = 0; i < clusters.size(); i++)
    {
        set<int> scan_ids;
        for(const int& point_id : clusters[i])
        {
            int scan_id = point_idx_to_image[point_id].first;
            scan_ids.insert(scan_id);
        }
        if(scan_ids.size() > 1)
        {
            cluster_filtered.push_back(clusters[i]);
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloud_segmented;
    for(size_t i = 0; i < cluster_filtered.size(); i++)
    {
        // 生成随机颜色
        int r = rand() % 255;
        int g = rand() % 255;
        int b = rand() % 255;
        for(const int point_id : cluster_filtered[i])
        {
            pcl::PointXYZRGB pt;
            pt.x = cloud->points[point_id].x;
            pt.y = cloud->points[point_id].y;
            pt.z = cloud->points[point_id].z;
            pt.r = r;
            pt.g = g;
            pt.b = b;
            cloud_segmented.push_back(pt);
        }
    }
    pcl::io::savePCDFileBinary("cloud.pcd", *cloud);
    pcl::io::savePCDFileBinary("cloud_segmented.pcd", cloud_segmented);

    return cluster_filtered;

}
