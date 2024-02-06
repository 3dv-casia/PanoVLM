/*
 * @Author: Diantao Tu
 * @Date: 2022-03-16 14:20:41
 */

#include "Texture.h"

using namespace std;

Texture::Texture(const std::vector<Velodyne>& _lidars, const std::vector<Frame>& _frames, const Config& _config):
        lidars(_lidars), frames(_frames), config(_config)
{}

bool Texture::ColorizeLidarPointCloud(const double min_dist, const double max_dist)
{
    if(lidars.empty() || frames.empty())
    {
        LOG(ERROR) << "lidar or frames are empty";
        return false;
    }
    LOG(INFO) << "================ Colorize Lidar Point Cloud begin ==================";
    assert(lidars.size() == frames.size());
    lidar_colored.reserve(lidars.size());
    lidar_colored.resize(lidars.size());
    omp_set_num_threads(config.num_threads);
    #pragma omp parallel for
    for(Velodyne& l : lidars)
    {
        if(!l.IsPoseValid())
            continue;
        l.LoadLidar(l.name);
        // 对LiDAR进行Segmentation，去除小物体，去除掉一些由于遮挡、噪声形成的小点云，这样结果会更加美观
        l.ReOrderVLP();
        l.Segmentation();
    }
    const double min_dist_sq = min_dist * min_dist;
    const double max_dist_sq = max_dist * max_dist;
    #pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid() || !frames[i].IsPoseValid())
            continue;
        cv::Mat img_color = frames[i].GetImageColor();
        cv::Mat img_hsv;
        cv::cvtColor(img_color, img_hsv, cv::COLOR_BGR2HSV);
        pcl::PointCloud<pcl::PointXYZRGB> cloud_colored;
        Equirectangular eq(frames[i].GetImageRows(), frames[i].GetImageCols());
        Eigen::Matrix4d T_cl = frames[i].GetPose().inverse() * lidars[i].GetPose();
        for(const pcl::PointXYZI& pt_raw : lidars[i].cloud_scan)
        {
            const double distance = pt_raw.x * pt_raw.x + pt_raw.y * pt_raw.y + pt_raw.z * pt_raw.z; 
            if(distance < min_dist_sq || distance > max_dist_sq)
                continue;
            Eigen::Vector3d point = PclPonit2EigenVecd(pt_raw);
            point = (T_cl * point.homogeneous()).hnormalized();
            Eigen::Vector2d pt_projected = eq.CamToImage(point);
            cv::Point2i pt_pixel(round(pt_projected.x()), round(pt_projected.y()));
            if(!eq.IsInside(pt_pixel))
                continue;
            // H -色调 S - 饱和度 V - 明度
            cv::Vec3b hsv = img_hsv.at<cv::Vec3b>(pt_pixel);
            if(hsv[0] >= 100 && hsv[0] <= 124 && hsv[1] >= 43 && hsv[1] <= 200 && hsv[2] >= 150 && hsv[2] <= 255)
                continue;
            pcl::PointXYZRGB pt_colored;
            cv::Vec3b bgr = img_color.at<cv::Vec3b>(pt_pixel);
            pt_colored.x = pt_raw.x;
            pt_colored.y = pt_raw.y;
            pt_colored.z = pt_raw.z;
            pt_colored.b = bgr[0];
            pt_colored.g = bgr[1];
            pt_colored.r = bgr[2];
            cloud_colored.push_back(pt_colored);
        }
        lidar_colored[i] = cloud_colored;

    }
    LOG(INFO) << "================ Colorize Lidar Point Cloud end ==================";

    return lidar_colored.size() > 0;
}

pcl::PointCloud<pcl::PointXYZRGB> Texture::FuseCloud(int skip)
{
    assert(skip >= 0);
    pcl::PointCloud<pcl::PointXYZRGB> cloud_fused;
    for(size_t i = 0; i < lidar_colored.size(); i += (skip + 1))
    {
        if(!lidars[i].IsPoseValid())
            continue;
        pcl::PointCloud<pcl::PointXYZRGB> cloud_world;
        pcl::transformPointCloud(lidar_colored[i], cloud_world, lidars[i].GetPose());
        cloud_fused += cloud_world;
    }
    return cloud_fused;
}

const std::vector<pcl::PointCloud<pcl::PointXYZRGB>>& Texture::GetColoredLidar() const
{
    return lidar_colored;
}