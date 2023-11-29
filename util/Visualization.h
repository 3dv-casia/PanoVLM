/*
 * @Author: Diantao Tu
 * @Date: 2021-10-22 17:15:42
 */

#ifndef _VISUALIZATION_H_
#define _VISUALIZATION_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h> 
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <glog/logging.h>
#include "../sensors/Equirectangular.h"
#include "../base/common.h"
#include "../base/Geometry.hpp"
#include "../joint_optimization/CameraLidarLinePair.h"

// 这里包含的都是各种各样的可视化函数，基本都是用来debug的，比如雷达投影到图像、在图像上画直线等等

using namespace std;

//灰度图转为彩虹图:灰度值255~0分别对应：红、橙、黄、绿、青、蓝。
cv::Vec3b Gray2Color(uchar gray);

/**
 * @description: 把输入的BGR颜色转换为HSV色域
 * @param bgr {Vec3b&} BGR颜色，取值范围0-255
 * @return HSV颜色，取值范围 0-1
 */
cv::Vec3f BGR2HSV(const cv::Vec3b& bgr);

/**
 * @description: 在图像上以指定颜色画一条线
 * @param img {Mat} 要画线的图像，如果要画彩色的线需要输入三通道图像
 * @param line {Vec4f} 直线的表达式，用起点和终点表示
 * @param color {Scalar} 线的颜色
 * @param thickness {int} 线的宽度
 * @param panoramic {bool} 是否是在全景图像上画直线
 * @param id 图像直线的id，如果id > 0就在图像直线的起始位置显示id
 * @return {*}
 */
void DrawLine(cv::Mat& img, const cv::Vec4f& line, cv::Scalar color, int thickness, bool panoramic=false, int id = -1 );

/**
 * @description: 为每一条线都画一张图像，因此会产生大量图像，慎用
 * @param path 所有图像的保存路径
 * @param img 图像本身，灰度图或三通道彩色图都可以
 * @param lines 所有的直线
 * @param color 直线颜色，所有直线都是相同颜色的
 * @param thickness 直线宽度
 * @param panoramic 这张图像是否是全景图像
 * @return {*}
 */
void DrawEachLine(const string& path, const cv::Mat& img, const vector<cv::Vec4f>& lines, 
                const cv::Scalar& color, const int thickness,  const bool panoramic = false);

/**
 * @description: 把多条直线画在同一张图像上，可以支持多种颜色
 * @param img 图像本身，灰度图或三通道彩色图都可以
 * @param lines 所有的直线
 * @param colors 所有颜色，第1条直线用第1种颜色，第n条直线用第n种颜色，第n+1条直线用第1种颜色
 * @param thickness 直线宽度
 * @param panoramic 这张图像是否是全景图像
 * @param draw_id 在图像上显示直线id
 * @param ids 直线id，如果draw_id设置为false则此参数无效
 * @return 画了直线的图像，三通道彩色图
 */
cv::Mat DrawLinesOnImage(const cv::Mat& img, const vector<cv::Vec4f>& lines, const vector<cv::Scalar>& colors, 
                        const int thickness, const bool panoramic = false, const bool draw_id = false, 
                        const std::vector<int>& ids = std::vector<int>());

/**
 * @description: 在图像上画出图像直线和雷达直线的匹配关系，图像直线用红色表示，雷达直线用蓝色表示
 * @param img_gray 灰度图
 * @param line_pairs 在当前图像上的所有图像-LiDAR直线匹配关系
 * @param T_cl 从LiDAR到相机的变换矩阵
 * @param thickness 直线宽度
 * @param draw_id 在图像上显示直线id，设置为true则会把line_pairs里的直线id画在图像上
 * @return {*}
 */
cv::Mat DrawLinePairsOnImage(const cv::Mat& img_gray, const vector<CameraLidarLinePair>& line_pairs, 
                            const Eigen::Matrix4d& T_cl, const int thickness = 3, const bool draw_id = false);

/**
 * @description: 画出两张图像上匹配的图像直线，用垂直的方式组合
 * @param img1 第一张图像，灰度图彩色图都可以
 * @param lines1 第一张图上的所有直线（不仅仅是匹配的直线）
 * @param img2 第二张图像，灰度图彩色图都可以
 * @param lines2 第二张图上的所有直线（不仅仅是匹配的直线）
 * @param matches 直线的匹配关系
 * @param panoramic 这张图像是否是全景图像
 * @param {*}
 * @return {*}
 */
cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::Vec4f>& lines1,
                            const cv::Mat& img2, const std::vector<cv::Vec4f>& lines2,
                            const std::vector<cv::DMatch>& matches, const bool panoramic = false);

// 和上一个方法一样，仅仅是直线表示形式不同
cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::line_descriptor::KeyLine>& key_lines1,
                            const cv::Mat& img2, const std::vector<cv::line_descriptor::KeyLine>& key_lines2,
                            const std::vector<cv::DMatch>& matches, const bool panoramic = false);

/**
 * @description: 把深度图变为伪彩色图显示，深度小显示蓝色，深度大显示红色
 * @param depth_map {Mat&} 深度图，可以是16位的或32位的
 * @param max_depth {float} 最大深度限制，-1代表使用深度图本身的最大深度
 * @param min_depth {float} 最小深度限制，-1代表使用深度图本身的最小深度
 * @return {*}
 */
cv::Mat DepthImageRGB(const cv::Mat& depth_map, const float max_depth = -1, const float min_depth = -1);

cv::Mat DepthImageGray(const cv::Mat& depth_map, const float max_depth, const float min_depth);

// 把深度图变为16位的灰度图，保留绝对尺度
cv::Mat DepthImageGray16(const cv::Mat& depth_map);

// 把深度图和彩色图结合到一起，有深度的时候显示深度，没深度的时候显示彩色图像
cv::Mat CombineDepthWithRGB(const cv::Mat& depth_image, const cv::Mat& rgb_image, 
                            const float max_depth = 10, const float min_depth = 0);

// 在图像上画特征点，图像可以是灰度图，也可以是彩色图
cv::Mat DrawKeyPoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, 
                        const vector<cv::Scalar>& colors = vector<cv::Scalar>());

// 把两张图以竖直方式组合起来，然后画出匹配的特征点，
// 输入图像可以是灰度图，也可以是彩色图
/**
 * @description: 把两张图以竖直方式组合起来，然后画出匹配的特征点
 * @param img1 第一张图像，灰度图彩色图都可以
 * @param keypoints1 第一张图上的所有特征点（不仅仅是匹配的特征点）
 * @param img2 第二张图像，灰度图彩色图都可以
 * @param keypoints2 第二张图上的所有特征点（不仅仅是匹配的特征点）
 * @param matches 直线的匹配关系
 * @param inlier_idx 当它为空时画出所有的匹配关系，非空时只画出包含在其中的匹配关系
 * @return {*}
 */
cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::KeyPoint> keypoints1,
                            const cv::Mat& img2, const std::vector<cv::KeyPoint> keypoints2,
                            const std::vector<cv::DMatch>& matches,
                            const std::vector<size_t>& inlier_idx = std::vector<size_t>() );

/**
 * @description: 可视化相机位姿，用一个四棱锥表示一个相机，其中相机的光心在四棱锥的上方的“顶点”位置，四棱锥的顶点
 *              到它对面的那个面就是相机的朝向。那么这个朝向可以是x轴，也可以是y轴，也可以是z轴
 * @param plyfile 文件名
 * @param R_wc_list 从相机到世界的旋转矩阵
 * @param t_wc_list 从相机到世界的平移向量
 * @param main_axis 主轴，表示可视化的方向是xyz中的哪个轴。0 = x轴，1 = y轴，2 = z轴
 * @return 是否保存成功
 */
bool CameraPoseVisualize(const string& plyfile, const eigen_vector<Eigen::Matrix3d>& R_wc_list, 
                        const eigen_vector<Eigen::Vector3d>& t_wc_list, const int main_axis = 2);

/**
 * @description: 可视化相机位姿，用一个小坐标系表示一个相机，每个小坐标系四个点，原点+3个坐标轴上的点，
 *              相比于上一个方法，这个方法能一下子看出三个轴的朝向，但是没那么好看
 * @param plyfile 文件名
 * @param R_wc_list 从相机到世界的旋转矩阵
 * @param t_wc_list 从相机到世界的平移向量
 * @return 是否保存成功
 */
bool CameraPoseVisualizeCoord(const string& plyfile, const eigen_vector<Eigen::Matrix3d>& R_wc_list, 
                        const eigen_vector<Eigen::Vector3d>& t_wc_list);


bool CameraCenterPCD(const string& file_name, const eigen_vector<Eigen::Vector3d>& t_wc_list);

void SaveDepthImageRaw(const cv::Mat& depth_image, const std::string file_path);

cv::Mat DrawNormalImage(const cv::Mat& normal_image, bool normalized);

/**
 * @description: 把雷达点投影到图像上
 * @param cloud 要投影的点云
 * @param image 点云要投影的图像，必须是单通道的灰度图
 * @param K 内参
 * @param T_cl 点云到相机的变换
 * @return 投影结果，三通道灰度图，点云的投影点用红色表示
 */
template<typename T>
cv::Mat ProjectLidar2ImageGray(const pcl::PointCloud<T> cloud, const cv::Mat image, 
                        const Eigen::Matrix3f K, const Eigen::Matrix4f T_cl)
{
    if(image.channels() != 1)
    {
        cout << "input image is not gray scale" << endl;
        return cv::Mat() ;
    }
    // 把单通道的灰度图变成三通道的灰度图，这是因为投影的点要用彩色表示，需要三通道
    cv::Mat img_out;
    cv::cvtColor(image, img_out, CV_GRAY2BGR);
    bool high_res = (img_out.rows * img_out.cols > 1280 * 720);
    pcl::PointCloud<T> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);
    for(T p:cloud_trans.points)
    {
        Eigen::Vector3f point(p.x, p.y, p.z);
        point = K * point;
        if(point[2] <= 0)
            continue;
        float real_depth = point[2];
        int u = ceil(point[0] / real_depth);
        int v = ceil(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255);
        if(!high_res)
            continue;
        u = floor(point[0] / real_depth);
        v = floor(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255);   

        u = ceil(point[0] / real_depth);
        v = floor(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255);   

        u = floor(point[0] / real_depth);
        v = ceil(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = cv::Vec3b(0,0,255); 
    }
    return img_out;
}

/**
 * @description: 把雷达点投影到图像上，投影的结果根据深度用彩虹图表示
 * @param cloud 要投影的点云
 * @param image 彩色图或灰度图
 * @param K 内参
 * @param T_cl 从雷达到相机的变换矩阵
 * @param max_depth 投影的最大深度，超过max_depth的点一律用红色表示
 * @param min_depth 投影的最小深度，小于min_depth的点一律用蓝色表示
 * @return {cv::Mat} 投影结果
 */
template<typename T>
cv::Mat ProjectLidar2ImageRGB(const pcl::PointCloud<T> cloud, const cv::Mat image, 
                        const Eigen::Matrix3f K, const Eigen::Matrix4f T_cl, 
                        const float min_depth = 0,const float max_depth = 10)
{
    cv::Mat img_out;
    if(image.channels() == 3)
        img_out = image.clone();
    else if(image.channels() == 1)
        cv::cvtColor(image, img_out, CV_GRAY2BGR);
    else
    {
        cout << "error : image channel is neither 1 nor 3" << endl;
        return cv::Mat();
    }

    const float depth_diff = max_depth - min_depth;
    bool high_res = (img_out.rows * img_out.cols > 1280 * 720);
    pcl::PointCloud<T> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);
    for(T p:cloud_trans.points)
    {
        Eigen::Vector3f point(p.x, p.y, p.z);
        point = K * point;
        if(point[2] <= 0)
            continue;
        float real_depth = point[2];
        int u = ceil(point[0] / real_depth);
        int v = ceil(point[1] / real_depth);
        if(point[2] > max_depth)
            point[2] = max_depth;
        else if(point[2] < min_depth)
            point[2] = min_depth;
        uchar relative_depth = static_cast<unsigned char>((point[2]-min_depth) / depth_diff * 255);
        cv::Vec3b color = Gray2Color(relative_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = color;   // u,v 是坐标，那么相应的行列就是第v行第u列
        // if the image is high resolution, set the project point to 4 pixel for better virtualization
        if(!high_res)
            continue;
        u = floor(point[0] / real_depth);
        v = floor(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = color;   

        u = ceil(point[0] / real_depth);
        v = floor(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = color;   

        u = floor(point[0] / real_depth);
        v = ceil(point[1] / real_depth);
        if(u < img_out.cols && v < img_out.rows && v > 0 && u > 0)       
            img_out.at<cv::Vec3b>(v,u) = color;
    }
    return img_out;
}

/**
 * @description: 把点云投影到全景图像上，投影点用红色表示，图像是灰度图
 * @param cloud 要投影的点云
 * @param image 灰度图
 * @param T_cl 变换矩阵，从点云变换到相机坐标系
 * @param size 投影点在图像上的大小，像素为单位
 * @return 投影图像
 */
template<typename T>
cv::Mat ProjectLidar2PanoramaGray(const pcl::PointCloud<T>& cloud, const cv::Mat image, 
                         const Eigen::Matrix4d T_cl, const size_t size = 3)
{
    if(image.channels() != 1)
    {
        cout << "input image is not gray scale" << endl;
        return cv::Mat() ;
    }
    // 把单通道的灰度图变成三通道的灰度图，这是因为投影的点要用彩色表示，需要三通道
    cv::Mat img_out;
    cv::cvtColor(image, img_out, CV_GRAY2BGR);
    bool high_res = (img_out.rows * img_out.cols > 1280 * 720);
    pcl::PointCloud<T> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);
    Equirectangular eq(image.rows, image.cols);
    
    for(const T& p : cloud_trans.points)
    {
        cv::Point3f point(p.x, p.y, p.z);
        cv::Point2f pixel = eq.SphereToImage(eq.CamToSphere(point));
        cv::Point2i rb(ceil(pixel.x) + size / 2, ceil(pixel.y) + size / 2);   // right bottom
        cv::Point2i lt(floor(pixel.x) - size / 2, floor(pixel.y) - size / 2); // left up
        while(!eq.IsInside(rb))
        {
            rb.x -= 1;
            rb.y -= 1;
        }
        while(!eq.IsInside(lt))              
        {
            lt.x += 1;
            lt.y += 1;
        }
        for(int u = lt.y; u <= rb.y; u++)
        {
            for(int v = lt.x; v <= rb.x; v++)
                img_out.at<cv::Vec3b>(u,v) = cv::Vec3b(0,0, 255);
        }

    }
    return img_out;
}

template<typename T>
cv::Mat ProjectLidar2PanoramaRGB(const pcl::PointCloud<T>& cloud, const cv::Mat image, 
                        const Eigen::Matrix4d T_cl, const float min_depth = 0,const float max_depth = 10, 
                        const size_t size = 3)
{
    cv::Mat img_out;
    if(image.channels() == 3)
        img_out = image.clone();
    else if(image.channels() == 1)
        cv::cvtColor(image, img_out, CV_GRAY2BGR);
    else
    {
        cout << "error : image channel is neither 1 nor 3" << endl;
        return cv::Mat();
    }
    const float depth_diff = max_depth - min_depth;

    pcl::PointCloud<T> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);
    Equirectangular eq(image.rows, image.cols);
    
    for(T p:cloud_trans.points)
    {
        cv::Point3f point(p.x, p.y, p.z);
        cv::Point2f pixel = eq.SphereToImage(eq.CamToSphere(point));
        cv::Point2i rb(ceil(pixel.x) + size / 2, ceil(pixel.y) + size / 2);   // right bottom
        cv::Point2i lt(floor(pixel.x) - size / 2, floor(pixel.y) - size / 2); // left up
        if(!eq.IsInside(rb))
        {
            continue;
        }
        if(!eq.IsInside(lt))              
        {
            continue;
        }
        float depth = sqrt(p.x*p.x + p.y * p.y + p.z * p.z);
        if(depth > max_depth)
            depth = max_depth;
        else if(depth < min_depth)
            depth = min_depth;
        uchar relative_depth = static_cast<unsigned char>((depth - min_depth) / depth_diff * 255);
        cv::Vec3b color = Gray2Color(relative_depth);

        for(int u = lt.y; u <= rb.y; u++)
        {
            for(int v = lt.x; v <= rb.x; v++)
                img_out.at<cv::Vec3b>(u,v) = color;
        }
    }
    return img_out;
}

// 把雷达点变成稀疏的深度图
template<typename T>
cv::Mat ProjectLidar2PanoramaDepth(const pcl::PointCloud<T>& cloud, const int rows, const int cols, 
                        const Eigen::Matrix4d& T_cl, const size_t size = 3)
{
    cv::Mat img_out = cv::Mat::zeros(rows, cols, CV_16U);

    pcl::PointCloud<T> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);
    Equirectangular eq(rows, cols);
    
    for(const T& p : cloud_trans.points)
    {
        cv::Point3f point(p.x, p.y, p.z);
        cv::Point2f pixel = eq.SphereToImage(eq.CamToSphere(point));
        cv::Point2i rb(ceil(pixel.x) + size / 2, ceil(pixel.y) + size / 2);   // right bottom
        cv::Point2i lt(floor(pixel.x) - size / 2, floor(pixel.y) - size / 2); // left up
        if(!eq.IsInside(rb))
        {
            continue;
        }
        if(!eq.IsInside(lt))              
        {
            continue;
        }
        float depth = sqrt(p.x*p.x + p.y * p.y + p.z * p.z);
        uint16_t relative_depth = static_cast<uint16_t>(depth * 256.0);

        for(int u = lt.y; u <= rb.y; u++)
        {
            for(int v = lt.x; v <= rb.x; v++)
                img_out.at<uint16_t>(u,v) = relative_depth;
        }
    }
    return img_out;
}

template<typename T>
pcl::PointCloud<pcl::PointXYZI> ColorizeCloudByTime(const pcl::PointCloud<T>& cloud)
{
    pcl::PointCloud<pcl::PointXYZI> cloud_out;
    for(size_t i = 0; i < cloud.size(); i++)
    {
        pcl::PointXYZI point;
        point.x = cloud.points[i].x;
        point.y = cloud.points[i].y;
        point.z = cloud.points[i].z;
        point.intensity = 1.f * i / cloud.size();
        cloud_out.push_back(point);
    }
    cloud_out.height = 1;
    cloud_out.width = static_cast<unsigned>(cloud_out.size());
    return cloud_out;
}

#endif

