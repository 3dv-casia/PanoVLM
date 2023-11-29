/*
 * @Author: Diantao Tu
 * @Date: 2021-11-04 19:23:30
 */

#ifndef _CAMERA_LIDAR_LINE_ASSOCIATE_
#define _CAMERA_LIDAR_LINE_ASSOCIATE_


#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h> 
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <map>
#include <glog/logging.h>

#include "../base/Geometry.hpp"
#include "../util/Visualization.h"
#include "../sensors/Equirectangular.h"
#include "../sensors/Velodyne.h"
#include "../sensors/Frame.h"

#include "CameraLidarLinePair.h"

class CameraLidarLineAssociate
{
private:
    int rows, cols;
    cv::Mat img_gray;       // 目前只有可视化的时候需要它,所以不需要可视化的时候不用传递这个参数，节省内存
    std::vector<CameraLidarLinePair> line_pairs;   // 最终形成的图像直线和雷达直线的匹配对

    /**
     * @description: 过滤直线匹配对
     * @param filter_by_angle {bool} 是否根据角度对匹配的直线对进行过滤
     * @param filter_by_length {bool} 是否根据投影长度对匹配的直线对进行过滤
     */    
    void Filter(bool filter_by_angle, bool filter_by_length);
    /**
     * @description: 对输入的点云进行RANSAC拟合，得到一条直线
     * @param cloud 输入的点云
     * @param line_coeff {ModelCoefficients&} 拟合得到的直线参数
     * @param inlier 内点序号
     * @return {*} 拟合是否成功
     */    
    bool FitLineRANSAC(const pcl::PointCloud<pcl::PointXYZI>& cloud, pcl::ModelCoefficients& line_coeff, vector<size_t>& inlier);

    // 对已有的匹配对进行过滤，只允许图像直线和LiDAR直线是一对一的匹配，禁止多重匹配
    void UniqueLinePair(const std::vector<cv::Vec4f>& lines, const eigen_vector<Eigen::Vector3d>& lidar_lines_endpoint);

    /**
     * @description: 对已有的外参进行扰动，分别在三个轴上进行旋转和平移，旋转角度为rotation_step，平移为 translation_step
     * @param T_cl {Matrix4d&} 要被扰动的外参
     * @param rotation_step {float} 旋转的角度，单位为度
     * @param translation_step {float} 平移的距离，单位为米
     * @return 扰动后的结果，共3^6=729个，其中第一个是输入的T_cl
     */    
    eigen_vector<Eigen::Matrix4d> PerturbCalibration(const Eigen::Matrix4d& T_cl, const float rotation_step, const float translation_step);
                

public:
    CameraLidarLineAssociate(int _rows, int _cols, cv::Mat _img_gray);
    CameraLidarLineAssociate(int _rows, int _cols);
    // 普通图像  尚未完成
    void Associate(const std::vector<cv::Vec4f>& lines, const pcl::PointCloud<pcl::PointXYZI> point_cloud, 
                    const Eigen::Matrix4f T_cl, const Eigen::Matrix3f K);
    /**
     * @description: 适用于全景图像的直线-LiDAR特征关联，基本方法是把雷达点投影到图像上，找到在落在图像直线附近的雷达点，用这些雷达点
     *              拟合一条直线，作为与图像直线相关联的雷达直线。还包含一些过滤方法，滤掉明显错误的直线匹配关系
     * @param lines 所有的图像直线，每个直线用起点+终点来表示
     * @param point_cloud 雷达点云
     * @param T_cl 从雷达到相机的变换
     * @return {*}
     */    
    void Associate(const std::vector<cv::Vec4f>& lines, const pcl::PointCloud<pcl::PointXYZI> point_cloud, 
                    const Eigen::Matrix4d T_cl);

    // 全景图像，与上面的区别是这里使用的点云是已经进行了分割的，所以每个segment里应该就是一条直线，这种情况下进行匹配
    // 应该会有更好的结果
    void Associate(const std::vector<cv::Vec4f>& lines, const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const Eigen::Matrix4d T_cl);

    /**
     * @description: 适用于全景图像的直线-LiDAR特征关联，基本方法是计算所有雷达点到所有图像直线形成平面的夹角，对于每个图像直线，找到
     *               和它夹角比较小的那些雷达点，看这些雷达点是否大多集中在一两个雷达的segment内，如果是的话就进行匹配。
     *              这个方法和之前的方法区别在于：以前的都是用像素距离来衡量点到直线是否足够近，但是全景图像在上下两端畸变严重，像素距离
     *              无法真正度量点到直线距离；以前的方法是一条图像直线只能匹配一条雷达直线，一条雷达直线却能匹配多条图像直线，现在
     *              只能图像直线和LiDAR直线一对一的匹配，如果出现多重匹配就要进行剔除
     * @param lines 所有的图像直线，每个直线用起点+终点来表示
     * @param segmented_cloud 经过分割的点云，里面每一个点云都是认为是一条直线
     * @param segment_coeffs 每个segment所对应的直线参数
     * @param point_cloud 雷达点云，也就是segmented_cloud融合到一起并且去除重复点之后的结果
     * @param point_to_segment point_cloud里每个点分别处在哪些segment里
     * @param line_end_points 每条LiDAR直线的起点和终点，这个点是投影到LiDAR直线上的结果，不是LiDAR直线本身的点
     * @param T_cl 从雷达到相机的变换
     * @param multiple_association 是否允许图像直线和LiDAR直线之间存在一对多匹配，也就是一条直线对应多条直线的情况。设置为false会对匹配关系
     *                          进行过滤，仅保留一对一的匹配关系
     * @param image_line_mask 图像直线的mask，指定某些图像直线不参与直线匹配
     * @param lidar_line_mask LiDAR直线的mask，指定某些雷达直线不参与直线匹配
     * @return {*}
     */    
    void AssociateByAngle(const std::vector<cv::Vec4f>& lines, const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const eigen_vector<Vector6d>& segment_coeffs,
                    const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
                    const std::vector<std::set<int>>& point_to_segment,
                    const eigen_vector<Eigen::Vector3d>& line_end_points,
                    const Eigen::Matrix4d T_cl, const bool multiple_association = false,
                    const std::vector<bool>& image_line_mask = std::vector<bool>(),
                    const std::vector<bool>& lidar_line_mask = std::vector<bool>());

    // 对输入的初始T_cl进行扰动，使用扰动结果进行直线匹配，保留最好的匹配结果。这个扰动过程是迭代的，也就是扰动一次，选择一个最好结果，
    // 然后在此基础上继续扰动，选择最好结果。这个思路是从相机雷达标定得到的，比如
    // Automatic Online Calibration of Cameras and Lasers - RSS 2013
    // Line-based Automatic Extrinsic Calibration of LiDAR and Camera - ICRA 2021
    Eigen::Matrix4d AssociateRandomDisturbance(const std::vector<cv::Vec4f>& lines, const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const eigen_vector<Vector6d>& segment_coeffs,
                    const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
                    const std::vector<std::set<int>>& point_to_segment,
                    const eigen_vector<Eigen::Vector3d>& line_end_points,
                    const Eigen::Matrix4d T_cl, const bool multiple_association = false,
                    const std::vector<bool>& image_line_mask = std::vector<bool>(),
                    const std::vector<bool>& lidar_line_mask = std::vector<bool>());
    
    Eigen::Matrix4d AssociateRandomDisturbance(const std::vector<cv::Vec4f>& lines,
                    const Frame& frame, const Velodyne& lidar,
                    const Eigen::Matrix4d T_cl, const bool multiple_association = false,
                    const std::vector<bool>& image_line_mask = std::vector<bool>(),
                    const std::vector<bool>& lidar_line_mask = std::vector<bool>());

    const std::vector<CameraLidarLinePair> GetAssociatedPairs();
    ~CameraLidarLineAssociate();
};



#endif