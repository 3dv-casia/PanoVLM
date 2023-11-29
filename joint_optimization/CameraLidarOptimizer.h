/*
 * @Author: Diantao Tu
 * @Date: 2021-10-22 15:06:46
 */

#ifndef _CAMERA_LIDAR_OPTIMIZER_
#define _CAMERA_LIDAR_OPTIMIZER_

#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <glog/logging.h>
#include <omp.h>

#include "../base/CostFunction.h"
#include "../base/Config.h"
#include "../base/Geometry.hpp"
#include "../base/ProcessBar.h"

#include "../sensors/Equirectangular.h"
#include "../sensors/Velodyne.h"
#include "../sensors/Frame.h"

#include "../util/PanoramaLine.h"
#include "../util/MatchPair.h"
#include "../util/Tracks.h"
#include "../util/FileIO.h"
#include "../util/Optimization.h"

#include "../sfm/Structure.h"
#include "../sfm/Triangulate.h"

#include "../lidar_mapping/LidarFeatureAssociate.h"
#include "../lidar_mapping/LidarLineMatch.h"

#include "CameraLidarTrackAssociate.h"
#include "CameraLidarLineAssociate.h"
#include "PanoramaLineMatch.h"

enum OptimizationMode{
    CALIBRATION = 1,
    MAPPING = 2
};

class CameraLidarOptimizer
{
private:
    Eigen::Matrix4d T_cl_init;
    Eigen::Matrix4d T_cl_optimized;
    const Config config;
    std::vector<Frame> frames;
    std::vector<Velodyne> lidars;
    std::vector<PanoramaLine> image_lines_all;
    std::vector<PointTrack> structure;
    int optimization_mode;
    // 对单一的图像和雷达之间进行直线关联，也就是说要求图像和雷达数量是相同的，第一张图像和第一帧雷达，第二张图像和第二帧雷达
    // 以此类推，这种情况适用于进行标定，只需要获得相机和雷达间的一个变换
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> AssociateLineSingle(Eigen::Matrix4d T_cl);

    /**
     * @description: 一张图像关联多帧雷达，图像和雷达之间的数量可以不同，图像和雷达之间也没有固定的位姿约束
     *              注意：要求雷达和图像必须都有位姿，且雷达在局部坐标系下
     * @param neighbor_size {int} 每张图像匹配的雷达数目
     * @param temporal {bool} 是否按照时域匹配。时域匹配就是按照时间顺序匹配，例如第 k 张图像就匹配第k-3,k-2,k-1,k,k+1,k+2个雷达。
     *              设置为false则按照空间匹配，也就是找距离当前图像最近的k个雷达数据进行匹配
     * @param use_lidar_track 使用LiDAR的track进行掩模，只有LiDAR直线属于某一条LiDAR track的时候才能参与匹配，这主要是由于
     *                      LiDAR直线提取有时候会提取出不存在的直线，那么使用track就可以过滤掉这类直线
     * @param use_image_track 使用图像直线的track进行掩模，只有图像直线属于某一条track的时候才能参与匹配。但这个最好别用，因为
     *                      相对于LiDAR直线提取，图像的直线大多比较准确同时图像直线之间的匹配不是很好，所以会过滤掉大量图像直线，
     *                      这就会大大减少参与匹配的图像直线。
     * @return 匹配的图像直线和雷达直线对
     */    
    eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>> AssociateLineMulti(const int neighbor_size, const bool temporal = true, 
                    const bool use_lidar_track = false, const bool use_image_track = false);
    
    /**
     * @description: 为每张图像找到和它近邻匹配的neighbor_size 个 LiDAR
     * @param neighbor_size {int} 近邻的数量
     * @param temporal {bool} 是否按照时间近邻寻找。设置为true的时候，第i帧图像就会和 i-2, i-1, i, i+1, i+2 这样在时间上连续的雷达去匹配
     *                  设置为false的时候，就使用图像和LiDAR位姿找空间的近邻
     * @return {*}
     */    
    std::vector<std::vector<int>> NeighborEachFrame(const int neighbor_size, const bool temporal = true);

    /**
     * @description: 通过track生成每个雷达直线对应的掩模，如果某条LiDAR直线被包含进了track，那么对应位置为true，否则为false
     * @param min_track_length {int} track至少要跨越min_track_length个雷达数据帧
     * @param neighbor_size {int} 生成track的时候，每帧雷达会找neighbor_size个近邻
     * @return 生成的掩模
     */    
    std::vector<std::vector<bool>> LidarMaskByTrack(const int min_track_length = 3, const int neighbor_size = 3);

    // 参数同上
    std::vector<std::vector<bool>> ImageMaskByTrack(const int min_track_length = 3, const int neighbor_size = 3);

    // 设置匹配的直线对的权重，根据图像直线的角度来设置，给予水平方向的直线较高的权重，竖直方向的直线较低的权重
    // 目前仅能应用于全景图像
    bool SetLineWeight(vector<CameraLidarLinePair>& line_pair, const size_t frame_idx, const size_t lidar_idx);

    // 对特征点进行关联，形成track，然后三角化三维点，这个和SfM里是一模一样的
    bool EstimateStructure(const std::vector<MatchPair>& image_pairs);

    
    // 根据三角化的点调整相机的位姿以及三维点的坐标，就是SfM里的全局BA，和它一模一样
    bool GlobalBundleAdjustment(std::vector<PointTrack>& structure, bool refine_structure, bool refine_rotation, bool refine_translation);

    // 图像和雷达是一对一的关系的时候进行优化的函数，也就是calibration时使用的函数
    int Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, const Eigen::Matrix4d& T_cl);
    
    /**
     * @description: 图像和雷达是一对多的关系的时候进行优化的函数，也就是mapping的时候使用的函数
     * @param line_pairs 匹配的直线对
     * @param structure 三角化的三维点
     * @param refine_camera_rotation 设置为false则固定相机旋转不变
     * @param refine_camera_trans 设置为false则固定相机平移不变
     * @param refine_lidar_rotation 设置为false则固定雷达旋转不变
     * @param refine_lidar_trans 设置为false则固定雷达平移不变
     * @param refine_structure 设置为false则固定三角化的三维点不变
     * @param cost 优化结束后最终的残差
     * @param steps 优化进行了多少步
     * @return 是否成功
     */    
    int Optimize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs, std::vector<PointTrack>& structure,
                const bool refine_camera_rotation, const bool refine_camera_trans, const bool refine_lidar_rotation, 
                const bool refine_lidar_trans, const bool refine_structure,
                double& cost, int& steps);
    
public:
    CameraLidarOptimizer(const Eigen::Matrix4d _T_cl);
    CameraLidarOptimizer(const Eigen::Matrix4f _T_cl);
    CameraLidarOptimizer(const Eigen::Matrix4d _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config);
    CameraLidarOptimizer(const Eigen::Matrix4f _T_cl, const std::vector<Velodyne>& lidars, 
                        const std::vector<Frame>& frames, const Config& _config);

    // 提取每张图像上的直线
    // visualization 用来决定是否要可视化提取的直线，如果为true就会输出直线图像到 config.joint_result_path
    bool ExtractImageLines(string image_line_folder, bool visualization = false);
    // 提取每帧雷达的直线特征
    // visualization 用来决定是否要可视化提取的LiDAR特征，如果为true就会输出特征点云到 config.joint_result_path
    bool ExtractLidarLines(bool visualization = false);
    // 进行联合优化
    // visualization 用来决定是否要可视化匹配的直线特征，如果为true就会输出匹配到 config.joint_result_path
    bool JointOptimize(bool visualization = false);
    // 设置优化的模式是calibration还是mapping
    void SetOptimizationMode(int mode);
    // 可视化匹配的直线，雷达到图像投影，等等
    void Visualize(const eigen_map<std::pair<size_t, size_t>, std::vector<CameraLidarLinePair>>& line_pairs_all, 
                    const std::string path, int line_width = 12, int point_size = 5);

    /**
     * @description: 把激光雷达点云根据位姿融合起来
     * @param skip 每隔skip个点云进行融合，skip=0代表每个点云都参与融合
     * @param min_range 点云的最小距离，每个雷达点云里小于这个距离的点都会被过滤
     * @param max_range 点云的最大距离，每个雷达点云里大于这个距离的点都会被过滤
     * @return 融合后的点云
    */
    pcl::PointCloud<PointType> FuseLidar(int skip, double min_range, double max_range);
                    
    // 用于测试直线匹配的函数，debug用
    void TestLineAssociate(const vector<pair<int,int>>& image_lidar_pairs);
    // 用于测试track匹配的函数，debug用
    void TestTrackAssociate();
    // 用于测试随机扰动匹配的函数，debug用
    void TestRandomAssociate();
    const std::vector<Frame>& GetFrames();
    const std::vector<Velodyne>& GetLidars();
    Eigen::Matrix4d GetResult();
    // 得到所有图像的全局旋转 R_wc, with_invalid=true代表返回的旋转中包含没有计算位姿的frame
    eigen_vector<Eigen::Vector3d> GetCameraTranslation(bool with_invalid=false);
    // 得到所有图像的全局旋转 t_wc, with_invalid=true代表返回的平移中包含没有计算位姿的frame
    eigen_vector<Eigen::Matrix3d> GetCameraRotation(bool with_invalid=false);
    // 得到所有图像的名字，with_invalid=true代表返回的名字中包含没有计算位姿的frame
    std::vector<std::string> GetImageNames(bool with_invalid=false);
    // 得到所有雷达的全局旋转 R_wc, with_invalid=true代表返回的旋转中包含没有计算位姿的frame
    eigen_vector<Eigen::Vector3d> GetLidarTranslation(bool with_invalid=false);
    // 得到所有雷达的全局旋转 t_wc, with_invalid=true代表返回的平移中包含没有计算位姿的frame
    eigen_vector<Eigen::Matrix3d> GetLidarRotation(bool with_invalid=false);
    // 得到所有雷达的名字，with_invalid=true代表返回的名字中包含没有计算位姿的frame
    std::vector<std::string> GetLidarNames(bool with_invalid=false);

    void SetFrames(const std::vector<Frame>& _frames);

    void SetLidars(const std::vector<Velodyne>& _lidars);

    bool ExportStructureBinary(const std::string file_name);
};



#endif