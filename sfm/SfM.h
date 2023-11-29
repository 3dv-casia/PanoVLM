/*
 * @Author: Diantao Tu
 * @Date: 2021-11-21 14:47:10
 */

#ifndef _SFM_H_
#define _SFM_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <Eigen/Sparse>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <omp.h>

#include "../base/EssentialMatrix.h"
#include "../base/Config.h"
#include "../base/ProcessBar.h"
#include "../base/CostFunction.h"
#include "../base/common.h"

#include "../sensors/Velodyne.h"
#include "../sensors/Equirectangular.h"
#include "../sensors/Frame.h"

#include "../util/DepthCompletion.h"
#include "../util/Visualization.h"
#include "../util/Tracks.h"
#include "../util/MatchPair.h"
#include "../util/FileIO.h"
#include "../util/Optimization.h"

#include "PoseGraph.h"
#include "Triangulate.h"
#include "l1_solver_admm.hpp"
#include "LinearProgramming.h"
#include "VLAD.h"
#include "BATA.h"
#include "Structure.h"
#include "RotationAveraging.h"
#include "TranslationAveraging.h"



// 进行图片匹配时的方法
enum FrameMatchMethod{
    EXHAUSTIVE = 0x01,              // 所有图像之间都匹配
    CONTIGUOUS = 0x01 << 1,         // 只匹配连续的N张图片
    VLAD = 0x01 << 2,               // 使用VLAD匹配最相近的N张图像
    GPS = 0x01 << 3,                // 使用GPS寻找近邻的图像
    GPS_VLAD = 0x01 << 4            // 使用VLAD匹配最近的N张，然后用GPS过滤距离太远的
};

// 进行旋转平均时的方法
enum RotationAveragingMethod{
    ROTATION_AVERAGING_L1 = 1,      // 这个方法最好
    ROTATION_AVERAGING_L2 = 2
};

enum TranslationAveragingMethod{
    TRANSLATION_AVERAGING_SOFTL1 = 1,   // 这个方法最好
    TRANSLATION_AVERAGING_L1 = 2,
    TRANSLATION_AVERAGING_CHORDAL = 3,
    TRANSLATION_AVERAGING_L2IRLS = 4,    // 这个方法也很好
    TRANSLATION_AVERAGING_BATA = 5,
    TRANSLATION_AVERAGING_LUD = 6
};

class SfM
{
private:
    std::vector<MatchPair> image_pairs;     // 匹配的图像对(相对位姿，两视图三角化的点，匹配的特征点)
    std::vector<Frame> frames;              // 图像数据(图像，特征点，描述子)
    std::vector<Velodyne> lidars;           // 雷达数据
    std::vector<PointTrack> structure;      // 世界坐标系下的三维点
    bool track_triangulated;                // 是否已经三角化特征点了
    const Config& config;
    // 检查旋转和平移是否是符合真实情况的
    // 也就是把特征点三角化，然后看三角化后的点和球心的射线 与 原特征点和球心的射线 ，这两个射线的夹角是否足够小
    // inlier_idx 是三角化后得到的内点的索引
    int CheckRT(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, 
            const std::vector<bool>& is_inlier, const std::vector<cv::DMatch>& matches, 
            const std::vector<cv::Point3f>& keypoints1, 
            const std::vector<cv::Point3f>& keypoints2,
            double& parallax, eigen_vector<Eigen::Vector3d>& triangulated_points,
            std::vector<size_t>& inlier_idx);

    // 经过分解本质矩阵并且得到正确的旋转和平移后，进行近一步的优化，使相对位姿更准确
    bool RefineRelativePose(MatchPair& image_pair);

    /**
     * @description: 设置相对平移的尺度，这是根据从雷达投影的深度图得到的。首先把相对平移设置为单位向量，然后进行三角化得到一些点，
     *              把这些点投影到图像上得到深度d1，投影点本身在深度图上的深度为d2，那么d2/d1得到的就是平移的尺度
     *              这里还有一些噪点，因此还需要一定的过滤方法
     * @param eq {Equirectangular&} 用来投影的全景图像模型
     * @param pair {MatchPair&} 要设置相对平移的尺度的图像对
     * @return {*} 是否成功
     */    
    bool SetTranslationScaleDepthMap(const Equirectangular& eq, MatchPair& pair);

    bool SetTranslationScaleDepthMap(const bool keep_no_scale);

    /**
     * @description: 根据GPS信息确定相对平移的尺度
     * @param gps_file {string&} gps文件路径（这个gps结果是xyz的形式，不是经纬高的形式）
     * @param overwrite {bool} 根据gps得到的尺度和根据深度图得到的尺度可能不相同，是否用gps结果覆盖深度图结果
     * @return {*} 是否成功
     */    
    bool SetTranslationScaleGPS(const std::string& gps_file, bool overwrite);

    // 通过有尺度图像对设置无尺度图像对的尺度
    // 主要就是先找到一个无尺度图像对，然后根据近邻关系找到他周围的有尺度图像对，接着把近邻的图像对尺度插值到当前无尺度的图像对上
    bool PropagateTranslationScale();

    /**
     * @description: 从输入的图像对中找到最大的边双连通的子图并返回构成这个子图的图像对
     * @param pairs 初始的图像对
     * @param nodes 最大的子图对应的图像的id
     * @return 构成最大边双连通子图的图像对
     */    
    std::vector<MatchPair> LargestBiconnectedGraph(const std::vector<MatchPair>& pairs, std::set<size_t>& nodes);

    bool RemoveFarPoints(double threshold);

    // 把image_pairs里的图像id进行一次重新的映射，保证映射为连续的0-m个数字
    // 这是因为原本image_pairs经过了之前的各种过滤之后，只留下了较少的一部分，但是这一部分的图像id可能有很多空缺，例如
    // 图像id为 1 3 4 5 6 7 14，这样看起来id最大是14最小是1，有14个图像，但是实际上只有7个图像，因此要把它映射成为
    // 0 1 2 3 4 5 6 这种连续的映射关系, 这种映射是保顺序的，也就是原本的id为 3和5，映射之后是1和3，原本的id满足 3<5，
    // 映射后的id依然满足 1 < 3, 这种相对的大小关系是保留的
    // forward 是从老的id到新的id   key=old id  value=new id
    // backward 是从新的id到老的id  key=new id  value=old id
    void ReIndex(const std::vector<MatchPair>& pairs,
                std::map<size_t, size_t>& forward, std::map<size_t, size_t>& backward);
    // pose graph中每个长度为3的回环称为一个triplet，对于每个triplet，经过三个节点的相对旋转后，应该回到单位阵，
    // 也就是没旋转的状态。这个函数就是用来检测所有的triplet是否满足绕一圈后回到单位矩阵，如果和单位阵的差异在
    // angle_threshold以内，就可以接受，否则就过滤掉
    std::vector<MatchPair> FilterByTriplet(const std::vector<MatchPair>& init_pairs, const double angle_threshold, std::set<size_t>& covered_frames);
    
    // 在已知全局旋转的情况下重新计算图像对之间的相对平移，使用是三焦点张量 trifocal tensor 以及 AC-RANSAC方法
    // 尚未完成
    bool EstimateRelativeTwithRotation();
    // 使用LiDAR的相对位姿对图像的相对位姿进行过滤，也就是计算图像对对应的LiDAR对的相对位姿，
    // 如果LiDAR对的相对平移的角度和图像对的相对平移的角度差异不大，那就保留下来，如果差异太大就把这个图像对过滤掉
    // 这里有个问题在于LiDAR的相对位姿仅能在近邻的LiDAR间计算，如果图像对中两张图像距离较远，那么就需要“跳跃”多个LiDAR对
    // 比如图像1和图像15组成了一对，但是LiDAR-1和LiDAR-15之间距离太远，没法计算相对位姿，那么就计算LiDAR-1和LiDAR-5，
    // LiDAR-5和LiDAR-10，LiDAR-10和LiDAR-15，这样复合位姿得到LiDAR-1和LiDAR-15的相对位姿
    bool RefineRelativeTranslation();

    // 用于debug，显示当前三维点在各个图像上对应的特征点
    void VisualizeTrack(const PointTrack& track, const string path);
public:
    SfM(const Config& _config);
    // 读取图像并提取特征点，特征点默认使用了rootSIFT
    bool ReadImages(const std::vector<std::string>& image_names, const cv::Mat& mask = cv::Mat());
    // 把lidar数据投影到图像上得到稀疏深度图，然后补全得到稠密深度图
    bool ComputeDepthImage(const Eigen::Matrix4d T_cl);
    // 初始化所有的图像之间的匹配关系，仅仅是初始化，不判断匹配是否正确
    // 初始化有两种类型，如果是连续(CONTIGUOUS)，那么就会初始化为匹配从当前图像开始连续的 frame_match_size 张图像
    // 如果是全部匹配(EXHAUSIVE)，那么frame_match_size参数会被忽略，初始化为所有图像对之间都进行匹配
    bool InitImagePairs(const int frame_match_type);

    // 这部分是对初始生成的图像匹配对进行滤除，只保留稳定的匹配关系
    // 1. 根据特征点以及初始化的匹配关系进行匹配，如果匹配的特征数很少，就认为匹配关系是错误的
    // 2. 经过特征匹配后，使用匹配的特征计算本质矩阵E，如果能成功算出一个E，就认为匹配是正确的
    // 3. 然后可以得到pose graph，然后去除pose graph中不是<边双连通>(bi-edge-connected)的部分
    bool MatchImagePairs(const int matches_threshold = -1);

    bool FilterImagePairs(const int triangulation_num_threshold = 15, const float triangulation_angle_threshold = 3, const bool keep_no_scale = true);

    // 计算图像对之间的相对位姿，主要有以下几个步骤
    // 1. 在 MatchImagePairs 中已经计算了本质矩阵E，对其进行分解，得到4组R,t
    // 2. 使用每一组的相对位姿三角化，选择三角化最准确的一组相对位姿作为最终的R,t
    // 3. 非线性优化相对位姿
    // 4. 设置相对平移的尺度， keep_no_scale=true代表保留那些没有尺度的匹配对
    bool EstimateRelativePose(bool keep_no_scale=true);
    // 计算全局的旋转, 主要有以下几个步骤
    // 1. 找到所有的triplet，然后判断这个triplet的三个节点经过一圈的旋转后是否回到了没旋转的状态，并以此过滤掉不好的triplet
    // 2. 过滤triplet后，重新生成pose graph然后仅保留边双连通的部分，和MatchImagePairs的第2步一样
    // 3. 对图像id进行重映射，原因参照 ReIndex 的注释
    // 4. 计算全局的旋转，目前有两种方法
    // 4.1. 使用L2正则方法，之后还要进行ceres优化，使图像对之间的全局旋转尽可能接近他们的相对旋转
    // 4.2. 使用L1正则方法，也就是基于IRLS方法计算得到的
    bool EstimateGlobalRotation(const int method=ROTATION_AVERAGING_L1);
    // 计算全局的平移，主要有以下几个步骤
    // 1. 找到所有具有全局旋转的frame，以及这些frame对应的匹配关系
    // 2. 根据这些匹配关系找到triplet，然后生成pose graph并过滤，仅保留边双连通的部分，和MatchImagePairs的第2步一样
    // 3. 根据triplet进行再一次的相对平移计算，得到更准确的相对平移（这一步尚未完成）
    // 4. 把图像匹配对分为有尺度和无尺度的两部分，使用直接线性变换（DLT）方法对有尺度的图像对计算得到初始的全局平移
    // 5. 把所有图像对都利用起来得到更准确的全局平移
    bool EstimateGlobalTranslation(const int method=TRANSLATION_AVERAGING_SOFTL1);
    // 把图像之间的特征点连接起来得到track，然后三角化得到最初的三维点(structure)
    // todo：使用基于RANSAC的方法进行更加鲁棒的三角化
    bool EstimateStructure();
    
    /**
     * @description: 进行全局BA
     * @param residual_type 误差类型，可以是基于角度的，也可以是基于像素距离的
     * @param residual_threshold 误差阈值，误差超过这个阈值的三维点都会被删除。<0 代表不删除点
     * @param refine_structure 是否优化三维点位置
     * @param refine_rotation 是否优化相机的旋转
     * @param refine_translation 是否优化相机的平移
     * @return BA是否成功
     */    
    bool GlobalBundleAdjustment(int residual_type = RESIDUAL_TYPE::PIXEL_RESIDUAL, float residual_threshold = -1, bool refine_structure=true, 
                            bool refine_rotation=true, bool refine_translation=true);

    // 把某一帧设置为世界坐标系的原点，也就是把某一帧当做世界坐标系来使用，一般都是设置为第一帧
    /**
     * @description: 把某一帧设置为世界坐标系的原点，也就是把某一帧当做世界坐标系来使用，一般都是设置为第一帧
     * @param frame_idx 设置为世界坐标系的图像id，如果这一帧位姿不可用，则依次使用 frame_idx+1 frame_idx+2 直到成功
     * @return 是否成功
     */    
    bool SetToOrigin(size_t frame_idx = 0);
    // 对三维点云染色
    bool ColorizeStructure();
    // 保存image paris到本地文件，以txt的格式，不完全导出，只能导出基本的信息
    bool ExportMatchPairTXT(const std::string file_name);
    // 从本地文件读取image pair，以txt的格式
    bool LoadMatchPairTXT(const std::string file_name);
    // 保存image pairs到本地，以二进制的形式，保存所有数据
    // 注意：在保存前会首先清空foler内所有文件，所以一定不要保存到某些重要的位置
    bool ExportMatchPairBinary(const std::string folder);
    // 读取本地保存的image pairs，以二进制形式
    bool LoadMatchPairBinary(const std::string folder);
    // 以二进制形式保存frame到本地
    bool ExportFrameBinary(const std::string folder);
    // 从本地已有的frame数据读取，就不需要再提取特征点了，因为提特征太慢了
    bool LoadFrameBinary(const std::string& image_path, const std::string& frame_path, const bool skip_descriptor = false);
    // 把三维点云导出为pcd格式
    bool ExportStructurePCD(const std::string file_name);
    // 把三维点云导出为二进制形式，保留所有信息
    bool ExportStructureBinary(const std::string file_name);
    // 读取二进制形式的三维点云
    bool LoadStructureBinary(const std::string file_name);
    // 读取txt格式的gps数据并赋予frame
    bool LoadGPS(const std::string file_name);
    // 得到所有图像的全局旋转 R_wc, with_invalid=true代表返回的旋转中包含没有计算位姿的frame
    eigen_vector<Eigen::Matrix3d> GetGlobalRotation(bool with_invalid=false);
    // 得到所有图像的全局旋转 t_wc, with_invalid=true代表返回的平移中包含没有计算位姿的frame
    eigen_vector<Eigen::Vector3d> GetGlobalTranslation(bool with_invalid=false);
    // 得到所有图像的名字，with_invalid=true代表返回的名字中包含没有计算位姿的frame
    std::vector<std::string> GetFrameNames(bool with_invalid=false);

    const std::vector<Frame>& GetFrames() const;

    const std::vector<Velodyne>& GetLidars() const;

    void SetLidars(const std::vector<Velodyne>& _lidars);
    
    // 用于debug
    bool test();

    bool test2();

    bool test_sift();

    bool test_pipeline();

    bool test_GPS_sync();

    bool CASIA();

    /**
     * @description: 把指定范围内的图像对设置为直线运动，也就是相对旋转为单位阵，相对平移为[0,0,1]*scale。注意：这个会覆盖当前范围内的所有图像对
     * @param idx1 {size_t} 范围起始的图像idx
     * @param idx2 {size_t} 范围结束的图像idx
     * @param length {size_t} 图像对索引差异的最大值，也就是说 idx1到idx+length范围的图像设置为直线运动， idx1+1到dix1+1+length范围的图像设置为直线运动，以此类推
     * @return {*}
     */    
    bool SetToStraightMotion(size_t idx1, size_t idx2, size_t length);

    /**
     * @description: 重新计算指定范围内的图像对的相对位姿，主要是万一RANSAC结果恰好不太行
     * @param idx1 {size_t} 范围起始的图像idx
     * @param idx2 {size_t} 范围结束的图像idx
     * @return {*}
     */    
    bool ReComputePairs(size_t idx1, size_t idx2);

    /**
     * @description: 增加一个指定的图像对，这个图像对的相对位姿是依靠近邻的已有图像对计算出来的。
     *               目前尚未完成
     * @param idx1 {size_t} 图像对的第一张图像idx
     * @param idx2 {size_t} 图像对的第二张图像idx
     * @param straight_motion {bool} 是否是直线运动
     * @return {*}
     */    
    bool AddPair(size_t idx1, size_t idx2, bool straight_motion=false);

    /**
     * @description: 过滤掉所有不是直线运动的图像对，但是只删去图像idx差异在一定范围内的图像对
     *              也就是说，认为短时间内都应该是直线运动的，不满足该条件的图像对会被删除
     * @param motion_duration {int} 图像对里图像idx差异的最大值，小于这个值认为应该是直线运动
     * @return {*}
     */    
    bool FilterByStraightMotion(int motion_duration);

    /**
     * @description: 过滤掉图像idx差异在某个范围内的所有图像对，不是对所有图像对都过滤，只有图像对中两张图像在 start_idx和end_idx之间才会被过滤
     * @param min_diff {int} 图像对里图像idx差异的最小值
     * @param max_diff {int} 图像对里图像idx差异的最大值
     * @param start_idx {int} 起始的图像idx
     * @param end_idx {int} 结束的图像idx
     * @return {*}
     */    
    bool FilterByIndexDifference(int min_diff, int max_diff, int strat_idx = 0, int end_idx = 9999);

    /**
     * @description: 输出所有图像对的相对位姿，这里输出的相对位姿更符合人的直觉，适合直接观察。
     *              图像对里的相对位姿是从1到2的变换，这里输出的是2->1的变换，相当于是2在1坐标系下的位置，更符合时间顺序，因为在时间上，2是在1之后的。
     *              相对平移用三维向量表示，相对旋转用轴角表示，可以更直接的看出旋转轴和旋转的角度。
     * @param file_path {string&} 输出文件的路径
     * @return {*}
     */    
    void PrintRelativePose(const string& file_path);

    /**
     * @description: 输出所有图像的绝对位姿，同样是更符合人的直觉。旋转用的轴角
     * @param file_path {string&} 输出文件的路径
     * @return {*}
     */    
    void PrintGlobalPose(const string& file_path);

    /**
     * @description: 让整体的位姿都增加一个尺度
     * @param scale {double} 要增加的尺度
     * @return {*}
     */    
    void AddScaleToPose(double scale);

    /**
     * @description: 清空所有图像对的尺度，也就是所有图像对尺度都是1 
     * @return {*}
     */    
    void ClearPairScale();


    ~SfM();
};


#endif