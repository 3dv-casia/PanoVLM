/*
 * @Author: Diantao Tu
 * @Date: 2022-06-28 17:31:08
 */
#ifndef _MVS_H_
#define _MVS_H_

#include <vector>
#include <set>
#include <glog/logging.h>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/ximgproc.hpp>

#include "../base/common.h"
#include "../base/Math.h"
#include "../base/Config.h"
#include "../base/Geometry.hpp"
#include "../base/ProcessBar.h"

#include "../sensors/Frame.h"
#include "../sensors/Velodyne.h"

#include "../util/Tracks.h"
#include "../util/FileIO.h"
#include "../util/DepthCompletion.h"
#include "../util/Optimization.h"

#include "../sfm/Structure.h"

enum NeighborSelection
{
    SFM_POINTS = 1,
    NEAREST_NEIGHBOR = 2    // 这个更好用，因为是全景图像
};

enum Propagate
{
    CHECKER_BOARD = 1,      // 按照棋盘格传播
    SEQUENTIAL = 2          // 从左上到右下顺序传播
};

// 邻域图像信息
struct NeighborInfo
{
    size_t id;
    cv::Matx33f R_nr;   // reference to neighbor 
    cv::Vec3f t_nr;     // reference to neighbor
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NeighborInfo(const size_t _id, const Eigen::Matrix3d& R, const Eigen::Vector3d& t):id(_id)
    {
        R_nr = cv::Matx33f(R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2), R(2,0), R(2,1), R(2,2));
        t_nr = cv::Vec3f(t(0), t(1), t(2));
    }
};

struct NeighborPixel
{
    cv::Point3f point;
    cv::Vec3f normal;
    float depth;
};


class MVS
{
private:
    float scaleRanges[12] = {1.f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f, 0.00048828125f};

    std::vector<Frame> frames;              // 所有的图像信息
    std::vector<Velodyne> lidars;           // 所有的雷达信息
    std::vector<PointTrack> structure;      // SfM的三维点
    std::vector<std::vector<NeighborInfo>> neighbors;   // 每张图像的近邻图像信息
    Equirectangular eq;
    Config config;
    int ncc_window_size;
    int num_texels;
    float sigma_color;
    float sigma_spatial;
    float smoothBonus = 0.95f;                   // NCC平滑的程度，1代表不使用平滑，0代表极度平滑
    float smoothBonusDepth = 1.f - smoothBonus;       
    float smoothBonusNormal = (1.f - smoothBonus) * 0.96;
    float smoothSigmaDepth = -1.f / (2.f * 0.02f * 0.02f);
    float smoothSigmaNormal = -1.f / (2.f * 0.22f * 0.22f);
    cv::RNG rng;

    // 下面的这几个变量只在顺序传播的时候起作用，主要是用于防止图像重复加载或者错误释放导致的问题
    // 比如1和2 3近邻，2和 1 4近邻，那么在读取图像的时候，1作为参考图像的时候要加载1,2,3的灰度图，2作为参考图像的时候要加载1,2,4的灰度图
    // 然而这两个线程是同时进行的，就会导致1,2被加载了两次，浪费时间和内存，所以在加载图像的时候使用互斥量，防止重复加载。这是第一个作用。
    // 当图像1完成深度估计后就要释放自己的灰度图，节省内存，但此时图像2还没完成深度图计算，它需要图像1的信息。那么此时图像1就不能释放，
    // 需要等到图像2也完成了计算才能释放。这时就需要一个图像引用计数的功能，每产生一次引用，就+1.对于当前情况来说，图像1和2的引用数都是2，
    // 图像3和4的引用数量是1。当1完成计算后，把1,2,3的引用各减少1，此时可以释放图像3，但不能释放图像1和2，因为它们的引用不是0.
    std::vector<std::mutex> frame_mutex;   
    std::vector<int> frame_gray_count;
    std::vector<int> frame_conf_count;
    std::vector<int> frame_depth_count;
    std::vector<int> frame_normal_count;
    std::vector<int> frame_depth_filter_count;

    void ResetFrameCount();

    /**
     * @description: 根据相机位姿选择最近的几张图像作为近邻
     * @param neighbor_size {int} 近邻数量
     * @param sq_distance_threshold 近邻之间最小距离的平方
     * @return 是否成功
     */    
    bool SelectNeighborKNN(int neighbor_size, float sq_distance_threshold = 0.f);
    /**
     * @description: 根据SfM点在各个图像上的可视角度来确定每张图像的近邻
     * @param neighbor_size {int} 近邻数量
     * @param sq_distance_threshold 近邻之间最小距离的平方
     * @return 是否成功
     */    
    bool SelectNeighborSFM(int neighbor_size, float sq_distance_threshold = 0.f);
   
    /**
     * @description: 初始化当前图像的深度图和法向量图
     * @param ref_id {int} 当前图像id
     * @param mask {Mat&} 掩模
     * @param use_lidar {bool} 是否使用雷达信息
     * @return 初始化是否成功
     */
    bool InitDepthNormal(int ref_id, const cv::Mat& mask, bool use_lidar = true);

    /**
     * @description: 初始化置信度图
     * @param ref_id {int} 当前图像id
     * @param enable_parallel {bool} 是否开始并行计算
     * @param use_geometry {bool} 是否使用几何一致性计算NCC
     * @return {*}
     */    
    bool InitConfMap(int ref_id, bool enable_parallel, bool use_geometry);

    bool InitPatchMap(int ref_id, bool enable_parallel);

    bool FillPixelPatch(int ref_id, const cv::Point2i& pt, PixelPatch& patch);

    /**
     * @description: 计算在ref_id这帧图像的pt点以normlal为法向量以depth为深度的时候，得到的评分
     * @param ref_id {int} 参考图像的id
     * @param pt {Point2i&} 像素点坐标
     * @param normal {Vec3f&} 当前点的法向量
     * @param depth {float} 当前点的深度
     * @param geometric_consistency 是否使用邻域图像的几何一致性来平滑NCC计算
     * @param score_neighbor 当前像素的深度和法向量投影到每个邻域上的得分
     * @param plane 平面参数 a b c d，用于NCC平滑
     * @param close_neighbors 距离较近的像素信息，用于NCC平滑
     * @return 评分
     */
    float ScorePixel(int ref_id, const cv::Point2i& pt, const cv::Vec3f& normal, const float depth, 
                    const PixelPatch& patch_info,
                    const bool geometric_consistency = false,
                    std::vector<std::pair<float,int>>& score_neighbor = * (new std::vector<std::pair<float,int>>()),
                    const cv::Vec4f& plane = cv::Vec4f(0,0,0),
                    const std::vector<NeighborPixel>& close_neighbors = std::vector<NeighborPixel>()
                    );
    
    float ScorePixelSphere(int ref_id, const cv::Point2i& pt, const cv::Vec3f& normal, const float depth, 
                    std::vector<std::pair<float,int>>& score_neighbor = * (new std::vector<std::pair<float,int>>()),
                    const cv::Vec4f& plane = cv::Vec4f(0,0,0),
                    const std::vector<NeighborPixel>& close_neighbors = std::vector<NeighborPixel>()
                    );
    
    bool ProcessPixel(int ref_id, const cv::Point2i& pt, const vector<cv::Point2i>& neighbor, 
                        const PixelPatch& patch,
                        const bool geometric_consistency
                        );
    
    /**
     * @description: 以顺序的方式进行传播，也就是从左上->右下，从右下->左上这样
     * @param ref_id {int} 图像id
     * @param iter 当前的迭代次数，用于决定是从左上到右下，或是反过来
     * @param geometric_consistency 计算NCC的时候是否使用几何一致性
     * @return {*} 是否成功
     */    
    bool PropagateSequential(int ref_id, const int iter, const bool geometric_consistency = false);

    /**
     * @description: 以棋盘格的形式进行传播，也就是把图像分成棋盘格的样子，一次传播黑色格子，一次传播白色格子
     * @param ref_id {int} 图像id
     * @param geometric_consistency 计算NCC的时候是否使用几何一致性
     * @return {*} 是否成功
     */    
    bool PropagateCheckerBoard(int ref_id, const bool geometric_consistency = false);

    /**
     * @description: 对像素点的深度值和法向量进行一定程度的扰动，得到更好的结果。这个扰动方式是参考colmap来的
     * @param ref_id {int} 图像id
     * @param pt {Point2i&} 像素位置
     * @param geometric_consistency 计算NCC的时候是否使用几何一致性
     * @return {*} 扰动是否成功
     */    
    bool PerturbDepthNormal(int ref_id, const cv::Point2i& pt, const bool geometric_consistency = false);
    
    // 这个扰动是参考OpenMVS来的，但是没有使用平面平滑
    bool PerturbDepthNormal2(int ref_id, const cv::Point2i& pt, const bool geometric_consistency = false);
    
    // 这个扰动是参考OpenMVS来的，使用了平面平滑，和OpenMVS本身的扰动一样
    bool PerturbDepthNormal3(int ref_id, const cv::Point2i& pt, const PixelPatch& patch, const vector<NeighborPixel>& neighbor_pixels, 
                            const bool geometric_consistency = false, const bool perturb_depth = true, const bool perturb_normal = true);
    
    /**
     * @description: 从当前点周围按照棋盘格的样式选择出一定数量的近邻点，用于传播法向量和深度
     * @param frame {Frame&} 当前的图像
     * @param pt {Point2i&} 当前点
     * @param neighbor_size {int} 近邻数量
     * @return 近邻的坐标
     */    
    std::vector<cv::Point2i> CheckerBoardSampling(const Frame& frame, const cv::Point2i& pt, const int neighbor_size);
   
    /**
     * @description: 通过双线性插值对目标点进行灰度采样
     * @param img_gray {Mat&} 灰度图
     * @param pt {Point2f} 目标点
     * @return 目标点的灰度值
     */
    float Sample(const cv::Mat& img_gray, const cv::Point2f& pt);
    
    template<typename Functor>
    float Sample(const cv::Mat& img, const cv::Point2f& pt, const Functor& functor);

    /**
     * @description: 对输入的法向量进行随机扰动
     * @param normal {Vec3f&} 初始法向量
     * @param perturbation {float} 扰动程度，以弧度为单位
     * @return {*} 扰动后的法向量
     */    
    cv::Vec3f PerturbNormal(const cv::Vec3f& normal, const float perturbation);

    /**
     * @description: 对输入的深度进行随机扰动
     * @param depth {float} 初始深度
     * @param perturbation {float} 扰动程度
     * @return 扰动后的深度
     */    
    float PerturbDepth(const float depth, const float perturbation);

    cv::Vec3f GenerateRandomNormal(const cv::Point2i& pt);

    /**
     * @description: 每次估计完深度图之后进行后续的内存释放操作
     * @param ref_id {int} 参考图像id
     * @param geometric {bool} 当前阶段是否使用了几何一致性的NCC
     */
    void FinishEstimation(int ref_id, bool geometric);

    void RemoveSmallSegments(int ref_id);
    
    // 从OpenMVS抄的，用于填补深度图中小的空洞
    void GapInterpolation(int ref_id);
    
    /**
     * @description: 从OpenMVS抄的，用于插值邻居像素的深度到当前像素，具体代码理解见"公式推到.md"中的"近邻深度插值"部分
     * @param pt {Point2i&} 当前像素
     * @param nx {Point2i&} 邻居像素
     * @param depth {float&} 邻居像素深度
     * @param normal {Vec3f&} 邻居像素法向量
     * @return {*} 插值后的深度
     */    
    float InterpolatePixel(const cv::Point2i& pt, const cv::Point2i& nx, const float& depth, const cv::Vec3f& normal);

    // 和上一个一样的结果，只是这个版本理解起来更简单，但是计算速度更慢
    float InterpolatePixel2(const cv::Point2i& pt, const cv::Point2i& nx, const float& depth, const cv::Vec3f& normal);

    void CorrectNormal(const cv::Point2i& pt, cv::Vec3f& normal);

    /**
     * @description: 把当前图像的深度图投影到邻域图像上
     * @param ref_id {int} 当前图像id
     * @param info {NeighborInfo&} 邻域图像信息
     * @return {*} 投影后的深度图
     */
    cv::Mat ProjectDepthToNeighbor(int ref_id, const NeighborInfo& info);

    /**
     * @description: 把邻域图像的深度图和置信度图投影到参考图像上
     * @param ref_id 参考图像id
     * @param info 邻域图像信息
     * @param project_depth 是否投影深度
     * @param use_filtered_depth 是否使用经过过滤后的深度图
     * @param depth_projected 投影后的深度图
     * @param project_conf 是否投影置信度图
     * @param conf_projected 投影后的置信度图
     * @return {*}
     */    
    void ProjectDepthConfToRef(int ref_id, const NeighborInfo& info, 
                        bool project_depth = true, bool use_filtered_depth = true, cv::Mat& depth_projected = *(new cv::Mat()), 
                        bool project_conf = false, cv::Mat& conf_project = *(new cv::Mat()));

    /**
     * @description: 把深度图变成点云
     * @param ref_id {int} 参考图像id
     * @param use_filtered_depth {bool} 是否使用经过过滤后的深度图
     * @return {*} 点云
     */    
    pcl::PointCloud<pcl::PointXYZRGB> DepthImageToCloud(int ref_id, bool use_filtered_depth = true);

    /**
     * @description: 把深度图和法向量图变成点云
     * @param ref_id {int} 参考图像id
     * @param use_filtered_depth {bool} 是否使用经过过滤后的深度图
     * @return {*} 点云
     */    
    pcl::PointCloud<pcl::PointXYZRGBNormal> DepthNormalToCloud(int ref_id, bool use_filtered_depth = true);

    float ConfToWeight(const float& conf, const float& depth);

    void ConvertNCC2Conf(cv::Mat& conf);

    cv::Mat ExtractGroundPixel(const cv::Mat& ground_depth);

    /**
     * 以下几个函数都是用来debug的
    */
    cv::Mat ProjectRGBToNeighbor(int ref_id, const NeighborInfo& info, bool use_filtered_depth = true);
    

public:
    MVS(const std::vector<Frame>& _frames, const std::vector<Velodyne>& _lidars, const Config& _config);

    bool EstimateStructure();
    /**
     * @description: 为每一张图像选择近邻图像
     * @param neighbor_size {int} 近邻的数量
     * @param method {int} 选择近邻图像的方法
     * @param min_distance 近邻图像之间的最近距离，图像之间离得太近可能会导致深度计算不太准确
     * @return 是否成功
     */    
    bool SelectNeighborViews(int neighbor_size, int method = NeighborSelection::NEAREST_NEIGHBOR, float min_distance = 0.f);

    /**
     * @description: 为所有图像计算深度图
     * @param method {int} 邻域像素传播的方法
     * @param mask {Mat&} 
     * @return {*} 是否成功
     */    
    bool EstimateDepthMaps(int method, const cv::Mat& mask);

    /**
     * @description: 对深度图进行过滤
     * @return {*} 是否成功
     */    
    bool FilterDepthMaps();

    /**
     * @description: 对深度图进行融合，得到三维模型 
     * @return {*} 是否成功
     */    
    bool FuseDepthMaps();
    
    /**
     * @description: 顺序传播时的初始化过程，包括初始化深度图、法向量图、置信度图
     * @param ref_id {int} 要初始化的图像id
     * @param mask {Mat&} 掩模，被掩模的部分不初始化，在之后的流程中也直接略过
     * @param enable_parallel 初始化过程中是否使用多线程计算
     * @param use_lidar 初始化过程中是否使用雷达信息
     * @param use_geometry 计算NCC的时候是否使用几何一致性
     * @return {*} 初始化是否成功
     */    
    bool Initialize(int ref_id, const cv::Mat& mask, const bool enable_parallel, const bool use_lidar = true, const bool use_geometry = false);

    bool RefineCameraPose();

    /**
     * @description: 计算深度图
     * @param ref_id {int} 图像id
     * @param propagate {int} 传播方式，顺序传播或棋盘格传播
     * @param max_iter 迭代次数上限
     * @param conf_threshold 过滤的置信度阈值
     * @return {*} 是否成功
     */
    bool EstimateDepthMapSingle(int ref_id, int propagate, int max_iter, float conf_threshold, bool use_geometry);

    /**
     * @description: 对深度图进行过滤，主要是根据几何一致性
     * @param ref_id {int} 图像id
     * @return {*} 是否成功
     */
    bool FilterDepthImage(int ref_id);

    bool FilterDepthImageRefine(int ref_id);

    bool FilterDepthImage2(int ref_id);

    bool LoadStructure(std::string file_name);

    pcl::PointCloud<pcl::PointXYZRGB> MergeDepthImages(int skip = 2, bool use_filtered_depth = true);

    pcl::PointCloud<pcl::PointXYZRGB> FuseDepthImages(bool use_filtered_depth = true);

    const std::vector<std::vector<NeighborInfo>>& GetNeighbors() const;

    // 得到所有图像的全局旋转 R_wc, with_invalid=true代表返回的旋转中包含没有计算位姿的frame
    eigen_vector<Eigen::Matrix3d> GetGlobalRotation(bool with_invalid=false);
    // 得到所有图像的全局旋转 t_wc, with_invalid=true代表返回的平移中包含没有计算位姿的frame
    eigen_vector<Eigen::Vector3d> GetGlobalTranslation(bool with_invalid=false);
    // 得到所有图像的名字，with_invalid=true代表返回的名字中包含没有计算位姿的frame
    std::vector<std::string> GetFrameNames(bool with_invalid=false);
    
    ~MVS();

    /* 以下的函数用于测试和debug */

    void test(const std::set<int>& ids, const cv::Mat& mask);

    void test2(const cv::Mat& mask);

    void test3(const std::set<int>& ids, const cv::Mat& mask);

    void test_ground(const std::set<int>& ids, const cv::Mat& mask);

    void FuseLidarDepth(const std::set<int>& ids, const cv::Mat& mask);

    static cv::Mat ExtractImagePatch(const cv::Mat& src_image, const int row, const int col, int half_row, int half_col);

    void PrintPatch(const PixelPatch& patch);
};

#endif