/*
 * @Author: Diantao Tu
 * @Date: 2021-11-01 10:45:31
 */

// 用于对全景图像进行直线提取，过滤，融合

#ifndef _PANORAMA_LINE_H_
#define _PANORAMA_LINE_H_

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <boost/serialization/serialization.hpp>

#include "../base/Serialization.h"
#include "../base/Geometry.hpp"
#include "../base/Config.h"
#include "../sensors/Equirectangular.h"
#include "../util/Visualization.h"


class PanoramaLine
{
private:
    std::vector<cv::Vec4f> lines;
    cv::Mat img_depth;
    cv::Mat img_gray;
    cv::Mat descriptor;
    std::string name;           // 图像保存的目录
    int rows, cols;             // 图像的尺寸
    std::vector<cv::line_descriptor::KeyLine> init_keylines;    // 初始的提取出的直线
    std::vector<std::vector<size_t>> init_to_final;
    std::vector<std::vector<size_t>> final_to_init;

    /**
     * @description: 把一组图像直线变成对应的平面参数，并且把图像上直线的起点和终点变换到球坐标系下
     * @param lines 直线的表达式
     * @param planes 直线对应的平面参数
     * @param points_in_sphere 直线的端点在球坐标系下的表达式
     * @return 无
     */    
    bool LinesToPlane(const std::vector<cv::Vec4f>& lines, std::vector<cv::Vec4f>& planes, std::vector<cv::Point3f>& points_in_sphere);
    /**
     * @description: 融合一组直线，融合成一条单独直线
     * @param line_group {vector<cv::Vec4f>} 一组直线的表达式
     * @param ransac {bool} 融合直线的方法，是否使用ransac拟合得到更好的直线,这里使用false，因为ransac效果并不好
     * @return 融合后的直线
     */    
    cv::Vec4f Fuse(std::vector<cv::Vec4f> line_group, bool ransac=false);
    // 融合离得近的线
    int FuseNearLines(bool ncc=false);
    // 融合离得远的线,也会融合一小部分重合的线
    int FuseFarLines(float dist_threshold = 600, float ncc_threshold = -0.2);
    // 融合重合的线
    int FuseOverlapLines();
    // 融合同一个物理边缘由于被展开成二维图像而分割的线
    int FuseBoundaryLines();
    // 根据近邻关系把互为近邻的那些直线选出来分为一个个group，之后每个group会被融合为一条直线
    std::vector<cv::Vec4f> FindNeighbors(std::vector<bool>& fused, std::vector<std::vector<int>> neighbor_idx, size_t line_idx);
    // 根据像素长度过滤边缘
    bool FilterByLength(float length_threshold = 100);  
    // 根据直线的起点终点与圆心的夹角过滤边缘，也就是根据扇形的圆心角过滤
    bool FilterByAngle(float angle_threshold = 10);
    // 根据长度和角度共同判断是否过滤，也就是把上面的两个相结合一下，要求至少达到两个条件之一
    // 这是因为可能有的线是向着远方延伸的，所以它在图像上的角度可能不大，但是长度很长，这种需要保留下来
    // 还有就是直线离的很近，而且是和相机前方垂直，导致角度很大，但是实际长度并不长，这种过滤掉
    // 所以要把角度阈值和长度阈值都设置的高一些
    bool FilterByLengthAngle(float length_threshold = 150, float angle_threshold = 15, cv::Mat mask = cv::Mat());
    // 在一条直线上进行多个点的采样，计算采样点之间的NCC，如果NCC都比较高，就保留这条直线，如果出现了以下这种情况
    // 直线上有一点A，在A点左侧的采样点之间的NCC都比较高，在A点右侧的采样点之间的NCC都比较高，这就说明是两条不同的空间
    // 直线恰巧处于同一平面上，因而被识别成了同一直线，对于这种情况，就以A为界把直线分割为两部分
    int FilterByNCC(const float ncc_threshold=0);
    // 根据直线对应的初始直线的数量和覆盖率来过滤直线
    // 如果当前直线包含的初始直线数量超过 count_threshold 就保留直线
    // 如果当前直线包含的初始直线能覆盖当前直线 cover_threshold 的区域，就保留直线
    bool FileterByInitLine(int count_threshold, float cover_threshold);

    cv::Mat OccupiedMatrix(int line_width=5);

    // 设置初始小直线和最终直线之间的映射关系，其实就是设置
    // init_to_final 和 final_to_init 的值
    // dis_threshold 是距离阈值，也就是初始直线和最终直线之间匹配的距离阈值，以像素为单位
    bool SetLineMap(int dis_threshold);

    // 这个方法目前还有问题，所以不要用
    cv::Vec4f FindPlaneRansac(std::vector<cv::Point3f> points);
    
    // 计算两个点处的NCC值，方法来源于colmap
    float ComputeNCC(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2f& point1, const cv::Point2f& point2, const int half_window = 21);

public:
    PanoramaLine();
    PanoramaLine(const cv::Mat& _img, int _id = -1);

    int id;

    /**
     * @description: 使用LSD方法检测图片上的直线，由于图片上下畸变太大，一般只检测中间区域
     * @param vertical_s {float} 检测区域在竖直方向上起始的角度
     * @param vertical_e {float} 检测区域在竖直方向上终止的角度
     * @return {*}
     */    
    void Detect(float vertical_s = 70.0, float vertical_e = -70.0);

    // 检测直线，使用mask进行掩模，任何起点或终点在mask内的线都会被过滤掉
    void Detect(const cv::Mat& mask );

    // 融合属于同一个物理边缘的不同直线
    int Fuse(float ncc_threshold, bool visualization=false);   

    // 设置深度图
    void SetDepthImage(const cv::Mat& depth);
    // 设置灰度图
    void SetImageGray(const cv::Mat& gray);
    // 设置图像路径
    void SetName(const std::string& _name);
    // 得到灰度图
    const cv::Mat GetImageGray() const;  
    // 得到最终的直线
    const std::vector<cv::Vec4f>& GetLines() const;
    // 得到初始的检测的直线
    const std::vector<cv::line_descriptor::KeyLine>& GetInitLines() const;
    // 得到初始直线对应的描述子
    const cv::Mat& GetLineDescriptor() const;
    // 释放描述子所占空间
    void ReleaseDescriptor();
    // 得到初始直线到最终直线的匹配
    const std::vector<std::vector<size_t>>& GetInitToFinal() const;
    // 得到最终直线到初始直线的匹配
    const std::vector<std::vector<size_t>>& GetFinalToInit() const;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & rows;
        ar & cols;
        ar & name;
        ar & lines;
        ar & init_keylines;
        ar & init_to_final;
        ar & final_to_init;
        ar & descriptor;
    }
    ~PanoramaLine();
};



#endif