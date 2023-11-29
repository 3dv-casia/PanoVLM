/*
 * @Author: Diantao Tu
 * @Date: 2021-11-19 13:34:53
 */

#ifndef _FRAME_H_
#define _FRAME_H_

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <glog/logging.h>
#include <boost/serialization/serialization.hpp>

#include "../base/Serialization.h"
#include "../util/DepthCompletion.h"
#include "../util/SIFT.h"

// MVS中计算NCC时存储以某个像素为中心的patch信息，用于加速NCC计算
struct PixelPatch
{
    std::vector<float> texels0;
    std::vector<float> weight;
    float sq0;

    PixelPatch(int num_texels):sq0(0)
    {
        texels0.resize(num_texels);
        weight.resize(num_texels);
    }

    ~PixelPatch()
    {
        std::vector<float>().swap(texels0);
        std::vector<float>().swap(weight);
    }
};

typedef std::vector<PixelPatch> PatchMap;


class Frame
{
private:
    int scale;          // 图像的尺度，修改图像尺寸，0代表原尺寸，1代表放大1倍，-1代表缩小1倍
    int rows;           // 图像的行数
    int cols;           // 图像的列数
    cv::Mat img_gray;   // 灰度图，绝大部分情况下这个都是空的，因为存全景图像太占内存了
    cv::Mat img_color;  // 彩色图，绝大部分情况下都是空的
    
    
    Eigen::Matrix3d R_wc;
    Eigen::Vector3d t_wc;
    Eigen::Vector3d gps;        // GPS 数据，保存 XYZ

    // SIFT 特征提取相关
    std::vector<cv::KeyPoint> keypoints_all;    // 特征点
    cv::Mat descriptor;     // 描述子

public:
    int id;
    std::string name;
    cv::Mat conf_map;       // 置信度图，用于MVS
    cv::Mat normal_map;     // 法向量图，用于MVS
    cv::Mat depth_map;      // 深度图，用于MVS和SfM，MVS中以CV_32F存储数据，SfM中以CV_16U存储数据
    cv::Mat depth_filter;
    cv::Mat depth_constant; // 用于MVS中设置某些点的深度为固定不变的（法向量可变）
    PatchMap patch_map;

    // 用于debug
    std::map<int, std::vector<std::pair<float,int>>> score_each_pixel;


    Frame(int _rows, int _cols, int _id, const std::string _name);

    /**
     * @description: 从图像上提取SIFT特征点
     * @param num_keypoints {int} 提取特征点的数量
     * @param mask {Mat&} 掩模，在掩模区域内不提取特征
     * @param root_sift {bool} 是否使用root sift
     * @return 提取是否成功
     */    
    bool ExtractKeyPoints(int num_keypoints = 8096, const cv::Mat& mask = cv::Mat());
    
    bool ComputeDescriptor(bool root_sift);

    const std::vector<cv::KeyPoint>& GetKeyPoints() const;

    // 释放特征点所占的内存
    void ReleaseKeyPoints();

    const cv::Mat GetDescriptor() const;
    // 释放描述子所占的内存
    void ReleaseDescriptor();
    // 设置位姿
    void SetPose(const Eigen::Matrix3d _R_wc, const Eigen::Vector3d _t_wc);
    void SetPose(const Eigen::Matrix4d T_wc);
    void SetRotation(const Eigen::Matrix3d _R_wc);
    void SetTranslation(const Eigen::Vector3d _t_wc);

    void SetImageScale(int s);

    // 获取的是T_wc
    const Eigen::Matrix4d GetPose() const;

    // 是否有特征点
    bool HasKeyPoints();

    // 从指定路径读取灰度图并保存在frame的内存中
    void LoadImageGray(const std::string& path);

    // 释放frame中灰度图占用的空间
    void ReleaseImageGray();

    // 获取的是灰度图的内存拷贝，如果当前frame中没有保存灰度图的话，会直接从文件中读取
    // 因此速度比较慢，但优势在于节约内存，frame本身不保存灰度图
    // 如果只是偶尔使用一次灰度图的话，这个比较好
    const cv::Mat GetImageGray() const;

    // 获取的是灰度图的引用，如果当前frame中没有保存灰度图的话，返回的引用就是空的
    // 所以在调用这个函数之前必须先使用 LoadImageGray 来设置灰度图
    // 这个速度很快，但是会占用内存，因为在frame中保存了灰度图，经常使用灰度图可以用这个函数
    const cv::Mat& GetImageGrayRef() const;

    // 同 LoadImageGray
    void LoadImageColor(const std::string& path);

    // 同 ReleaseImageColor
    void ReleaseImageColor();

    // 这个和 GetImageGray 一样，直接从本地读取一张彩色图返回
    const cv::Mat GetImageColor() const;

    // 获取的是彩色图像的引用，如果当前frame没有保存彩色图，则返回的引用为空
    // 调用函数前必须先使用 LoadImageColor 来设置彩色图
    const cv::Mat& GetImageColorRef() const;
    
    const int GetImageRows() const;

    const int GetImageCols() const;

    const int GetImageScale() const;

    const bool IsPoseValid() const;

    const bool IsGPSValid() const;

    void SetGPS(const Eigen::Vector3d& _gps);

    const Eigen::Vector3d& GetGPS() const;

    template<typename T>
    inline bool IsInside(const cv::Point_<T>& pt, int row_margin = 0, int col_margin = 0) const
    {
        return pt.x >= col_margin && pt.y >= row_margin && pt.x < cols - col_margin && pt.y < rows - row_margin;
    }

    template<typename T>
    inline bool IsInside(const Eigen::Matrix<T, 2, 1>& pt, int row_margin = 0, int col_margin = 0) const 
    {
        return pt.x() >= col_margin && pt.y() >= row_margin && pt.x() < cols - col_margin && pt.y() < rows - row_margin;
    }

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & rows;
        ar & cols;
        ar & keypoints_all;
        ar & id;
        ar & name;
        ar & descriptor;
        ar & R_wc;
        ar & t_wc;
        // ar & gps;
    }

    ~Frame();
};







#endif