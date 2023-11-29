/*
 * @Author: Diantao Tu
 * @Date: 2021-11-22 16:47:11
 */

#include "Frame.h"
#include "../util/Visualization.h"
using namespace std;

Frame::Frame(int _rows, int _cols, int _id, const std::string _name):
    id(_id),name(_name),rows(_rows),cols(_cols),scale(0)
{
    R_wc = Eigen::Matrix3d::Zero();
    t_wc = std::numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
    gps = std::numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
}

void Frame::LoadImageGray(const std::string& path)
{
    img_gray = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    if(scale > 0)
        for(int s = scale; s > 0; s--)
            cv::pyrUp(img_gray, img_gray);
    else if(scale < 0)
        for(int s = scale; s < 0; s++)
            cv::pyrDown(img_gray, img_gray);
}

void Frame::ReleaseImageGray()
{
    if(!img_gray.empty())
        img_gray.release();
}

void Frame::LoadImageColor(const std::string& path)
{
    img_color = cv::imread(path);
    if(scale > 0)
        for(int s = scale; s > 0; s--)
            cv::pyrUp(img_color, img_color);
    else if(scale < 0)
        for(int s = scale; s < 0; s++)
            cv::pyrDown(img_color, img_color);
}

void Frame::ReleaseImageColor()
{
    if(!img_color.empty())
        img_color.release();
}

void Frame::SetPose(const Eigen::Matrix3d _R_wc, const Eigen::Vector3d _t_wc)
{
    R_wc = _R_wc;
    t_wc = _t_wc;
}

void Frame::SetPose(const Eigen::Matrix4d T_wc)
{
    R_wc = T_wc.block<3,3>(0,0);
    t_wc = T_wc.block<3,1>(0,3);
}

void Frame::SetRotation(const Eigen::Matrix3d _R_wc)
{
    R_wc = _R_wc;
}

void Frame::SetTranslation(const Eigen::Vector3d _t_wc)
{
    t_wc = _t_wc;
}

void Frame::SetImageScale(int new_scale)
{
    // 计算尺度的变化量
    int diff_scale = new_scale - scale;
    // 更新尺度
    scale = new_scale;
    if(diff_scale > 0)
    {
        for(int s = diff_scale; s > 0; s--)
        {
            rows *= 2;
            cols *= 2;
            if(!img_gray.empty())
                cv::pyrUp(img_gray, img_gray);
            if(!img_color.empty())
                cv::pyrUp(img_color, img_color);
        }
    }
    else if(diff_scale < 0)
    {
        for(int s = diff_scale; s < 0; s++)
        {
            rows = (rows + 1) / 2;
            cols = (cols + 1) / 2;
            if(!img_gray.empty())
                cv::pyrDown(img_gray, img_gray);
            if(!img_color.empty())
                cv::pyrDown(img_color, img_color);
        }
    }
    if(!img_gray.empty())
        assert(img_gray.rows == rows && img_gray.cols == cols);
    if(!img_color.empty())
        assert(img_color.rows == rows && img_color.cols == cols);
}

bool Frame::ExtractKeyPoints(int num_keypoints, const cv::Mat& mask)
{
    // 提取sift特征
    // return ExtractSIFT(img_gray, keypoints_all, num_keypoints, mask);
    // 用四叉树提取更均匀的sift特征
    return ExtractSIFTQuadtree(img_gray, keypoints_all, 0, num_keypoints, mask);
}

bool Frame::ComputeDescriptor(bool root_sift)
{
    return ComputeSIFTDescriptor(img_gray, keypoints_all, descriptor, root_sift);
}

void Frame::ReleaseDescriptor()
{
    if(!descriptor.empty())
        descriptor.release();
}

const Eigen::Matrix4d Frame::GetPose() const
{
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    T_wc.block<3,3>(0,0) = R_wc;
    T_wc.block<3,1>(0,3) = t_wc;
    return T_wc;
}

const vector<cv::KeyPoint>& Frame::GetKeyPoints() const
{
    return keypoints_all; 
}

void Frame::ReleaseKeyPoints()
{
    vector<cv::KeyPoint>().swap(keypoints_all);
}

const cv::Mat Frame::GetDescriptor() const
{
    return descriptor;
}

const cv::Mat Frame::GetImageGray() const
{
    if(!img_gray.empty())
        return img_gray;
    cv::Mat tmp = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
    if(scale > 0)
        for(int s = scale; s > 0; s--)
            cv::pyrUp(tmp, tmp);
    else if(scale < 0)
        for(int s = scale; s < 0; s++)
            cv::pyrDown(tmp, tmp);
    return tmp;
}

const cv::Mat& Frame::GetImageGrayRef() const 
{
    return img_gray;
}

const cv::Mat Frame::GetImageColor() const
{
    cv::Mat tmp = cv::imread(name);
    if(scale > 0)
        for(int s = scale; s > 0; s--)
            cv::pyrUp(tmp, tmp);
    else if(scale < 0)
        for(int s = scale; s < 0; s++)
            cv::pyrDown(tmp, tmp);
    return tmp;
}

const cv::Mat& Frame::GetImageColorRef() const
{
    return img_color;
}

const int Frame::GetImageRows() const
{
    return rows;
}

const int Frame::GetImageCols() const
{
    return cols;
}

const int Frame::GetImageScale() const 
{
    return scale;
}

const bool Frame::IsPoseValid() const
{
    if(!isinf(t_wc(0)) && !isnan(t_wc(0)) && !isinf(t_wc(1)) && !isnan(t_wc(1)) && !isinf(t_wc(2)) && !isnan(t_wc(2)) && !R_wc.isZero())
        return true;
    return false;
}

const bool Frame::IsGPSValid() const 
{
    if(!isinf(gps(0)) && !isinf(gps(1)) && !isinf(gps(2)))
        return true;
    return false;
}

void Frame::SetGPS(const Eigen::Vector3d& _gps)
{
    gps = _gps;
}

const Eigen::Vector3d& Frame::GetGPS() const
{
    return gps;
}

Frame::~Frame()
{
}