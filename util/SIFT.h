/*
 * @Author: Diantao Tu
 * @Date: 2021-11-21 20:25:08
 */


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <numeric>

#ifdef USE_CUDA
#include <opencv2/cudafeatures2d.hpp>
#endif

struct Cell
{
    cv::Point2i left_top;           // 左上角像素坐标
    cv::Point2i right_bottom;       // 右下角像素坐标，注意右下角是包含在这个cell内的
    std::vector<int> keypoint_ids;  // 当前cell里包含的keypoint的id
    bool valid;                     // 用来指示当前cell是否可用
    Cell(const cv::Point2i& _left_top, const cv::Point2i& _right_bottom):
        left_top(_left_top), right_bottom(_right_bottom), valid(true)
    {}
    Cell(const int x1, const int y1, const int x2, const int y2):
        valid(true)
    {
        left_top = cv::Point2i(x1, y1);
        right_bottom = cv::Point2i(x2, y2);
    }
    std::vector<Cell> SplitIntoFour(const std::vector<cv::KeyPoint>& key_points)
    {
        std::vector<Cell> sub_cells;
        // 每个cell最小就是30像素长宽，再小就不能分割了
        if(right_bottom.x - left_top.x < 10 || right_bottom.y - left_top.y < 10 || !valid)
            return sub_cells;
        cv::Point2i center = (left_top + right_bottom) / 2;
        
        sub_cells.push_back(Cell(left_top, center));                                    // 左上角
        sub_cells.push_back(Cell(center.x + 1, left_top.y, right_bottom.x, center.y));  // 右上角
        sub_cells.push_back(Cell(left_top.x, center.y + 1, center.x, right_bottom.y));  // 左下角
        sub_cells.push_back(Cell(center + cv::Point2i(1,1), right_bottom));             // 右下角
        for(const int& id : keypoint_ids)
        {
            const cv::Point2f& pt = key_points[id].pt;
            if(pt.x > center.x)
            {
                if(pt.y <= center.y)
                    sub_cells[1].keypoint_ids.push_back(id);
                else 
                    sub_cells[3].keypoint_ids.push_back(id);
            }
            else 
            {
                if(pt.y <= center.y)
                    sub_cells[0].keypoint_ids.push_back(id);
                else 
                    sub_cells[2].keypoint_ids.push_back(id);
            }
        }
        for(Cell& c : sub_cells)
            c.valid = !c.keypoint_ids.empty();
        // 当前cell已经被分成四个了，那么当前cell就要被标记为不可用
        valid = false;
        keypoint_ids.clear();
        return sub_cells;
    }
    template<typename T>
    inline bool IsInside(cv::Point_<T> pt)
    {
        return std::round(pt.x) >= left_top.x && std::round(pt.x) <= right_bottom.x && 
                std::round(pt.y) >= left_top.y && std::round(pt.y) <= right_bottom.y;
    }
};

// Three things everyone should know to improve object retrieval - CVPR 2012
void RootSIFT(cv::Mat& descriptor);

/**
 * @description: 提取SIFT特征
 * @param img_gray 8-bit灰度图
 * @param keypoints 提取出的特征点
 * @param num_sift 希望提取的sift特征点数量
 * @param mask 掩模，mask=0的区域不会用于提取特征点
 * @param root_sift 是否使用root sift，来自于 Three things everyone should know to improve object retrieval - CVPR2012
 * @return 是否成功提取特征点
 */
bool ExtractSIFT(const cv::Mat& img_gray, std::vector<cv::KeyPoint>& keypoints, 
                const int num_sift = 8096, const cv::Mat& mask = cv::Mat());

// 使用四叉树进行均匀的特征分散，参数和上一个都一样
// 也就是先在图像上提取特征，然后把图像分成一个个小方格，在每个小方格上只保留一个相应最大的特征点
// 这个想法是从ORB-SLAM里借鉴的
bool DistributeSIFTQuadtree(const cv::Mat& img_gray, std::vector<cv::KeyPoint>& keypoints, 
                const int num_sift = 8096, const cv::Mat& mask = cv::Mat());

// 使用四叉树进行均匀的特征提取
// 先把图像分成一个个小方格，然后在各个小方格上提取特征
// tree_depth 四叉树的深度，0代表只分成一个方格，也就是退化成ExtractSIFT形式
// depth = 1 代表分成4个方格， depth = 2 代表分成 16 个方格
bool ExtractSIFTQuadtree(const cv::Mat& img_gray, std::vector<cv::KeyPoint>& keypoints, 
                const int tree_depth = 1, const int num_sift = 8096,
                const cv::Mat& mask = cv::Mat());

/**
 * @description: 根据输入的特征点提取sfit描述子
 * @param img_gray {Mat&} 灰度图
 * @param keypoints 已经提取的sift特征点
 * @param descriptor 描述子组成的矩阵，每行是一个描述子
 * @param root_sift {bool} 是否使用root sift，来自于 Three things everyone should know to improve object retrieval - CVPR2012
 * @return 是否成功
 */
bool ComputeSIFTDescriptor(const cv::Mat& img_gray, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor, const bool root_sift = false);

/**
 * @description: SIFT特征匹配
 * @param descriptor1 第一张图的描述子
 * @param descriptor2 第二张图的描述子
 * @param dist_ratio_threshold 最近邻与次近邻的距离之比的阈值
 * @return 第一张到第二张特征的匹配结果
 */
std::vector<cv::DMatch> MatchSIFT(const cv::Mat& descriptor1, const cv::Mat& descriptor2, const float dist_ratio_threshold);

#ifdef USE_CUDA
// 使用CUDA进行描述子匹配
std::vector<cv::DMatch> MatchSIFT(const cv::cuda::GpuMat& descriptor1, const cv::cuda::GpuMat& descriptor2, const float dist_ratio_threshold);
#endif