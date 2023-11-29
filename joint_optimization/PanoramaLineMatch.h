/*
 * @Author: Diantao Tu
 * @Date: 2022-04-18 21:29:21
 */

#ifndef _PANORAMA_LINE_MATCH_H_
#define _PANORAMA_LINE_MATCH_H_

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <Eigen/Dense>
#include "../base/common.h"
#include "../util/PanoramaLine.h"
#include "../util/Visualization.h"
#include "../util/Tracks.h"

enum LineMatchMethod
{
    BASIC = 1,
    KNN = 2
};

class PanoramaLineMatcher 
{
private:
    const std::vector<PanoramaLine>& image_lines_all;
    // 给图像上的每一条直线都分配一个index {image_id, line_id} => index
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> feature_to_index;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> index_to_feature;
    // 特征之间的近邻关系
    std::map<uint32_t, std::vector<uint32_t>> neighbors_all;
    const eigen_vector<Eigen::Matrix3d> R_wc_list;
    const eigen_vector<Eigen::Vector3d> t_wc_list;
    // 每张图像会和之后的neighbor_size 张图像进行直线匹配
    int neighbor_size;
    // 生成的track至少包含min_track_length张图像
    int min_track_length;

    std::vector<LineTrack> tracks;

    /**
     * @description: 对初始LSD的小直线进行LBD匹配，并对匹配结果使用稀疏光流进行过滤
     * @param lines1 {PanoramaLine&} 第一张图像直线
     * @param lines2 {PanoramaLine&} 第二张图像直线
     * @param method {int} 匹配方法
     * @return 初始LSD直线之间的匹配结果
     */    
    std::vector<cv::DMatch> MatchInitLine(const PanoramaLine& lines1, const PanoramaLine& lines2, const int method);

    /**
     * @description: 对两张图像上的直线进行匹配,方法是先对图像上的初始小直线进行基于LBD的线匹配，然后对匹配结果使用稀疏光流进行过滤，
     *              得到更准确的初始直线匹配结果。由于最终的直线是通过初始直线聚合成的，因此可以通过初始直线的匹配关系判断最终直线的
     *              匹配关系。
     * @param lines1 {PanoramaLine&} 第一张图像直线
     * @param lines2 {PanoramaLine&} 第二张图像直线
     * @param method {int} 匹配方法
     * @param cross_check 对初始LSD直线匹配的时候进行交叉验证，也就是1和2匹配同时要求2和1匹配，这种匹配关系才能保留下来
     * @return 最终的长直线之间的匹配结果
     */
    std::vector<cv::DMatch> MatchPanoramaLine(const PanoramaLine& lines1, const PanoramaLine& lines2, const int method, bool cross_check=false);

    std::vector<cv::DMatch> FilterLineMatchOpticalFlow(const std::vector<cv::line_descriptor::KeyLine>& lines1, 
                                                        const std::vector<cv::line_descriptor::KeyLine>& lines2,
                                                        const cv::Mat& img_gray1, const cv::Mat& img_gray2,
                                                        const std::vector<cv::DMatch>& matches);
    // 对比两种图像直线匹配方法的差异，用于debug
    void CompareMatchMethod(const PanoramaLine& lines1, const PanoramaLine& lines2);

    /**
     * @description: 比较两个matches的差异，得到matches1中不被matches2包含的部分，用集合来说就是 matches1 / matches2，用于debug
     * @param matches1 第一个匹配集合
     * @param matches2 第二个匹配集合
     * @return 两个集合的差异
     */    
    std::vector<cv::DMatch> MatchesDiff(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2);

    /**
     * @description: 比较两个matches的差异，得到两个matches中的公共部分，也就是matches1和matches2的交集
     * @param matches1 第一个匹配集合
     * @param matches2 第二个匹配集合
     * @return 两个匹配集合的交集
     */    
    std::vector<cv::DMatch> MatchesIntersection(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2);
    
    /**
     * @description: 根据track里匹配直线的角度误差对track进行过滤，去除当前track里错误的匹配关系
     * @param angle_threshold {float} 角度误差的阈值
     * @return 经过过滤后，重新生成的track
     */    
    bool FilterTracks(const float angle_threshold);

    bool IsParallel(const std::vector<std::pair<uint32_t, uint32_t>>& features);

    /**
     * @description: 对某个track里的匹配对进行过滤
     * @param init_features 当前track里包含的匹配对，这个匹配对是 {image id，line id}构成的集合
     * @param planes_world 所有图像的所有直线在世界坐标系下形成的平面的方程
     * @param angle_threshold 过滤的角度阈值
     * @return 过滤后保留下来的匹配关系，这个匹配关系是用索引表示的，也就是 {idx1， idx2}，idx1对应于一个{image id，line id}，
     *          idx2 对应于另一个 {image id，line id}.所以输入和输出虽然都是 set<pair<uint,uint>> 但实际上代表的意思是不同的
     */    
    std::set<std::pair<uint32_t, uint32_t>> FilterPairsInTrack(const std::set<std::pair<uint32_t, uint32_t>>& init_features,
                                                                const vector<vector<cv::Vec4f>>& planes_world,
                                                                const float angle_threshold);
public:
    PanoramaLineMatcher(const std::vector<PanoramaLine>& _image_lines_all, const eigen_vector<Eigen::Matrix3d>& _R_wc_list,
                        const eigen_vector<Eigen::Vector3d>& _t_wc_list);
    /**
     * @description: 根据图像上直线的匹配关系生成track，track里的每一条图像直线都是同一个空间直线在不同图像上的投影
     * @param method {int} 图像直线匹配的方法
     * @return {*}
     */    
    bool GenerateTracks(const int method);

    // 在某些场景结构非常相似的地方，比如智能化大厦的南侧，会有大量重复且相互平行的结构，这些结构会提取出大量的相似且平行的直线
    // 某些track可能就会包含很多条这样的直线，而且当这些直线距离较远的时候，无法通过角度区分，因为角度差异很小，这种情况下就需要
    // 去除相似且平行的直线
    void RemoveParallelLines();
    
    void SetNeighborSize(const int size);

    void SetMinTrackLength(const int length);
    const std::vector<LineTrack>& GetTracks() const;

    static bool VisualizeTrack(const pair<uint32_t, set<pair<uint32_t, uint32_t>>>& track, const vector<PanoramaLine>& image_lines_all, const string path);

};


#endif