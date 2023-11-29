/*
 * @Author: Diantao Tu
 * @Date: 2022-04-29 15:30:24
 */

#ifndef _VLAD_H_
#define _VLAD_H_

#include <glog/logging.h>
#include <omp.h>
#include <numeric>
#include "../sensors/Frame.h"
#include "../base/Random.hpp"
#include "../base/Config.h"

enum VLAD_NORMALIZATION
{
    SIGNED_SQUARE_ROOTING = 0,      // Element wise power law normalization with alpha = 0.5 (SSR)
                                    // Aggregating local descriptors into compact codes  PAMI-2012
    INTRA_NORMALIZATION = 1,        // cluster wise L2 normalization, the so called "intra normalization"
                                    // All About VLAD.  CVPR-2013.
    RESIDUAL_NORMALIZATION_PWR_LAW = 2  // Per residual L2 normalization (RN)
                                        // followed by and element wise power law normalization with alpha = 0.2
                                        // Revisiting the VLAD image representation    ACM Multimedia-2013
};

class VLADMatcher
{
private:
    const std::vector<Frame>& frames;
    cv::Mat codebook;
    cv::Mat vlad_descriptors;
    int normalization_type;
    const Config& config;

    /**
     * @description: 使用Kmeans方法进行聚类
     * @param data {Mat&} 输入的数据，要求每一行是一个数据
     * @param num_center {int} 聚类的中心的数量
     * @param max_iteration {int} 最大迭代的次数
     * @param data_to_center 每个数据是属于哪个聚类中心的
     * @return 聚类中心，每一行是一个中心
     */    
    cv::Mat Kmeans(const cv::Mat& data, const size_t num_center, const int max_iteration, std::vector<size_t>& data_to_center);

public:
    VLADMatcher(const std::vector<Frame>& _frames, const Config& _config = *(new Config()),
                const int _normalization_type = RESIDUAL_NORMALIZATION_PWR_LAW);
    
    /**
     * @description: 生成VLAD的codebook，使用的是Kmeans算法
     * @param ratio {float} 学习codebook时使用的描述子的比例，1代表使用所有图像的所有描述子
     * @param book_size {int} codebook的聚类中心数量
     * @param max_iteration {int} 聚类时最大的迭代次数
     * @return {*}
     */    
    bool GenerateCodeBook(float ratio, const int book_size=128, const int max_iteration=25);

    // 计算每个frame的VLAD描述子
    bool ComputeVLADEmbedding();

    // 对每个frame根据VLAD描述子找到最相近的图像
    std::vector<std::vector<size_t>> FindNeighbors(int neighbor_size);

};



#endif