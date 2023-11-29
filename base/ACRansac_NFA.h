/*
 * @Author: Diantao Tu
 * @Date: 2021-11-30 16:25:54
 */
 
#ifndef _ACRANSAC_NFA_H
#define _ACRANSAC_NFA_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <assert.h>

class ACRansac_NFA
{
private:
    size_t sample_size;
    std::vector<double> residuals;
    std::vector<double> log_c_n, log_c_k;
    double log_e0;
    const bool quantified_evaluation;
    const double max_threshold;
    const size_t minimum_sample;
    double log_alpha0;
    double mult_error;

public:
    ACRansac_NFA(const size_t _sample_size, const size_t minimum_sample_size, const bool _quantified,
                const double _max_threshold = std::numeric_limits<double>::infinity());
    
    /**
     * @description: 根据误差 residual 计算NFA
     * @param inlier_idx 保存符合当前NFA的内点的索引
     * @param nfa 计算得到的nfa
     * @return 是否有足够多的内点
     */    
    bool ComputeNFA(std::vector<size_t>& inlier_idx, std::pair<double,double>& nfa);

    // 注意，这里的residual一定是所有匹配的误差，而不能仅仅传递内点的误差
    void SetResidual(const std::vector<double>& _residuals);

    void SetSampleSize(const size_t& _num_sample);
    ~ACRansac_NFA();
};


#endif



