/*
 * @Author: Diantao Tu
 * @Date: 2021-11-30 20:45:30
 */

#include "ACRansac_NFA.h"

using namespace std;
ACRansac_NFA::ACRansac_NFA(const size_t _sample_size, const size_t minimum_sample_size, const bool _quantified,
                const double _max_threshold)
                :quantified_evaluation(_quantified),sample_size(_sample_size),minimum_sample(minimum_sample_size),
                max_threshold(_max_threshold)
{
    assert(sample_size >= minimum_sample_size);
    residuals.resize(sample_size);
    log_e0 = log10(1.0 * static_cast<double>(sample_size - minimum_sample_size));
    log_alpha0 = log10(0.5);
    // 根据之后的代码，这里的mult_error就是计算NFA公式里的d，但为什么是0.25，openMVG里没有说明
    // 同时openMVG里point-to-point里的multError=1，point-to-line里的multError=0.5，都和原本论文中相比
    // 少了一半，没想明白为什么
    mult_error = 0.25;    
    // 预先计算各个数字的对数
    vector<double> log10_table(sample_size + 1);
    log10_table[0] = 0;
    for(int i = 1; i < log10_table.size(); i++)
        log10_table[i] = log10(i);
    // 计算 C_k^Sample   从k里面选Sample个
    for(int k = 0; k < minimum_sample + 1; k++)
        log_c_k.push_back(0);      
    for(int k = minimum_sample + 1; k <= sample_size; k++)
        log_c_k.push_back(log_c_k[k - 1] + log10_table[k] - log10_table[k - minimum_sample]);
    // 计算 C_n^k  从n里面选k个，n=所有匹配的特征点数量
    log_c_n.push_back(0);                           // log C_0^0 = 0
    log_c_n.push_back(log10_table[sample_size]);    // log C_n^1 = log n
    for(int k = 2; k <= sample_size; k++)
        log_c_n.push_back(log_c_n[k - 1] + log10_table[sample_size - k + 1] - log10_table[k]);
}

void ACRansac_NFA::SetResidual(const vector<double>& _residuals)
{
    residuals = _residuals;
}

// 计算NFA的时候只使用内点的误差，不使用所有的误差，那么为什么在SetResidual的时候不直接传递内点误差呢？非得传递所有误差
// 然后在取出内点误差，多麻烦。这是因为要保证inlier_idx是所有匹配的索引，而不是仅仅在内点里的索引，如果只是传递
// 内点的误差，就会导致后面计算得到的新的内点的索引是基于老的内点的索引，如此一来，就没法恢复出基于初始的匹配的索引了
bool ACRansac_NFA::ComputeNFA(std::vector<size_t>& inlier_idx, std::pair<double,double>& nfa)
{
    // 这是不太准确的计算方法，但是速度比较快，适用于提供了 max_threshold 的情况
    // 基本思路是使用直方图来统计各个误差的分布情况
    if(quantified_evaluation)
    {
        // 计算一个直方图
        const int num_bins = 20;
        const double bin_interval = (max_threshold - 0.0) / num_bins;
        vector<size_t> histo(num_bins, 0);
        size_t num_overflow = 0;    // 超出直方图上下界的residual的数量
        size_t num_underflow = 0;
        for(const double& res : residuals)
        {
            if(res < 0)
                num_underflow ++;
            else if (res >= max_threshold)
                num_overflow ++;
            else 
            {
                size_t bin_idx = res / bin_interval;
                histo[bin_idx]++;
            }
        }

        pair<double, double> best_nfa(numeric_limits<double>::infinity(), 0.0);
        size_t accumulate = 0;
        for(size_t bin = 0; bin < num_bins; bin++)
        {
            accumulate += histo[bin];
            double bin_start = static_cast<double>(bin) * bin_interval;
            if(accumulate > minimum_sample && bin_start > numeric_limits<float>::epsilon())
            {
                const double log_alpha = log_alpha0 + mult_error * log10(bin_start + numeric_limits<float>::epsilon());
                pair<double, double> curr_nfa(
                    log_e0 + log_alpha * static_cast<double>(accumulate - minimum_sample) + log_c_n[accumulate] + log_c_k[accumulate],
                    bin_start);
                if(curr_nfa.first < best_nfa.first && curr_nfa.first < 0)
                    best_nfa = curr_nfa;
            }
        }
        if(best_nfa.first >= 0)
            return false;
        nfa.first = best_nfa.first;
        nfa.second = best_nfa.second;

        for(size_t idx = 0; idx < sample_size; idx++)
        {
            if(residuals[idx] < nfa.second)
                inlier_idx.push_back(idx);
        }

        return inlier_idx.size() > minimum_sample;
    }
    // 这是比较准确的方法，计算所有的误差对应的nfa，然后选出最好的
    // 但是相应的，这个速度也会比较慢
    else 
    {
        // 先对所有的residual从小到大排序
        std::vector<std::pair<double, size_t>> sorted_residuals;
        sorted_residuals.clear();
        sorted_residuals.reserve(residuals.size());
        for(size_t i = 0; i < residuals.size(); i++)
            sorted_residuals.emplace_back(pair<double, size_t>(residuals[i], i));
        sort(sorted_residuals.begin(), sorted_residuals.end());
        
        pair<double, size_t> best_nfa(numeric_limits<double>::infinity(), 0);
        // 这里的k和k-1两个下标有一些让人疑惑，举个例子，假设使用8点法，那么在计算 NFA(M,k) 的时候，k最小为9
        // 但是存储的索引是从0开始的，所以第k小的误差是保存在数组的k-1位置的。但是在预先计算所有组合数的时候（也就是log C_n^k）
        // 也是从索引0开始的，那么就意味着第k个组合数是保存在数组的第k个位置。 总的来说，假设k=9，那么它对应的
        // 误差存储在9-1=8位置，它对应的组合数存储在9位置
        for(size_t k = minimum_sample + 1; k <= sample_size && sorted_residuals[k-1].first <= max_threshold; k++)
        {
            const double log_alpha = log_alpha0 + mult_error * log10(sorted_residuals[k-1].first + numeric_limits<float>::epsilon());
            pair<double, double> curr_nfa(
                log_e0 + log_alpha * static_cast<double>(k - minimum_sample) + log_c_n[k] + log_c_k[k],
                k);
            if(curr_nfa.first < best_nfa.first)
                best_nfa = curr_nfa;
        }

        nfa.first = best_nfa.first;
        nfa.second = sorted_residuals[best_nfa.second - 1].first;
        
        inlier_idx.clear();
        for(size_t i = 0; i < best_nfa.second; i++)
        {
            inlier_idx.push_back(sorted_residuals[i].second);
        }
        return inlier_idx.size() > minimum_sample;
    }
}

void ACRansac_NFA::SetSampleSize(const size_t& _num_sample)
{
    sample_size = _num_sample;
}

ACRansac_NFA::~ACRansac_NFA()
{
}

