/*
 * @Author: Diantao Tu
 * @Date: 2022-04-29 15:23:20
 */

#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <functional>

std::mt19937 CreateRandomEngine(bool use_fixed_seed) ;

/**
 * @description: 生成一组无重复的随机数
 * @param size 生成随机数的数量
 * @param min 随机数区间的下限
 * @param max 随机数区间的上限
 * @return 生成的随机数组
 */
template<typename T>
std::vector<T> CreateRandomArray(const size_t size, const T min, const T max)
{
    assert(min <= max);
    assert(size <= static_cast<size_t>(max - min + 1));
    std::mt19937 random_engine = CreateRandomEngine(false);
    std::uniform_int_distribution<T> uniform_int_distribution(min, max);
    const auto make_size = static_cast<size_t>(size * 1.2);
    std::vector<T> v;
    v.reserve(size);
    while (v.size() != size) {
        // Add random integer values
        while (v.size() < make_size) {
            v.push_back(uniform_int_distribution(random_engine));
        }

        // Sort to remove duplicates so that the last iterator of the deduplicated sequence goes into 'unique_end'
        std::sort(v.begin(), v.end());
        auto unique_end = std::unique(v.begin(), v.end());

        // If the vector size is too large, change it to an iterator up to the 'size'
        if (size < static_cast<size_t>(std::distance(v.begin(), unique_end))) {
            unique_end = std::next(v.begin(), size);
        }

        // Delete the portion from the duplication to the end
        v.erase(unique_end, v.end());
    }

    // Shuffle because they are in ascending order
    std::shuffle(v.begin(), v.end(), random_engine);

    return v;
}

/**
 * @description: 从给定的数组中随机选择出n个数
 * @param size 选择的数字的个数
 * @param src_array 给定的数组
 * @return 随机选择的n个数组成的数组
 */
template<typename T>
std::vector<T> CreateRandomArray(const size_t size ,const std::vector<T>& src_array)
{
    if(size > src_array.size())
        return std::vector<T>();
    else if(size == src_array.size())
        return src_array;
    std::vector<T> index = CreateRandomArray(size, size_t(0), size_t(src_array.size() - 1));
    std::vector<T> random_array;
    for(const T& idx : index)
        random_array.push_back(src_array[idx]);
    return random_array;
}

#endif