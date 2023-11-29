/*
 * @Author: Diantao Tu
 * @Date: 2022-04-29 15:44:50
 */
#include "Random.hpp"

std::mt19937 CreateRandomEngine(bool use_fixed_seed) 
{
    std::mt19937 random_engine;
    if (use_fixed_seed) 
        return std::mt19937();
    else 
    {
        std::random_device random_device;
        std::vector<std::uint_least32_t> v(10);
        std::generate(v.begin(), v.end(), std::ref(random_device));
        std::seed_seq seed(v.begin(), v.end());
        return std::mt19937(seed);
    }
}