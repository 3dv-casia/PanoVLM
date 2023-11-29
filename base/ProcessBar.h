/*
 * @Author: Diantao Tu
 * @Date: 2021-12-30 10:38:41
 */

#ifndef _PROCESS_BAR_H_
#define _PROCESS_BAR_H_

#include <glog/logging.h>
#include <mutex>
#include <vector>
#include <math.h>

class ProcessBar
{
private:
    std::mutex m;
    const size_t max_count;     // 进度的上限
    size_t count;               // 当前进度
    float gap;                  // 每隔多久输出一次，0.1就是每10%进度输出一次，0.05就是每5%进度输出一次
    std::vector<bool> reached;
public:
    ProcessBar(const size_t _max_count, const float _gap);
    void Add(const size_t num = 1);
    ~ProcessBar();
};

#endif