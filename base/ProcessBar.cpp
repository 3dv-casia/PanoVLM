/*
 * @Author: Diantao Tu
 * @Date: 2021-12-30 12:32:15
 */
#include "ProcessBar.h"

using namespace std;

ProcessBar::ProcessBar(const size_t _max_count, const float _gap)
        :max_count(_max_count),count(0),gap(_gap)
{
    const size_t length = ceil(1.0/gap);
    reached = vector<bool>(length, false);
}

void ProcessBar::Add(const size_t num)
{
    lock_guard<mutex> lock(m);
    count += num;
    size_t idx = 1.f * count / max_count / gap;
    if(reached[idx] == false)
    {
        LOG(INFO) << 100 * idx * gap << "% " << endl;
        reached[idx] = true;
    } 
}

ProcessBar::~ProcessBar()
{

}