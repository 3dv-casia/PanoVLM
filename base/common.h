/*
 * @Author: Diantao Tu
 * @Date: 2021-10-11 16:28:02
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include <boost/filesystem.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pcl/point_types.h>

using namespace std;

enum LIDAR_TYPE{
    UNKNOWN = 0,
    LIVOX = 1,
    VLP = 2
};

template<typename T>
using eigen_vector = std::vector<T, Eigen::aligned_allocator<T>>;
template<typename Key, typename Value>
using eigen_map = std::map<Key, Value, std::less<Key>, Eigen::aligned_allocator<std::pair<Key, Value>>>;


typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<float, 6, 1> Vector6f;

// 字符串分割
std::vector<std::string> SplitString(const std::string& str, const char& delimma);

// 判断字符串是否以某个字符串为结尾
inline bool EndWith(const std::string& str, const std::string& ends)
{
    if(str.size() < ends.size())
        return false;
    return str.rfind(ends) == (str.size() - ends.size());
}

vector<std::string> IterateFiles(const std::string& pathName, const vector<std::string>& fileTypes);

// 递归地遍历某个文件夹下的文件，并选出以 filtType 为结尾的文件
vector<std::string> IterateFiles(const std::string& pathName, const std::string& fileType);

// 从文件路径中选择出文件名，也就是从 /aaaa/bbbb/cccc.xyz -> cccc
inline std::string FileName(std::string full_path)
{
    string::size_type pos1 = full_path.find_last_of('/');
    string::size_type pos2 = full_path.find_last_of('.');
    return full_path.substr(pos1 + 1, pos2 - pos1 - 1);
}

// 从pcl的点变成Eigen的vector
// d代表double
template<typename T>
inline Eigen::Vector3d PclPonit2EigenVecd(const T& pcl_point)
{
    return Eigen::Vector3d(pcl_point.x, pcl_point.y, pcl_point.z);
}

// f代表float
template<typename T>
inline Eigen::Vector3f PclPonit2EigenVecf(const T& pcl_point)
{
    return Eigen::Vector3f(pcl_point.x, pcl_point.y, pcl_point.z);
}

template<typename T, typename N>
inline void EigenVec2PclPoint(const Eigen::Matrix<N, 3, 1>& vec, T& point)
{
    point.x = vec(0);
    point.y = vec(1);
    point.z = vec(2);
}

// 统计vector中出现最多的元素的个数
template<typename T>
std::pair<T, int> MaxNum(const std::vector<T>& vec)
{
    std::map<T, int> map;
    for(auto& v : vec)
    {
        if(map.find(v) == map.end())
            map[v] = 1;
        else
            map[v]++;
    }
    int max_num = 0;
    T max_key;
    for(auto& m : map)
    {
        if(m.second > max_num)
        {
            max_num = m.second;
            max_key = m.first;
        }
    }
    return std::make_pair(max_key, max_num);
}

template<typename T>
inline T str2num(string str)
{
    T num;
    stringstream sin(str);
    if(sin >> num) {
        return num;
    }
    else{
        return std::numeric_limits<T>::infinity();
    }
}

template<typename T>
inline string num2str(T num)
{
    ostringstream oss;
    if (oss << num) {
        string str(oss.str());
        return str;
    }
    else {
        cout << "num2str error" << endl;
        exit(0);
    }
}

// 模仿python的join函数，把一系列的数字变成字符串，并且中间用空格分割
template<typename T>
inline string Join(vector<T> list, const string& spliter=" ")
{
    string str;
    for(const T& num : list)
        str += (num2str(num) + spliter);
    return str;
}

template<typename T>
inline bool IsInside(const T& a, const T& lower, const T& upper)
{
    return (a >= lower && a <= upper);
}

bool FileNameComp(const string& a, const string& b);


#endif