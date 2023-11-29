/*
 * @Author: Diantao Tu
 * @Date: 2021-11-04 19:11:48
 */

#include "common.h"

using namespace std;

vector<string> SplitString(const string& str, const char& delimma)
{
    vector<string> split;
    stringstream ss(str);
    string tmp;
    while (getline(ss, tmp, delimma))
    {
        split.push_back(tmp);
    }
    return split;
}

// 递归地遍历某个文件夹下的文件，并选出以 filtType 为结尾的文件
vector<string> IterateFiles(const string& pathName, const vector<string>& fileTypes)
{
    vector<string> fileNames;
    if(!boost::filesystem::exists(pathName))
        return fileNames;
    
    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator iter(pathName); iter != endIter; ++iter)
    {
        if (boost::filesystem::is_regular_file(iter->status()))
        {
            std::string file_name = iter->path().string();
            for(const string& fileType : fileTypes)
                if(EndWith(file_name, fileType))
                    fileNames.push_back(iter->path().string());
        }
        else if (boost::filesystem::is_directory(iter->path()))
        {
            vector<string> names = IterateFiles(iter->path().string(), fileTypes);
            fileNames.insert(fileNames.end(), names.begin(), names.end());
        }
    }
    // 遍历文件后排序的结果是 xxxxx_1.pcd xxxx_10.pcd xxxx_11.pcd 这个顺序不对，应该是
    // _1.pcd  _2.pcd  _3.pcd 所以重新写个排序函数来排序
    sort(fileNames.begin(), fileNames.end(), FileNameComp);
    return fileNames;
}

vector<string> IterateFiles(const string& pathName, const string& fileType)
{
    return IterateFiles(pathName, vector<string>{fileType});
}

bool FileNameComp(const string& a, const string& b)
{
    string::size_type start = a.find_last_of('/');
    if(start == string::npos)
        start = 0;
    string::size_type end = a.find_last_of('.');
    while(start < end && !IsInside(a[start], '0', '9'))
        start++;
    int num_a = str2num<int>(a.substr(start, end - start));
    start = b.find_last_of('/');
    if(start == string::npos)
        start = 0;
    end = b.find_last_of('.');
    while(start < end && !IsInside(b[start], '0', '9'))
        start++;
    int num_b = str2num<int>(b.substr(start, end - start));

    return num_a < num_b;
}