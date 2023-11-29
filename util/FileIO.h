/*
 * @Author: Diantao Tu
 * @Date: 2021-11-19 13:50:16
 */

#ifndef _FILE_IO_H_
#define _FILE_IO_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <omp.h>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/detail/basic_iarchive.hpp>
#include <boost/archive/detail/basic_oarchive.hpp>

#include "../base/common.h"
#include "../sensors/Frame.h"
#include "MatchPair.h"
#include "PanoramaLine.h"
#include "Tracks.h"



/**
 * @description: 从文件中读取位姿并保存,位姿是以变换矩阵的顺序存储的,也就是12个数,R(0,0) R(0,1) R(0,2) t(0) R(1,0) ... t(1) R(2,0) ... t(2)
 * @param file_path 位姿文件保存的路径
 * @param with_invalid 设为true则会输出不可用的位姿，例如inf nan等，设为false则会过滤掉这些位姿
 * @param rotation_list 输出的旋转矩阵
 * @param trans_list 输出的平移向量
 * @param name_list 输出的文件名(如果有的话)
 * @return 读取是否成功
 */
bool ReadPoseT(std::string file_path, bool with_invalid, eigen_vector<Eigen::Matrix3d>& rotation_list, 
            eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list  );

// 和上面的一样，只不过是以四元数和平移向量形式存储的
// qx qy qz qw tx ty tz
bool ReadPoseQt(std::string file_path, eigen_vector<Eigen::Matrix3d>& rotation_list, 
            eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list );

void ExportPoseT(const std::string file_path, const eigen_vector<Eigen::Matrix3d>& rotation_list,
                const eigen_vector<Eigen::Vector3d>& trans_list,
                const std::vector<std::string>& name_list);

// 读取txt形式的GPS数据，格式为 file_name x y z 
// 注意：这个gps数据已经是经过变换之后的了，不是原始的经度、纬度、高度
bool ReadGPS(const std::string file_path, eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list);

// 以二进制形式保存match pair到指定路径
bool ExportMatchPair(const std::string folder, const std::vector<MatchPair>& pairs);

// 读取指定路径下的二进制形式保存的match pair
bool ReadMatchPair(const std::string folder, std::vector<MatchPair>& pairs, const int num_threads = 1);

bool ExportFrame(const std::string folder, const std::vector<Frame>& frames);

bool ReadFrames(const std::string frame_folder, const std::string image_folder, std::vector<Frame>& frames, 
                const int num_threads = 1, const bool skip_descriptor = false);

bool ExportOpenCVMat(const std::string& file_path, const cv::Mat& mat);

bool ReadOpenCVMat(const std::string& file_path, cv::Mat& mat);

// TODO: 保存深度图、法向量图、置信度图的时候，文件名是依据frame的索引来的，这么做是为了方便只读取部分的深度图，便于debug
// 等到全部完成后，需要把文件名改成根据Frame的名字来，这样就和上面的方法保持一致了
bool ExportFrameDepthAll(const std::string& folder, const vector<Frame>& frames, bool use_filtered_depth = false);

bool ReadFrameDepthAll(const std::string& folder, vector<Frame>& frames, const std::string& file_type, bool use_filtered_depth = false);

inline bool ReadFrameDepth(const std::string& file_path, Frame& frame, bool use_filtered_depth = false)
{
    ReadOpenCVMat(file_path, (use_filtered_depth ? frame.depth_filter : frame.depth_map));
    const cv::Mat& img_depth = (use_filtered_depth ? frame.depth_filter : frame.depth_map);
    if(img_depth.rows != frame.GetImageRows() || img_depth.cols != frame.GetImageCols())
        return false;
    return true;
}

bool ExportFrameNormalAll(const std::string& folder, const vector<Frame>& frames);

bool ReadFrameNormalAll(const std::string& folder, vector<Frame>& frames, const std::string& file_type);

inline bool ReadFrameNormal(const std::string& file_path, Frame& frame)
{
    ReadOpenCVMat(file_path, frame.normal_map);
    if(frame.normal_map.rows != frame.GetImageRows() || 
        frame.normal_map.cols != frame.GetImageCols())
        return false;
    return true;
}

bool ExportFrameConfAll(const std::string& folder, const vector<Frame>& frames);

inline bool ExportConfMap(const std::string& file_path, const cv::Mat& conf_map)
{
    if(conf_map.empty())
        return false;
    cv::Mat conf_16u = conf_map + 1.0;                  // 从 [-1,1] 变成[0,2]
    conf_16u *= 32767;                                  // conf范围只有[0,2] 而 CV_16U 可以存储[0,65535]
    conf_16u.convertTo(conf_16u, CV_16U);
    return ExportOpenCVMat(file_path, conf_16u);
}

bool ReadFrameConfAll(const std::string& folder, vector<Frame>& frames, const std::string& file_type);

inline bool ReadFrameConf(const std::string& file_path, Frame& frame)
{
    ReadOpenCVMat(file_path, frame.conf_map);
    frame.conf_map.convertTo(frame.conf_map, CV_32F);
    frame.conf_map /= 32767;
    frame.conf_map -= 1.f;
    if(frame.conf_map.rows != frame.GetImageRows() || 
        frame.conf_map.cols != frame.GetImageCols())
        return false;
    return true;
}

bool ExportPanoramaLines(const std::string folder, const std::vector<PanoramaLine>& image_lines_all);

bool ReadPanoramaLines(const std::string line_folder, const std::string image_folder, std::vector<PanoramaLine>& image_lines_all);

bool ExportPointTracks(const std::string file, const std::vector<PointTrack>& tracks);

bool ReadPointTracks(const std::string file, std::vector<PointTrack>& tracks);
#endif