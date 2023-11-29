/*
 * @Author: Diantao Tu
 * @Date: 2022-07-18 15:47:00
 */
#ifndef _STRUCTURE_H_
#define _STRUCTURE_H_

#include <omp.h>

#include "../base/Geometry.hpp"
#include "../base/Math.h"

#include "../sensors/Equirectangular.h"
#include "../sensors/Frame.h"

#include "../util/Tracks.h"
#include "../util/MatchPair.h"

#include "Triangulate.h"


/**
 * @description: 对图像对之间匹配的特征点进行三角化，形成SfM中的 structure
 * @param frames 所有图像信息
 * @param image_pairs 所有的图像对匹配信息
 * @return 三角化之后的空间点（structure）
 */
std::vector<PointTrack> TriangulateTracks(const std::vector<Frame>& frames, const std::vector<MatchPair>& image_pairs);

/**
 * @description: 过滤距离图像太远的三维点，具体方法是对每个三维点计算它到可见图像之间的距离，同时计算所有可见图像之间的最远距离，
 *              如果三维点到可见图像之间的距离大于可见图像间最远距离的threshold倍，就认为和这个三维点不准确，过滤掉
 * @param threshold {double&} 距离之比的阈值
 * @return 过滤的三维点的数量
 */
size_t FilterTracksToFar(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold);

/**
 * @description: 根据点的投影误差进行过滤，投影误差以像素为单位
 * @param threshold {double&} 误差阈值
 * @return 过滤的三维点数量
 */
size_t FilterTracksPixelResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold);

/**
 * @description: 根据点的投影误差进行过滤，投影误差以角度为单位
 * @param threshold {double&} 误差阈值
 * @return 过滤的三维点数量
 */
size_t FilterTracksAngleResidual(const std::vector<Frame>& frames, std::vector<PointTrack>& tracks, const double& threshold);

#endif