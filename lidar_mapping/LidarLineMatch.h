/*
 * @Author: Diantao Tu
 * @Date: 2022-04-22 15:12:44
 */

#ifndef _LIDAR_LINE_MATCH_H_
#define _LIDAR_LINE_MATCH_H_

#include "../sensors/Velodyne.h"
#include "../base/common.h"
#include "LidarFeatureAssociate.h"
#include "../util/Tracks.h"

class LidarLineMatch
{
private:
    const std::vector<Velodyne>& lidars;
    std::vector<LineTrack> tracks;
    // 每张图像会和之后的neighbor_size 张图像进行直线匹配
    int neighbor_size;
    // 生成的track至少包含min_track_length张图像
    int min_track_length;
public:
    LidarLineMatch(const std::vector<Velodyne>& _lidars);
    /**
     * @description: 对所有雷达进行匹配，生成由雷达直线组成的track
     * @param {*}
     * @return 匹配是否成功
     */    
    bool GenerateTracks();
    
    void SetNeighborSize(const int size);

    void SetMinTrackLength(const int length);

    const std::vector<LineTrack>& GetTracks() const;
    ~LidarLineMatch();
};





#endif