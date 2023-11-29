/*
 * @Author: Diantao Tu
 * @Date: 2022-04-24 10:23:45
 */
#include "LidarLineMatch.h"

using namespace std;
LidarLineMatch::LidarLineMatch(const std::vector<Velodyne>& _lidars):lidars(_lidars),neighbor_size(2),min_track_length(3)                          
{
}

void VisualizeTrack(const LineTrack& track, const std::vector<Velodyne>& lidars, const string folder, bool one_file=false)
{
    uint32_t track_id = track.id;
    if(one_file)
    {
        pcl::PointCloud<PointType> cloud;
        for(const auto& pair : track.feature_pairs)
            cloud += lidars[pair.first].edge_segmented[pair.second];
        pcl::io::savePCDFileASCII(folder + "/track" + num2str(track_id) + ".pcd", cloud);
        return;
    }
    map<uint32_t, set<uint32_t>> lidar_lines;
    for(const auto& pair : track.feature_pairs)
        lidar_lines[pair.first].insert(pair.second);
    for(map<uint32_t, set<uint32_t>>::const_iterator it = lidar_lines.begin(); it != lidar_lines.end(); it++)
    {
        const uint32_t& lidar_id = it->first;
        pcl::PointCloud<PointType> cloud;
        for(const uint32_t& line_id : it->second)
            cloud += lidars[lidar_id].edge_segmented[line_id];
        pcl::io::savePCDFileASCII(folder + "/track" + num2str(track_id) + "_" + num2str(lidar_id) + ".pcd", cloud);
    }
}

bool LidarLineMatch::GenerateTracks()
{
    vector<pair<size_t, size_t>> pairs;
    vector<set<pair<uint32_t, uint32_t>>> feature_each_pair;
    #if 1
    vector<vector<int>> neighbors = FindNeighbors(lidars, neighbor_size);
    #elif 0
    vecotr<vector<int>> neighbors = FindNeighborsConsecutive(lidars, neighbor_size);
    #elif 0
    // 用连续的K个近邻作为近邻帧
    for(int i = 0; i < int(lidars.size() - neighbor_size); i++)
    {
        for(int j = i + 1; j < lidars.size() && j <= i + neighbor_size; j++)
        {
            // 注意这里的顺序是lidars[j] 和 lidars[i]，因为需要的特征匹配是从 lidars[i]到lidars[j]的
            vector<Line2Line> associations = AssociateLine2Line(lidars[j], lidars[i], 0.3);
            set<pair<uint32_t, uint32_t>> feature_pairs;
            for(const Line2Line& ass : associations)
            {
                feature_pairs.insert({ass.neighbor_line_idx, ass.ref_line_idx});
            }
            feature_each_pair.push_back(feature_pairs);
            pairs.push_back({i,j});
        }
    }
    #endif
    for(int i = 0; i < neighbors.size(); i++)
    {
        if(!lidars[i].IsPoseValid())
            continue;
        for(const int& nei_id : neighbors[i])
        {
            vector<Line2Line> associations = AssociateLine2Line(lidars[nei_id], lidars[i], 0.3);
            set<pair<uint32_t, uint32_t>> feature_pairs;
            for(const Line2Line& ass : associations)
                feature_pairs.insert({ass.neighbor_line_idx, ass.ref_line_idx});
            
            feature_each_pair.push_back(feature_pairs);
            pairs.push_back({i, nei_id});
        }
    }
    TrackBuilder track_builder(true);
    track_builder.Build(pairs, feature_each_pair);
    track_builder.Filter(min_track_length);
    track_builder.ExportTracks(tracks);
    for(int i = 0; i < tracks.size(); i++)
        tracks[i].id = i;
    LOG(INFO) << "generate " << tracks.size() << " lidar line tracks";
    return true;
    // 以下部分是可视化
    for(const LineTrack& track : tracks)
        VisualizeTrack(track, lidars, "./", true);
    for(const Velodyne& lidar : lidars)
        pcl::io::savePCDFileASCII(num2str(lidar.id) + ".pcd", lidar.cloud_scan);
    return true;
}

void LidarLineMatch::SetNeighborSize(const int size)
{
    assert(size > 0);
    neighbor_size = size;
}

void LidarLineMatch::SetMinTrackLength(const int length)
{
    assert(length > 0);
    min_track_length = length;
}

const std::vector<LineTrack>& LidarLineMatch::GetTracks() const
{
    return tracks;
}

LidarLineMatch::~LidarLineMatch()
{
}