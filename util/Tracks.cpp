/*
 * @Author: Diantao Tu
 * @Date: 2021-12-05 10:20:47
 */

#include "Tracks.h"

using namespace std;

TrackBuilder::TrackBuilder(bool _allow_multiple_map):allow_multiple_map(_allow_multiple_map)
{
}

bool TrackBuilder::Build(const std::vector<std::pair<size_t, size_t>>& image_pairs, 
                const std::vector<std::vector<cv::DMatch>>& matches)
{
    assert(image_pairs.size() == matches.size());
    // 所有的二维特征点，保存的是 {image_id, feature_id}的集合
    set<pair<uint32_t, uint32_t>> all_features;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].first;
        const size_t idx2 = image_pairs[i].second;
        for(const cv::DMatch& curr_match : matches[i])
        {
            all_features.emplace(idx1, curr_match.queryIdx);
            all_features.emplace(idx2, curr_match.trainIdx);
        }
    }
    // 为每一个{image_id, feature_id}分配一个索引
    uint32_t count = 0;
    for(const pair<uint32_t, uint32_t>& feature : all_features)
    {
        feature_to_index.emplace(feature, count);
        index_to_feature.emplace(count, feature);
        count ++;
    }
    all_features.clear();
    max_id = count - 1;
    // 在 uf tree里根据一对图像上的特征匹配关系把对应的{image_id, feature_id}放到同一个集合里，
    // 在同一个集合里的{image_id, feature_id}就代表着它们对应于同一个三维空间点
    // 注意，这里真正操纵的是{image_id, feature_id}对应的index，而不是数据对本身
    uf_tree.InitSets(feature_to_index.size());
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].first;
        const size_t idx2 = image_pairs[i].second;
        for(const cv::DMatch& curr_match : matches[i])
        {
            const pair<uint32_t, uint32_t> p1(idx1, curr_match.queryIdx);
            const pair<uint32_t, uint32_t> p2(idx2, curr_match.trainIdx);
            uf_tree.Union(feature_to_index[p1], feature_to_index[p2]);
        }
    }
    return true;
}

bool TrackBuilder::Build(const std::vector<std::pair<size_t, size_t>>& image_pairs, 
                const std::vector<std::set<pair<uint32_t,uint32_t>>>& matches)
{
    assert(image_pairs.size() == matches.size());
    // 所有的二维特征点，保存的是 {image_id, feature_id}的集合
    set<pair<uint32_t, uint32_t>> all_features;
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].first;
        const size_t idx2 = image_pairs[i].second;
        for(const pair<uint32_t,uint32_t>& curr_match : matches[i])
        {
            all_features.emplace(idx1, curr_match.first);
            all_features.emplace(idx2, curr_match.second);
        }
    }
    // 为每一个{image_id, feature_id}分配一个索引
    uint32_t count = 0;
    for(const pair<uint32_t, uint32_t>& feature : all_features)
    {
        feature_to_index.emplace(feature, count);
        index_to_feature.emplace(count, feature);
        count ++;
    }
    all_features.clear();
    max_id = count - 1;
    // 在 uf tree里根据一对图像上的特征匹配关系把对应的{image_id, feature_id}放到同一个集合里，
    // 在同一个集合里的{image_id, feature_id}就代表着它们对应于同一个三维空间点
    // 注意，这里真正操纵的是{image_id, feature_id}对应的index，而不是数据对本身
    uf_tree.InitSets(feature_to_index.size());
    for(size_t i = 0; i < image_pairs.size(); i++)
    {
        const size_t idx1 = image_pairs[i].first;
        const size_t idx2 = image_pairs[i].second;
        for(const pair<uint32_t,uint32_t>& curr_match : matches[i])
        {
            const pair<uint32_t, uint32_t> p1(idx1, curr_match.first);
            const pair<uint32_t, uint32_t> p2(idx2, curr_match.second);
            uf_tree.Union(feature_to_index[p1], feature_to_index[p2]);
        }
    }
    return true;
}

bool TrackBuilder::Filter(const uint32_t length)
{
    map<uint32_t, set<uint32_t>> tracks;     // key=track_id  value={image_id, image_id, image_id, ...}
    set<uint32_t> problematic_track_id;     // {track_id, track_id, ...}

    for(size_t i = 0; i < feature_to_index.size(); i++)
    {
        // 找到id为i的特征点所在的集合的id,这个集合的id就是track id
        uint32_t track_id = uf_tree.Find(i);
        // feature是{image id, feature id}
        const pair<uint32_t, uint32_t> feature = index_to_feature[i];
        // 把当前feature 对应的image id插入到track id 对应的集合中，如果插入失败，就代表这个集合已经有了相同的
        // image id，那就代表着当前的track id对应的空间三维点在同一张图像上有两个特征点都对应着，明显是不对的，那就把这个
        // track标记为不可靠
        // 但是如果允许一个空间直线对应同一张图像上的多个直线，那么这个track就不会被标记为不可靠（这个主要是针对图像直线的track）
        if(tracks[track_id].insert(feature.first).second == false && !allow_multiple_map)
        {
            problematic_track_id.insert(track_id);
        }
    }
    // 检查track的长度，如果长度不够，也要标记为不可靠
    for(const auto& val : tracks)
    {
        if(val.second.size() < length)
            problematic_track_id.insert(val.first);
    }
    // 对于标记为不可靠的track，其实就是uf tree中不可靠的集合，要把这些集合重置
    for(uint32_t& root_index: uf_tree.m_cc_parent)
    {
        if(problematic_track_id.count(root_index) > 0)
        {
            uf_tree.m_cc_size[root_index] = 1;
            root_index = numeric_limits<uint32_t>::max();
        }
    }
    return true;
}

size_t TrackBuilder::TracksNumber() const
{
    std::set<uint32_t> parent_id(uf_tree.m_cc_parent.cbegin(), uf_tree.m_cc_parent.cend());
    // Erase the "special marker" that depicted rejected tracks
    parent_id.erase(std::numeric_limits<uint32_t>::max());
    return parent_id.size();
}

bool TrackBuilder::ExportTracks(std::map<uint32_t, std::set<std::pair<uint32_t, uint32_t>>>& tracks)
{
    tracks.clear();
    for(uint32_t i = 0; i < feature_to_index.size(); i++)
    {
        const pair<uint32_t, uint32_t>& feature = index_to_feature[i];
        const uint32_t track_id = uf_tree.m_cc_parent[i];
        // track id = max 代表track不可靠，cc_size=1代表当前track只有一张图像上的一个特征点，也是不可靠的
        if(track_id != numeric_limits<uint32_t>::max() && uf_tree.m_cc_size[track_id] > 1)
        {
            tracks[track_id].insert(feature);
        }
    }
    return tracks.size() > 0;
}

bool TrackBuilder::ExportTracks(std::vector<LineTrack>& tracks)
{
    map<uint32_t, size_t> track_id_to_index;
    tracks.clear();
    for(uint32_t i = 0; i < feature_to_index.size(); i++)
    {
        const pair<uint32_t, uint32_t>& feature = index_to_feature[i];
        const uint32_t track_id = uf_tree.m_cc_parent[i];
        // track id = max 代表track不可靠，cc_size=1代表当前track只有一张图像上的一个特征点，也是不可靠的
        if(track_id != numeric_limits<uint32_t>::max() && uf_tree.m_cc_size[track_id] > 1)
        {
            map<uint32_t, size_t>::const_iterator it = track_id_to_index.find(track_id);
            if(it != track_id_to_index.end())
            {
                tracks[it->second].feature_pairs.insert(feature);
            }
            else 
            {
                track_id_to_index[track_id] = tracks.size();
                tracks.push_back(LineTrack(track_id, feature));
            }
        }
    }
    return tracks.size() > 0;
}

size_t TrackBuilder::GetMaxID()
{
    return max_id;
}

TrackBuilder::~TrackBuilder()
{
}