/*
 * @Author: Diantao Tu
 * @Date: 2021-12-04 21:05:52
 */

#ifndef _TRACKS_H_
#define _TRACKS_H_

#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <limits>
#include <boost/serialization/serialization.hpp>
#include "../base/Serialization.h"

// Union-Find/Disjoint-Set data structure
//--
// A disjoint-set data structure also called a union–find data structure
// or merge–find set, is a data structure that keeps track of a set of elements
// partitioned into a number of disjoint (non-overlapping) subsets.
// It supports two operations:
// - Find: Determine which subset a particular element is in.
//   - It returns an item from this set that serves as its "representative";
// - Union: Join two subsets into a single subset.
// Sometime a Connected method is implemented:
// - Connected:
//   - By comparing the result of two Find operations, one can determine whether
//      two elements are in the same subset.
//--
struct UnionFind
{
	// Represent the DS/UF forest thanks to two array:
	// A parent 'pointer tree' where each node holds a reference to its parent node
	std::vector<unsigned int> m_cc_parent;
	// A rank array used for union by rank
	std::vector<unsigned int> m_cc_rank;
	// A 'size array' to know the size of each connected component
	std::vector<unsigned int> m_cc_size;

	// Init the UF structure with num_cc nodes
	void InitSets
	(
		const unsigned int num_cc
	)
	{
		// all set size are 1 (independent nodes)
		m_cc_size.resize(num_cc, 1);
		// Parents id have their own CC id {0,n}
		m_cc_parent.resize(num_cc);
		std::iota(m_cc_parent.begin(), m_cc_parent.end(), 0);
		// Rank array (0)
		m_cc_rank.resize(num_cc, 0);
	}

	// Return the number of nodes that have been initialized in the UF tree
	unsigned int GetNumNodes() const
	{
		return static_cast<unsigned int>(m_cc_size.size());
	}

	// Return the representative set id of I nth component
	unsigned int Find
	(
		unsigned int i
	)
	{
		// Recursively set all branch as children of root (Path compression)
		if (m_cc_parent[i] != i)
			m_cc_parent[i] = Find(m_cc_parent[i]);
		return m_cc_parent[i];
	}

	// Replace sets containing I and J with their union
	void Union
	(
		unsigned int i,
		unsigned int j
	)
	{
		i = Find(i);
		j = Find(j);
		if (i == j)
		{ // Already in the same set. Nothing to do
			return;
		}

		// x and y are not already in same set. Merge them.
		// Perform an union by rank:
		//  - always attach the smaller tree to the root of the larger tree
		if (m_cc_rank[i] < m_cc_rank[j])
		{
			m_cc_parent[i] = j;
			m_cc_size[j] += m_cc_size[i];
		}
		else
		{
			m_cc_parent[j] = i;
			m_cc_size[i] += m_cc_size[j];
			if (m_cc_rank[i] == m_cc_rank[j])
				++m_cc_rank[i];
		}
	}
};

struct PointTrack
{
	uint32_t id;
	std::set<std::pair<uint32_t, uint32_t>> feature_pairs;
	Eigen::Vector3d point_3d;     	// 空间点坐标
	Eigen::Vector3i rgb;			// 空间点颜色, 默认是白色(255,255,255)

	PointTrack():id(std::numeric_limits<uint32_t>::max()), rgb(Eigen::Vector3i::Ones() * 255){}
	PointTrack(const uint32_t& _id, const std::set<std::pair<uint32_t, uint32_t>>& _feature_pairs)
		:id(_id), feature_pairs(_feature_pairs), rgb(Eigen::Vector3i::Ones() * 255)
	{}
	PointTrack(const uint32_t& _id, const std::set<std::pair<uint32_t, uint32_t>>& _feature_pairs, const Eigen::Vector3d& point)
		:id(_id), feature_pairs(_feature_pairs), point_3d(point), rgb(Eigen::Vector3i::Ones() * 255)
	{}

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		ar & id;
		ar & feature_pairs;
		ar & point_3d;
		ar & rgb;
	}
};

struct LineTrack
{
	uint32_t id;
	std::set<std::pair<uint32_t, uint32_t>> feature_pairs;

	LineTrack():id(std::numeric_limits<uint32_t>::max()){}

  	LineTrack(const uint32_t& _id):id(_id){}

	LineTrack(const uint32_t& _id, const std::pair<uint32_t, uint32_t>& _feature_pair):id(_id)
	{
		feature_pairs.insert(_feature_pair);
	}
	
	LineTrack(const uint32_t& _id, const std::set<std::pair<uint32_t, uint32_t>>& _feature_pairs)
		:id(_id), feature_pairs(_feature_pairs)
	{}

	bool IsInside(const std::pair<uint32_t, uint32_t>& pair) const 
	{
		return feature_pairs.count(pair) > 0;
	}
};

class TrackBuilder
{
private:
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> feature_to_index;
    std::map<uint32_t, std::pair<uint32_t, uint32_t>> index_to_feature;
    UnionFind uf_tree;
    size_t max_id;		// 在理论上track的最大id，实际上只能比这个小
    bool allow_multiple_map;
public:
    TrackBuilder(bool _allow_multiple_map = false);
    /**
     * @description: 根据图像的特征之间的匹配关系建立track
     * @param image_pairs 图像对组成的vector，里面的每一项是一对匹配的图像id
     * @param matches 每一对匹配的图像之间的特征匹配关系，要求存储顺序和image_pairs里图像顺序一致
     * @return {*}
     */	
    bool Build(const std::vector<std::pair<size_t, size_t>>& image_pairs, 
                const std::vector<std::vector<cv::DMatch>>& matches);
    // 和上面的一样，只是匹配的特征用的是pair<uint32_t,uint32_t>来表示，没有使用cv::DMatch
    bool Build(const std::vector<std::pair<size_t, size_t>>& image_pairs, 
                const std::vector<std::set<std::pair<uint32_t,uint32_t>>>& matches);
    // 根据长度对track过滤，长度低于length的都过滤掉
    // length指的是track里包含的图像数目，如果只有两张图像，但是里面有4个特征（allow_multiple_map=true的情况下），
    // 这也依然会被过滤掉
    bool Filter(const uint32_t length = 2);
    // 有多少个track
    size_t TracksNumber() const;
    // 输出所有的track，形式为 {track id, {image id, feature id} {image id, feature id} {image id, feature id} }
    // key = track id    value = {image id, feature id}组成的集合
    bool ExportTracks(std::map<uint32_t, std::set<std::pair<uint32_t, uint32_t>>>& tracks);
    // 以LineTrack的形式输出track
	  bool ExportTracks(std::vector<LineTrack>& tracks);
    size_t GetMaxID();
    ~TrackBuilder();
};







#endif