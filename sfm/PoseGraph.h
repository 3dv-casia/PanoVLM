/*
 * @Author: Diantao Tu
 * @Date: 2021-11-25 15:22:37
 */
#ifndef _POSE_GRAPH_H_
#define _POSE_GRAPH_H_

#include <lemon/connectivity.h>
#include <lemon/list_graph.h>
#include <lemon/kruskal.h>

#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <assert.h>
#include <glog/logging.h>

struct Triplet
{
    u_int32_t i, j, k;
    Triplet(const uint32_t _i, const uint32_t _j, const uint32_t _k):i(_i),j(_j),k(_k){}
};


class PoseGraph
{
public:
    lemon::ListGraph g;
    // 在lemon的graph中，每一个node都有一个id，这个id是从0开始的连续增加的
    // 但是在建立pose graph时输入的edge里包含的node id可能不是连续的
    // 所以需要一个映射，记录输入的edge里的node id 和 graph里的node id的对应关系
    // 在node_map_id这个变量里，每一个node是它在pose graph里的node，而这个node对应的值是它在输入的edge里的node id
    std::unique_ptr<lemon::ListGraph::NodeMap<uint32_t> > node_map_id;

    PoseGraph(const std::vector<std::pair<size_t, size_t>>& edges);
    PoseGraph(const std::set<std::pair<size_t, size_t>>& edges);
    std::set<size_t> KeepLargestEdgeBiconnected();
    std::vector<std::vector<size_t>> FindMaximumSpanningTree();
    std::vector<std::vector<size_t>> FindMaximumSpanningTree(const std::map<std::pair<size_t, size_t>, double>& edge_weights);
    static std::vector<Triplet> FindTriplet(const std::vector<std::pair<size_t, size_t>>& edges);
    static std::vector<Triplet> FindTriplet(const std::set<std::pair<size_t, size_t>>& edges);
};





#endif