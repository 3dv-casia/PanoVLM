/*
 * @Author: Diantao Tu
 * @Date: 2021-11-28 20:35:21
 */

#include "PoseGraph.h"

using namespace std;

PoseGraph::PoseGraph(const std::vector<std::pair<size_t, size_t>>& edges)
{
    node_map_id.reset(new lemon::ListGraph::NodeMap<uint32_t>(g));

    set<int> nodes;
    for(const pair<size_t, size_t>& e : edges)
    {
        nodes.insert(e.first);
        nodes.insert(e.second);
    }

    // 生成node graph
    map<uint32_t, lemon::ListGraph::Node> id_to_node;
    for ( const auto & node_it : nodes)
    {
        const lemon::ListGraph::Node n = g.addNode();
        (*node_map_id)[n] = node_it;
        id_to_node[node_it] = n;
    }
    // 增加边（edge）
    for (const auto & pair_it : edges)
    {
        g.addEdge( id_to_node[pair_it.first], id_to_node[pair_it.second] );
    }
}

PoseGraph::PoseGraph(const std::set<std::pair<size_t,size_t>>& edges)
{
    node_map_id.reset(new lemon::ListGraph::NodeMap<uint32_t>(g));

    set<int> nodes;
    for(const pair<size_t, size_t>& e : edges)
    {
        nodes.insert(e.first);
        nodes.insert(e.second);
    }

    // 生成node graph
    map<uint32_t, lemon::ListGraph::Node> id_to_node;
    for ( const auto & node_it : nodes)
    {
        const lemon::ListGraph::Node n = g.addNode();
        (*node_map_id)[n] = node_it;
        id_to_node[node_it] = n;
    }
    // 增加边（edge）
    for (const auto & pair_it : edges)
    {
        g.addEdge( id_to_node[pair_it.first], id_to_node[pair_it.second] );
    }
}


set<size_t> PoseGraph::KeepLargestEdgeBiconnected()
{
    using Graph = lemon::ListGraph;
    Graph::EdgeMap<bool> cut_map(g);
    // 计算图中有多少个 边双连通 (bi-edge-connected, 2-edge-connected) 分量
    // 如果有超过两个分量，那么就会出现边割(cut edge)
    // 图论相关可参考 https://oi-wiki.org/graph/concept/#_8
    if(lemon::biEdgeConnectedCutEdges(g, cut_map) > 0)
    {
        Graph::EdgeIt edge_it(g);
        for(Graph::EdgeMap<bool>::MapIt it(cut_map); it != lemon::INVALID; ++it, ++edge_it)
        {
            if(*it)
                g.erase(edge_it);
        }
    }
    // 经过上面的步骤后，所有的节点都是边双连通的，但可能整个图被分割成了很多个互相独立的部分
    // 只保留最大的那个独立部分，其他部分都删去
    int num_component = lemon::countConnectedComponents(g);
    if(num_component < 1)
        return set<size_t>();
    else if(num_component > 1)
    {
        LOG(INFO) << "component count : " << num_component;
    }

    Graph::NodeMap<uint32_t> connect_node_map(g);
    lemon::connectedComponents(g, connect_node_map);
    // 把不同部分的节点保存在一个map中，key是子图的序号，value是各个相互独立的子图
    map<uint32_t, set<Graph::Node>> map_subgraphs;
    Graph::NodeIt node_it(g);
    for(Graph::NodeMap<uint32_t>::MapIt it(connect_node_map); it != lemon::INVALID; ++it, ++node_it)
        map_subgraphs[*it].insert(node_it);
    
    // 如果有多个部分，就输出每个部分的图像id，为了debug，看看是哪里产生了中断
    // 因为理论上应该只有一个component
    if(num_component > 1)
    {
        ofstream f("components.txt");
        for(map<uint32_t, set<Graph::Node>>::iterator it = map_subgraphs.begin(); it != map_subgraphs.end(); ++it)
        {
            for(const Graph::Node& node : it->second)
                f << (*node_map_id)[node] << " ";
            f << endl;
        }
        f.close();
    }

    // 找到节点最多的子图
    size_t count = 0;
    map<uint32_t, set<Graph::Node>>::iterator largest_subgraph = map_subgraphs.end();
    for(map<uint32_t, set<Graph::Node>>::iterator it = map_subgraphs.begin(); it != map_subgraphs.end(); ++it)
    {
        if(it->second.size() > count)
        {
            count = it->second.size();
            largest_subgraph = it;
        }
    }
    
    if(largest_subgraph == map_subgraphs.end())
        return set<size_t>();
    const set<Graph::Node>& node_set = largest_subgraph->second;
    set<size_t> remain_node_index;
    for(const Graph::Node& node : node_set)
    {
        const uint32_t id = (*node_map_id)[node];
        remain_node_index.insert(id);
    }
    return remain_node_index;
}

std::vector<std::vector<size_t> > PoseGraph::FindMaximumSpanningTree()
{
    assert(lemon::countConnectedComponents(g) == 1);
    // 每条边的权重，默认设置为1
    lemon::ListGraph::EdgeMap<double> weight_map(g);
    for(lemon::ListGraph::EdgeIt edge_it(g); edge_it != lemon::INVALID; ++edge_it)
        weight_map[edge_it] = 1;
    
    vector<lemon::ListGraph::Edge> tree_edges;
    lemon::kruskal(g, weight_map, std::back_inserter(tree_edges));
    // 把克鲁斯卡尔算法生成的最小生成树转成另一个形式输出
    vector<vector<size_t>> spanning_tree(lemon::countNodes(g));
    for(size_t i = 0; i < tree_edges.size(); i++)
    {
        const lemon::ListGraph::Edge& edge = tree_edges[i];
        const size_t u = (*node_map_id)[g.u(edge)];
        const size_t v = (*node_map_id)[g.v(edge)];
        spanning_tree[u].push_back(v);
        spanning_tree[v].push_back(u);
    }
    return spanning_tree;
}

std::vector<std::vector<size_t>> PoseGraph::FindMaximumSpanningTree(const std::map<std::pair<size_t, size_t>, double>& edge_weights)
{
    assert(lemon::countConnectedComponents(g) == 1);
    lemon::ListGraph::EdgeMap<double> weight_map(g);
    for(lemon::ListGraph::EdgeIt edge_it(g); edge_it != lemon::INVALID; ++edge_it)
    {
        const size_t u = (*node_map_id)[g.u(edge_it)];
        const size_t v = (*node_map_id)[g.v(edge_it)];
        map<pair<size_t, size_t>, double>::const_iterator it = edge_weights.find({min(u,v), max(u,v)});
        if(it != edge_weights.end())
            weight_map[edge_it] = it->second;
        else
            weight_map[edge_it] = 1e8;
    }

    vector<lemon::ListGraph::Edge> tree_edges;
    lemon::kruskal(g, weight_map, std::back_inserter(tree_edges));
    // 把克鲁斯卡尔算法生成的最小生成树转成另一个形式输出
    vector<vector<size_t>> spanning_tree(lemon::countNodes(g));
    for(size_t i = 0; i < tree_edges.size(); i++)
    {
        const lemon::ListGraph::Edge& edge = tree_edges[i];
        const size_t u = (*node_map_id)[g.u(edge)];
        const size_t v = (*node_map_id)[g.v(edge)];
        spanning_tree[u].push_back(v);
        spanning_tree[v].push_back(u);
    }
    return spanning_tree;
}


std::vector<Triplet> PoseGraph::FindTriplet(const std::vector<std::pair<size_t, size_t>>& edges)
{
    set<pair<size_t,size_t>> edges_set(edges.begin(), edges.end());
    return FindTriplet(edges_set);
}

std::vector<Triplet> PoseGraph::FindTriplet(const std::set<std::pair<size_t, size_t>>& edges)
{
    vector<Triplet> triplets;
    // 建立一个邻接图，保存每个顶点以及与该顶点直接相连的顶点 key=顶点  value=顶点的邻接点
    unordered_map<uint32_t, set<uint32_t>> adjacency;
    for(const pair<size_t, size_t>& e : edges)
    {
        adjacency[e.first].insert(e.second);
        adjacency[e.second].insert(e.first);
    }

    vector<uint32_t> node_candidate;
    // 遍历每一条边，然后根据上面的邻接图找到和这一条边的两个顶点都相连的顶点，那么这三个顶点构成了一个triplet
    for(const pair<size_t, size_t>& e : edges)
    {
        const set<uint32_t>& neighbor1 = adjacency.find(e.first)->second;
        const set<uint32_t>& neighbor2 = adjacency.find(e.second)->second;
        node_candidate.clear();
        set_intersection(neighbor1.cbegin(), neighbor1.cend(), 
                    neighbor2.cbegin(), neighbor2.cend(), back_inserter(node_candidate));
        for(const uint32_t& node : node_candidate)
        {
            vector<uint32_t> node_idx = {node, static_cast<uint32_t>(e.first), static_cast<uint32_t>(e.second)};
            sort(node_idx.begin(), node_idx.end());
            triplets.emplace_back(Triplet(node_idx[0], node_idx[1], node_idx[2]));
        }
        // 已经找完了所有关于当前边的triplet，那么就要把这条边的关系删除掉，不然后面会出现问题重复的triplet
        adjacency[e.first].erase(e.second);
        adjacency[e.second].erase(e.first);
    }
    return triplets;
}
