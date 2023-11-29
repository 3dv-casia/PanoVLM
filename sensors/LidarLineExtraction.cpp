/*
 * @Author: Diantao Tu
 * @Date: 2022-08-25 15:35:12
 */
#include "LidarLineExtraction.h"


using namespace std;

bool ExpandLine(const pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree, const pcl::PointCloud<pcl::PointXYZI>& cloud, const pcl::PointXYZI& start_point, 
                set<int>& line_points_idx)
{
    bool expand = false;
    eigen_vector<Eigen::Vector3d> line_points;
    
    for(const int& id : line_points_idx)
        line_points.emplace_back(PclPonit2EigenVecd(cloud.points[id]));
    
    double line_length;
    FurthestPoints(line_points, *(new int), *(new int), line_length);
    Vector6d line_coeff = FormLine(line_points, 3.0);
    vector<int> neighbor_idx;
    vector<float> neighbor_sq_distance;
    kd_tree->nearestKSearch(start_point, 5, neighbor_idx, neighbor_sq_distance);
    for(int i = 1; i < neighbor_idx.size(); i++)
    {
        const int& neighbor = neighbor_idx[i];
        if(line_points_idx.count(neighbor) > 0)
            continue;
        if(neighbor_sq_distance[i] > Square(line_length / 2))
            break;
        line_points.emplace_back(PclPonit2EigenVecd(cloud.points[neighbor]));
        // 根据形成直线的长度确定直线外点的阈值，直线越短，那么阈值就越小，直线越长，阈值就越大
        // 阈值的下限是2cm，上限是7cm
        // 下面这四行代码就是找到当前新加入的点与其他已有点之间的最远距离，并且把这个距离和之前得到
        // 的直线长度作对比，选择其中的最大值作为当前直线长度，由此确定阈值
        double curr_line_length = -1;
        for(Eigen::Vector3d& p : line_points)
            curr_line_length = max(curr_line_length, (p - line_points[line_points.size() - 1]).norm());
        curr_line_length = max(curr_line_length, line_length);
        Vector6d curr_line_coeff;
        if(curr_line_length < 2)
        {
            curr_line_coeff = FormLine(line_points, 5.0, 0.07);
            if(curr_line_coeff.isZero())
            {
                line_points.pop_back();
                continue;
            }
        }
        else
        {
            curr_line_coeff = FormLine(line_points, 20.0);
            double line_angle = PlaneAngle(Eigen::Vector3d(curr_line_coeff[3], curr_line_coeff[4], curr_line_coeff[5]).data(),
                Eigen::Vector3d(line_coeff[3], line_coeff[4], line_coeff[5]).data()) * 180.0 / M_PI;
            if(curr_line_coeff.isZero() || line_angle > 1)
            {
                line_points.pop_back();
                continue;
            }
        }
        expand = true;
        line_points_idx.insert(neighbor);
        line_length = curr_line_length;
        line_coeff = curr_line_coeff;
    }
    return expand;
}

std::vector<int> FindNeighbors(vector<bool>& fused, const std::vector<std::vector<int>>& neighbor_idx, size_t line_idx)
{
    set<int> group;
    vector<int> stack;
    vector<int> neighbors = neighbor_idx[line_idx];
    vector<int> group_lines;
    if(neighbors.empty())
    {
        group_lines.push_back(line_idx);
        return group_lines;
    }
    // 记录哪些直线被访问过了
    vector<bool> visited(neighbor_idx.size(), false);
    stack.insert(stack.end(), neighbors.begin(), neighbors.end());
    group.insert(neighbors.begin(), neighbors.end());
    group.insert(line_idx);
    visited[line_idx] = true;
    while (!stack.empty())
    {
        int idx = stack[stack.size() - 1];
        stack.pop_back();
        if(visited[idx])
            continue;
        visited[idx] = true;
        neighbors = neighbor_idx[idx];
        stack.insert(stack.end(), neighbors.begin(), neighbors.end());
        group.insert(neighbors.begin(), neighbors.end());
    }

    for(const int& g : group)
    {
        group_lines.push_back(g);
        fused[g] = true; 
    }
    return group_lines;
}


void FuseLines(const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segments, 
                        const eigen_vector<Vector6d>& line_coeffs,
                        const vector<int>& neighbors, bool optimize,
                        pcl::PointCloud<pcl::PointXYZI>& cloud_fused, Vector6d& line_coeff)
{
    if(neighbors.empty())
    {
        LOG(ERROR) << "neighbors is empty in FuseLines";
        return;
    }
    else if(neighbors.size() == 1)
    {
        cloud_fused = segments[neighbors[0]];
        line_coeff = line_coeffs[neighbors[0]];
        return;
    }
    pcl::PointCloud<pcl::PointXYZI> cloud;
    // 要融合到一起的多个segment里很多点都是重复的，为了减少计算量，就需要除掉重复部分
    // 每个点的intensity是其唯一标识，两个点intensity相同则是同一个点
    set<int> point_id;
    for(const int& n : neighbors)
    {
        for(const pcl::PointXYZI& p : segments[n])
        {
            if(point_id.count(int(p.intensity)) > 0)
                continue;
            point_id.insert(int(p.intensity));
            cloud.push_back(p);
        }
    }
    if(!optimize)
    {
        // 如果选择不优化的话，那就把所有的点都认为是当前直线，直线的参数就直接用最小二乘法计算一个出来
        // 所以在 FormLine 的第二个参数是 0，保证能计算出一个结果
        cloud_fused = cloud;
        eigen_vector<Eigen::Vector3d> line_points;
        for(const auto& p : cloud_fused)
            line_points.push_back(PclPonit2EigenVecd(p));
        line_coeff = FormLine(line_points, 0.0);
    }
    // 如果选择了优化，那么就用RANSAC拟合一条直线
    else
    {
        //创建一个模型参数对象，用于记录结果
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  //inliers表示内点的序号
        pcl::SACSegmentation<pcl::PointXYZI> seg;     // 创建一个分割器     
        pcl::ModelCoefficients coeff;
        seg.setOptimizeCoefficients(false);     // 不使用最小二乘法进行优化
        seg.setModelType(pcl::SACMODEL_LINE);  // Mandatory-设置目标几何形状
        seg.setMethodType(pcl::SAC_RANSAC);     //分割方法：随机采样法
        seg.setDistanceThreshold(0.02);         //设置误差容忍范围，也就是阈值
        seg.setInputCloud(cloud.makeShared());               //输入点云
        seg.segment(*inliers, coeff);   //分割点云，获得平面和法向量

        // 使用RANSAC拟合后，如果内点数很少，就说明这个直线特征不太可靠，那么就返回一个空值
        if(inliers->indices.size() <= 4)
        {
            cloud_fused = pcl::PointCloud<pcl::PointXYZI>();
            line_coeff = Vector6d::Zero();
            return;
        }
        eigen_vector<Eigen::Vector3d> line_points;
        for(const auto& idx : inliers->indices)
        {
            cloud_fused.push_back(cloud.points[idx]);
            line_points.emplace_back(PclPonit2EigenVecd(cloud.points[idx]));
        }
        line_coeff = FormLine(line_points, 5.0);
    }
}

void FuseLineSegments(std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs)
{
    assert(points_each_line.size() == line_coeffs.size());
    const double angle_threshold = 3.0 / 180.0 * M_PI;      // 两直线夹角阈值，超过这个值就认为不是近邻
    // 记录每个segment里点的id是什么，因为两条直线被认为是近邻的条件之一就是有一部分点是重合的
    vector<set<int>> segment_ids;           
    for(const auto& seg : points_each_line)
    {
        set<int> ids;
        for(const auto& p : seg.points)
            ids.insert(int(p.intensity));
        segment_ids.push_back(ids);
    }
    pcl::PointCloud<pcl::PointXYZ> line_centers;
    for(const Vector6d& coeff : line_coeffs)
        line_centers.push_back(pcl::PointXYZ(coeff[0], coeff[1], coeff[2]));
    vector<vector<int>> neighbor_idx(line_centers.size());
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    kd_tree.setInputCloud(line_centers.makeShared());
    for(size_t i = 0; i < line_centers.size(); i++)
    {
        vector<int> curr_neighbor;
        vector<float> sq_dist;
        kd_tree.nearestKSearch(line_centers[i], 5, curr_neighbor, sq_dist);
        for(size_t j = 0; j < curr_neighbor.size(); j++)
        {
            if(sq_dist[j] > 1)
                break;
            int n_idx = curr_neighbor[j];
            // 计算点到直线距离，这里借用了line_coeff的前三维，因为前三个数是直线经过的点，正好作为函数的输入
            double distance = PointToLineDistance3D(line_coeffs[i].data(), line_coeffs[n_idx].data());
            if(distance > 0.2)
                continue;
            distance = PointToLineDistance3D(line_coeffs[n_idx].data(), line_coeffs[i].data());
            if(distance > 0.2)
                continue;
            double line_angle = PlaneAngle(Eigen::Vector3d(line_coeffs[i][3], line_coeffs[i][4], line_coeffs[i][5]).data(),
                                        Eigen::Vector3d(line_coeffs[n_idx][3], line_coeffs[n_idx][4], line_coeffs[n_idx][5]).data());
            if(line_angle > angle_threshold)
                continue;
            const set<int>& curr_ids = segment_ids[i];
            const set<int>& neighbor_ids = segment_ids[n_idx];
            size_t num_same = 0;
            for(const int& id : curr_ids)
                num_same += neighbor_ids.count(id);
            if(num_same <= 2)
                continue;
            neighbor_idx[i].push_back(n_idx);
        }
    }
    vector<pcl::PointCloud<pcl::PointXYZI>> lines_fused;
    eigen_vector<Vector6d> line_coeffs_fused;
    vector<bool> fused(line_centers.size(), false);
    for(size_t line_idx = 0; line_idx < line_centers.size(); line_idx++)
    {
        if(fused[line_idx])
            continue;
        vector<int> group_lines;
        group_lines = FindNeighbors(fused, neighbor_idx, line_idx);
        pcl::PointCloud<pcl::PointXYZI> cloud_fused;
        Vector6d coeff_fused;
        FuseLines(points_each_line, line_coeffs, group_lines, true, cloud_fused, coeff_fused);
        if(!cloud_fused.empty())
        {
            lines_fused.push_back(cloud_fused);
            line_coeffs_fused.push_back(coeff_fused);
        }
    }

    points_each_line.swap(lines_fused);
    line_coeffs.swap(line_coeffs_fused);
}



void FilterLineByLength(std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs)
{
    float distance_threshold = 0.3 ;       // 距离过滤阈值是30cm
    vector<pcl::PointCloud<pcl::PointXYZI>> good_lines;
    eigen_vector<Vector6d> good_line_coeffs;
    for(size_t line_id = 0; line_id < points_each_line.size(); line_id++)
    {
        const pcl::PointCloud<pcl::PointXYZI>& line_points = points_each_line[line_id];
        double max_distance = -1;
        int start = -1, end = -1;
        FurthestPoints(line_points, start, end, max_distance);
        if(max_distance > distance_threshold)
        {
            good_lines.push_back(line_points);
            good_line_coeffs.push_back(line_coeffs[line_id]);
        }
    }
    points_each_line = good_lines;
    line_coeffs = good_line_coeffs;
}

void FilterLineByScan(const std::vector<std::pair<size_t, size_t> >& point_idx_to_image , std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs)
{
    vector<pcl::PointCloud<pcl::PointXYZI>> good_lines;
    eigen_vector<Vector6d> good_line_coeffs;
    for(size_t line_id = 0; line_id < points_each_line.size(); line_id++)
    {
        const pcl::PointCloud<pcl::PointXYZI>& line_points = points_each_line[line_id];
        // 统计当前segment里各个点分别属于哪条扫描线，要求各个点都属于不同的扫描线
        set<size_t> scan_ids;
        for(const auto& p : line_points)
            scan_ids.insert(point_idx_to_image[static_cast<int>(p.intensity)].first);

        if(scan_ids.size() >= line_points.size() / 2 && scan_ids.size() >= 3)
        {
            good_lines.push_back(line_points);
            good_line_coeffs.push_back(line_coeffs[line_id]);
        }
    }
    points_each_line = good_lines;
    line_coeffs = good_line_coeffs;
}

void ExtractLineFeatures(const pcl::PointCloud<pcl::PointXYZI>& edge_points, const std::vector<std::pair<size_t, size_t> >& point_idx_to_image,
    std::vector<pcl::PointCloud<pcl::PointXYZI>>& points_each_line, eigen_vector<Vector6d>& line_coeffs)
                        
{
    if(edge_points.empty())
    {
        LOG(ERROR) << "no edge points in lidar cloud, something wrong";
        return;
    }
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kd_tree->setInputCloud(edge_points.makeShared());

    uchar* visited = (uchar*)malloc(edge_points.size() * sizeof(uchar));
    fill(visited, visited + edge_points.size(), 0);     

    int seg_count = 0;
    // 遍历所有点
    for(size_t idx = 0; idx < edge_points.size(); idx ++)
    {
        if(visited[idx])
            continue;
        visited[idx] = true;
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // 找到距离当前点最近的k+1个点, +1是因为最近的点永远是自己，所以要多找一个点
        kd_tree->nearestKSearch(edge_points.points[idx], 5, pointSearchInd, pointSearchSqDis);
        // 记录当前点能形成的所有的直线
        vector<set<int>> all_segment;
        eigen_vector<Eigen::Vector3d> line_points;
        for(int i = 1; i < pointSearchInd.size() - 1; i++)
        {
            for(int j = i + 1; j < pointSearchInd.size(); j++)
            {
                line_points.clear();
                int idx1 = pointSearchInd[i];
                int idx2 = pointSearchInd[j];
                // 判断当前找到的两个近邻点是否已经被其他的segment包括了，如果已经包括了，那就跳过这两个近邻点
                // 因为如果某个segment已经包含了这两个近邻点，那么以当前两个点为基础向外扩张的话，得到的结果大概率就是
                // 那个已存在的segment，那就会出现重复的segment，没必要重新计算了
                bool skip = false;
                for(const set<int>& seg : all_segment)
                {
                    if(seg.count(idx1) > 0 && seg.count(idx2) > 0)
                    {
                        skip = true;
                        break;
                    }
                }
                if(skip)    continue;
                line_points.push_back(PclPonit2EigenVecd(edge_points.points[idx]));
                line_points.push_back(PclPonit2EigenVecd(edge_points.points[idx1]));
                line_points.push_back(PclPonit2EigenVecd(edge_points.points[idx2]));
                // 判断目前的三个点是否能形成一条直线，不能形成就直接continue
                if(FormLine(line_points, 5.0).isZero())
                    continue;
                set<int> curr_segment = {(int)idx, idx1, idx2};
                int line_start, line_end;
                double line_length;
                FurthestPoints(line_points, line_start, line_end, line_length);
                bool expand1 = true, expand2 = true;
                while(expand1 || expand2)
                {
                    pcl::PointXYZI tmp;
                    EigenVec2PclPoint(line_points[line_start], tmp);
                    expand1 = ExpandLine(kd_tree, edge_points, tmp, curr_segment);
                    EigenVec2PclPoint(line_points[line_end], tmp);
                    expand2 = ExpandLine(kd_tree, edge_points, tmp, curr_segment);
                    line_points.clear();
                    for(const auto& id : curr_segment)
                        line_points.emplace_back(PclPonit2EigenVecd(edge_points.points[id]));
                    FurthestPoints(line_points, line_start, line_end, line_length);
                }
                if(curr_segment.size() >= 5)
                {
                    assert(line_points.size() == curr_segment.size());
                    pcl::PointCloud<pcl::PointXYZI> cloud;
                    for(const int& s : curr_segment)
                    {
                        visited[s] = true;
                        cloud.push_back(edge_points.points[s]);
                    }
                    points_each_line.push_back(cloud);
                    line_coeffs.push_back(FormLine(line_points, 1.0));
                }

            }
        }
    }
    FuseLineSegments(points_each_line, line_coeffs);
    FilterLineByScan(point_idx_to_image, points_each_line, line_coeffs);
    FilterLineByLength(points_each_line, line_coeffs);
    free(visited);
}

