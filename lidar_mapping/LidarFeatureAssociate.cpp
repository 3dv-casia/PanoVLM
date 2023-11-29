
#include "LidarFeatureAssociate.h"

using namespace std;

std::vector<std::vector<int>> FindNeighborsConsecutive(const std::vector<Velodyne>& lidars , const int neighbor_size)
{
    vector<vector<int>> neighbors_all;
    for(int i = 0; i < int(lidars.size()) - neighbor_size; i++)
    {
        vector<int> neighbors;
        for(int j = i + 1; j < lidars.size() && j <= i + neighbor_size; j++)
            neighbors.push_back(j);
        neighbors_all.push_back(neighbors);
    }
    return neighbors_all;
}

std::vector<std::vector<int>> FindNeighbors(const std::vector<Velodyne>& lidars , const int neighbor_size)
{
    vector<vector<int>> neighbors_all;
    // 记录下每个雷达帧的位置, 为了之后进行近邻搜索
    pcl::PointCloud<pcl::PointXYZI> lidar_center;
    for(size_t i = 0; i < lidars.size(); i++)
    {
        if(!lidars[i].IsPoseValid() || !lidars[i].valid)
            continue;
        pcl::PointXYZI center;
        Eigen::Matrix4d T_wl = lidars[i].GetPose();
        Eigen::Vector3d t_wl = T_wl.block<3,1>(0,3);
        center.x = t_wl.x();
        center.y = t_wl.y();
        center.z = t_wl.z();
        // 设置intensity只是为了知道当前的点对应于哪一帧雷达，因为可能有的雷达没有位姿就没被记录下来
        center.intensity = i;   
        lidar_center.push_back(center);
    }
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_center(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kd_center->setInputCloud(lidar_center.makeShared());
    // 找到最近邻的几个数据，并进行相互关联
    for(size_t i = 0; i < lidars.size(); i++)
    {
        vector<int> neighbors;
        if(lidars[i].IsPoseValid())
        {
            Eigen::Vector3d t_wl = lidars[i].GetPose().block<3,1>(0,3);
            pcl::PointXYZI center;
            center.x = t_wl.x();
            center.y = t_wl.y();
            center.z = t_wl.z();
            center.intensity = i;
            kd_center->nearestKSearch(center, neighbor_size, neighbors, *(new vector<float>()));
            neighbors.erase(neighbors.begin());   // 删去第一个, 因为第一个总是自己
            // 之前的neighbors保存的是在lidar center这个点云里的索引，但不是所有的雷达帧位姿都被保存在那里，所以
            // 点云里的索引不是真正的lidar的索引，每个点的intensity才是这个点对应的LiDAR的索引
            for(int& n_idx : neighbors)
                n_idx = lidar_center[n_idx].intensity;
            // 在初始位姿不太准确的情况下，很有可能当前LiDAR搜索到的近邻LiDAR都是与它在时间上相隔较远的。
            // 对于倾斜放置的LiDAR来说，这种情况就意味着LiDAR之间匹配的特征点会很少。
            // 这是因为倾斜放置后，LiDAR点云和前进方向有着很大的相关性
            // 因此强制加入该雷达前后帧作为其近邻
            set<int> neighbors_set(neighbors.begin(), neighbors.end());
            int neighbor_idx = i - 1;
            while (neighbor_idx >= 0 && !lidars[neighbor_idx].IsPoseValid())
                neighbor_idx--;
            if(neighbor_idx >= 0 && neighbors_set.count(neighbor_idx) == 0)
                neighbors.push_back(neighbor_idx);

            neighbor_idx = i + 1;
            while (neighbor_idx < lidars.size() && !lidars[neighbor_idx].IsPoseValid())
                neighbor_idx++;
            if(neighbor_idx < lidars.size() && neighbors_set.count(neighbor_idx) == 0)
                neighbors.push_back(neighbor_idx);
            
            // 在大尺度场景下，如果有了回环，那么就需要把回环的LiDAR也加入到近邻中
            // 但是可能会出现K近邻搜索到的都是时间上连续的雷达帧，回环帧没有搜索到，因为大场景下可能回环距离较远
            // 所以要主动按照距离搜索，把可能的回环帧加进来
            vector<int> candidate_loop_neighbors;
            double search_radius = 20.0;
            int loop_length = 200;          // 超过200帧才认为是回环
            kd_center->radiusSearch(center, search_radius, candidate_loop_neighbors, *(new vector<float>()));
            for(int& n_idx : candidate_loop_neighbors)
            {
                n_idx = lidar_center[n_idx].intensity;
                int same_loop = 0;
                for(set<int>::iterator it = neighbors_set.begin(); it != neighbors_set.end(); it++)
                {
                    if(abs(n_idx - *it) <= loop_length)
                        same_loop++;
                    if(same_loop >= 2)
                        break;
                }
                if(same_loop < 2 && neighbors_set.count(n_idx) == 0)
                {
                    neighbors.push_back(n_idx);
                    neighbors_set.insert(n_idx);
                }
            }
            
        }
        // 如果某个LiDAR的位姿是不可用的，那就把与它拍摄时间接近的几个LiDAR作为近邻，所以这要求雷达数据必须按照时间顺序存放
        // 应该不会有人乱序存放吧，不会吧 ~ ~ ~ 
        else 
        {
            for(int j = -neighbor_size / 2; j <= neighbor_size / 2; j++)
                neighbors.push_back(i - j);
        }
        neighbors_all.push_back(neighbors);
    }
    return neighbors_all;
}

// 根据直线关联情况判断到底哪些直线之间是真的相互匹配的，这里直线的关联情况就是用矩阵 line_matrix储存的，
// 其中矩阵的第i行第j列的数字k代表 neighbor LiDAR 的第i条直线中有k个点和 reference LiDAR的第j条直线相匹配。
// 根据矩阵找匹配直线的具体方法就是找到当前直线a的所有点相关联
// 的直线{b1,b2,b3,...}，然后找到其中的最大值b，这就意味着当前直线最有可能和b相匹配。
// 然后计算直线a和b之间的夹角，如果夹角比较小就认为这两条直线是匹配的。但这种方法有个问题，可能会存在
// neighbor lidar中的多条直线和reference lidar中的某一条直线相匹配，如果出现了这种问题就要进行一下过滤，
// 过滤方法是出现一对多的情况时，只保留空间距离最近的匹配关系，其余关系舍弃
std::vector<Line2Line> FindAssociations(const Velodyne& ref_lidar, const Velodyne& nei_lidar,
                                        const eigen_vector<Vector6d>& line_coeffs_ref_world,
                                        const eigen_vector<Vector6d>& line_coeffs_nei_world,
                                        const Eigen::MatrixXi& line_matrix,
                                        const bool visualization = false)
{
    vector<Line2Line> associations;
    eigen_map<int, Line2Line> map_associations;
    for(int seg_idx = 0; seg_idx < nei_lidar.edge_segmented.size(); seg_idx++)
    {
        Eigen::MatrixXi::Index max_col;
        int max_count = line_matrix.row(seg_idx).maxCoeff(&max_col);
        if(max_count < nei_lidar.edge_segmented[seg_idx].size() / 2)
            continue;
        // 用平面夹角的计算来代替直线夹角计算，效果是一样的，因为实际上都是计算两个向量的夹角，而且
        // 都要求夹角是锐角. 要注意这里两条直线的方向应该是他们在世界坐标系下的方向，而非各自的局部坐标系下。
        Eigen::Vector3d direction_ref = line_coeffs_ref_world[max_col].block<3,1>(3, 0);
        Eigen::Vector3d direction_nei = line_coeffs_nei_world[seg_idx].block<3,1>(3, 0);
        if(PlaneAngle(direction_ref.data(), direction_nei.data()) * 180.0 / M_PI > 7)
            continue;
        // 这里又使用了局部坐标系下的直线参数
        direction_ref = ref_lidar.segment_coeffs[max_col].block<3,1>(3, 0);
        
        Eigen::Vector3d point_on_line = ref_lidar.segment_coeffs[max_col].block<3,1>(0,0);
        Eigen::Vector3d point_a = 0.1 * direction_ref + point_on_line;
        Eigen::Vector3d point_b = -0.1 * direction_ref + point_on_line;
        
        eigen_map<int, Line2Line>::iterator it = map_associations.find(int(max_col));
        if(it == map_associations.end())
            map_associations.insert({max_col, Line2Line(seg_idx, max_col, point_a, point_b)});
        else 
        {
            // 计算neighbor中相互冲突的两条直线的中点分别到reference直线的距离，保留距离较小的那个
            // 这里只使用了一个点到直线的距离来代替直线之间的距离，可能不太准确，可以考虑使用
            // 所有点到直线的距离，然后求平均
            Eigen::Vector3d point1 = line_coeffs_nei_world[it->second.neighbor_line_idx].block<3,1>(0,0);
            double dis1 = PointToLineDistance3D(point1.data(), line_coeffs_ref_world[max_col].data());
            Eigen::Vector3d point2 = line_coeffs_nei_world[seg_idx].block<3,1>(0,0);
            double dis2 = PointToLineDistance3D(point2.data(), line_coeffs_ref_world[max_col].data());
            if(dis2 < dis1)
            {
                it->second = Line2Line(seg_idx, max_col, point_a, point_b);
            }
        }
    }
    
    for(eigen_map<int, Line2Line>::iterator it = map_associations.begin(); it != map_associations.end(); it++)
        associations.push_back(it->second);

    // 显示直线到直线的特征匹配，用于debug
    if(visualization)
    {
        for(int i = 0; i < associations.size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZRGBL> cloud;
            // 当前neighbor点用白色表示
            pcl::PointXYZRGBL pt;
            pt.r = pt.g = pt.b = 255;
            pt.label = associations[i].neighbor_line_idx;
            for(const PointType& point : nei_lidar.edge_segmented[pt.label])
            {
                pt.x = point.x; pt.y = point.y; pt.z = point.z;
                cloud.push_back(pt);
            }
            // 当前reference用红色点表示
            pt.b = pt.g = 0;
            pt.label = associations[i].ref_line_idx;
            for(const PointType& point : ref_lidar.edge_segmented[pt.label])
            {
                pt.x = point.x; pt.y = point.y; pt.z = point.z;
                cloud.push_back(pt);
            }
            pcl::io::savePCDFileASCII(num2str(ref_lidar.id) + "_" + num2str(nei_lidar.id) + "_line2line_" + num2str(i) + ".pcd", cloud);

        }
    }
    return associations;
}

inline bool CheckLidarCoordinate(const Velodyne& lidar)
{
    if(!lidar.IsInWorldCoordinate())
    {
        LOG(ERROR) << "lidar " << lidar.id << " is not in world coordinate";
        return false;
    }
    return true;
}

inline bool CheckLidarSegment(const Velodyne& lidar)
{
    if(lidar.edge_segmented.empty())
    {
        LOG(WARNING) << "lidar " << lidar.id << " is not segmented or no line points";
        return false;
    }
    return true;
}

eigen_vector<Vector6d> TransformLines(const eigen_vector<Vector6d>& line_coeffs, const Eigen::Matrix4d& T)
{
    eigen_vector<Vector6d> line_coeffs_transformed;
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    for(const Vector6d& coeff : line_coeffs)
    {
        Eigen::Vector3d point = coeff.block<3,1>(0, 0);
        Eigen::Vector3d direction = coeff.block<3,1>(3, 0);
        point = R * point + t;
        direction = R * direction;
        Vector6d coeff_world;
        coeff_world.block<3,1>(0, 0) = point;
        coeff_world.block<3,1>(3, 0) = direction;
        line_coeffs_transformed.push_back(coeff_world);
    }
    return line_coeffs_transformed;
}

vector<Point2Line> AssociatePoint2LineSegmentKNN(const Velodyne& ref_lidar, const Velodyne& nei_lidar, const float dist_threshold,
                                                bool visualization)
{
    vector<Point2Line> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar) || 
        !CheckLidarSegment(ref_lidar) || !CheckLidarSegment(nei_lidar))
    {
        return associations;
    }

    const float sq_dist_threshold = dist_threshold * dist_threshold;

    const eigen_vector<Vector6d>& line_coeffs_ref = ref_lidar.segment_coeffs;
    pcl::KdTreeFLANN<PointType>::Ptr kd_corner(new pcl::KdTreeFLANN<PointType>());
    kd_corner->setInputCloud(ref_lidar.cornerLessSharp.makeShared());

    for(size_t idx = 0; idx < nei_lidar.cornerLessSharp.size(); idx++)
    {
        PointType point = nei_lidar.cornerLessSharp.points[idx];
        
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        const int k_search_size = 5;
        kd_corner->nearestKSearch(point, k_search_size, pointSearchInd, pointSearchSqDis);
        if(pointSearchSqDis[k_search_size - 1] > sq_dist_threshold)
            continue;
        // 统计一下近邻的几个点分别是属于哪些segment，key=segment id，value=当前segment出现的次数
        map<size_t, size_t> seg_count;
        for(int& point_idx : pointSearchInd)
        {
            // 同一个点可能属于多个segment
            for(const size_t& seg_idx : ref_lidar.point_to_segment[point_idx])
                seg_count[seg_idx]++;
        }

        for(map<size_t, size_t>::const_iterator it = seg_count.begin(); it != seg_count.end(); it++)
        {
            // 只有当大部分的近邻点都属于同一个segment时，才认为这种关联是可靠的
            // 也就是说，当前点P的近邻点要都属于同一个segment，否则就认为点到直线的关联不可靠
            if(it->second < k_search_size - 0)
                continue;
            int valid_seg_id = it->first;
            // 直线经过的点和直线方向
            Eigen::Vector3d point_on_line(line_coeffs_ref[valid_seg_id][0], line_coeffs_ref[valid_seg_id][1], line_coeffs_ref[valid_seg_id][2]);
            Eigen::Vector3d unit_direction(line_coeffs_ref[valid_seg_id][3], line_coeffs_ref[valid_seg_id][4], line_coeffs_ref[valid_seg_id][5]);
            Eigen::Vector3d point_a, point_b;
            point_a = 0.1 * unit_direction + point_on_line;
            point_b = -0.1 * unit_direction + point_on_line;
            Eigen::Vector3d lidar_point(point.x, point.y, point.z);
            lidar_point = nei_lidar.World2Local(lidar_point);
            associations.push_back(Point2Line(lidar_point, point_a, point_b));

            // 这是用来显示当前匹配的特征,用于debug
            if(visualization)
            {
                pcl::PointCloud<pcl::PointXYZRGB> cloud;
                // 当前neighbor点用白色表示
                pcl::PointXYZRGB pt;
                pt.r = pt.g = pt.b = 255;
                pt.x = point.x; pt.y = point.y; pt.z = point.z;
                cloud.push_back(pt);
                // 当前点所匹配的直线点用红色表示
                pt.g = pt.b = 0;
                for(const PointType& ref_point : ref_lidar.edge_segmented[valid_seg_id])
                {
                    pt.x = ref_point.x; pt.y = ref_point.y; pt.z = ref_point.z;
                    cloud.push_back(pt);
                }
                // 当前点匹配直线的两个点用绿色表示
                pt.r = 0; pt.g = 255;
                EigenVec2PclPoint(ref_lidar.Local2World(point_a), pt);
                cloud.push_back(pt);
                EigenVec2PclPoint(ref_lidar.Local2World(point_b), pt);
                cloud.push_back(pt);
                pcl::io::savePCDFileASCII(num2str(ref_lidar.id) + "_" + num2str(nei_lidar.id) + "_point2line_" + num2str(associations.size()-1) + ".pcd", cloud);
            }
        }
    }
    return associations;
}

vector<Point2Line> AssociatePoint2LineSegment(const Velodyne& ref_lidar, const Velodyne& nei_lidar, const float dist_threshold,
                                                bool visualization)
{
    vector<Point2Line> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar) || 
        !CheckLidarSegment(ref_lidar) || !CheckLidarSegment(nei_lidar))
    {
        return associations;
    }
    // reference的直线变换到世界坐标系下
    const eigen_vector<Vector6d>& line_coeffs_ref = ref_lidar.segment_coeffs;
    const eigen_vector<Vector6d> line_coeffs_ref_world = TransformLines(ref_lidar.segment_coeffs, ref_lidar.GetPose());
    for(size_t idx = 0; idx < nei_lidar.cornerLessSharp.size(); idx++)
    {
        Eigen::Vector3d point = PclPonit2EigenVecd(nei_lidar.cornerLessSharp.points[idx]);
        double min_distance = numeric_limits<double>::max();
        int valid_seg_id  = -1;
        for(int seg_id = 0; seg_id < line_coeffs_ref_world.size(); seg_id++)
        {
            double distance = PointToLineDistance3D(point.data(), line_coeffs_ref_world[seg_id].data());
            if(distance < min_distance)
            {
                min_distance = distance;
                valid_seg_id = seg_id;
            }
        }
        if(min_distance <= dist_threshold)
        {
            // 直线经过的点和直线方向
            Eigen::Vector3d point_on_line(line_coeffs_ref[valid_seg_id][0], line_coeffs_ref[valid_seg_id][1], line_coeffs_ref[valid_seg_id][2]);
            Eigen::Vector3d unit_direction(line_coeffs_ref[valid_seg_id][3], line_coeffs_ref[valid_seg_id][4], line_coeffs_ref[valid_seg_id][5]);
            Eigen::Vector3d point_a, point_b;
            point_a = 0.1 * unit_direction + point_on_line;
            point_b = -0.1 * unit_direction + point_on_line;
            associations.push_back(Point2Line(nei_lidar.World2Local(point), point_a, point_b));

            // 这是用来显示当前匹配的特征,用于debug
            if(visualization)
            {
                pcl::PointCloud<pcl::PointXYZRGB> cloud;
                // 当前neighbor点用白色表示
                pcl::PointXYZRGB pt;
                pt.r = pt.g = pt.b = 255;
                EigenVec2PclPoint(point, pt);
                cloud.push_back(pt);
                // 当前点所匹配的直线点用红色表示
                pt.g = pt.b = 0;
                for(const PointType& ref_point : ref_lidar.edge_segmented[valid_seg_id])
                {
                    pt.x = ref_point.x; pt.y = ref_point.y; pt.z = ref_point.z;
                    cloud.push_back(pt);
                }
                // 当前点匹配直线的两个点用绿色表示
                pt.r = 0; pt.g = 255;
                EigenVec2PclPoint(ref_lidar.Local2World(point_a), pt);
                cloud.push_back(pt);
                EigenVec2PclPoint(ref_lidar.Local2World(point_b), pt);
                cloud.push_back(pt);
                pcl::io::savePCDFileASCII(num2str(ref_lidar.id) + "_" + num2str(nei_lidar.id) + "_point2line_" + num2str(associations.size()-1) + ".pcd", cloud);
            }
        }
    }
   
    return associations;
}

vector<Line2Line> AssociateLine2LineKNN(const Velodyne& ref_lidar, const Velodyne& nei_lidar, const float dist_threshold,
                                                bool visualization)
{
    vector<Line2Line> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar) || 
        !CheckLidarSegment(ref_lidar) || !CheckLidarSegment(nei_lidar))
    {
        return associations;
    }
    const float sq_dist_threshold = dist_threshold * dist_threshold;
    const eigen_vector<Vector6d>& line_coeffs_ref = ref_lidar.segment_coeffs;
    // 把neighbor和reference的直线变换到世界坐标系下
    eigen_vector<Vector6d> line_coeffs_nei_world = TransformLines(nei_lidar.segment_coeffs, nei_lidar.GetPose());
    eigen_vector<Vector6d> line_coeffs_ref_world = TransformLines(ref_lidar.segment_coeffs, ref_lidar.GetPose());
    pcl::KdTreeFLANN<PointType>::Ptr kd_corner(new pcl::KdTreeFLANN<PointType>());
    kd_corner->setInputCloud(ref_lidar.cornerLessSharp.makeShared());
    // 这里使用了一种其他的特征关联方式，如果直线1中的大部分点都和直线2是相关的，那么就认为这两个直线是相关的
    // 因此这需要遍历所有的直线特征点之后才能确定哪些直线是相关联的，那么就需要一个计数器来记录当前直线相关联的直线
    // 并选择出占多数的直线
    Eigen::MatrixXi line_association(nei_lidar.edge_segmented.size(), ref_lidar.edge_segmented.size());
    line_association.fill(0);

    for(size_t idx = 0; idx < nei_lidar.cornerLessSharp.size(); idx++)
    {
        PointType point = nei_lidar.cornerLessSharp.points[idx];
        
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        const int k_search_size = 5;
        kd_corner->nearestKSearch(point, k_search_size, pointSearchInd, pointSearchSqDis);
        if(pointSearchSqDis[k_search_size - 1] > sq_dist_threshold)
            continue;
        // 统计一下近邻的几个点分别是属于哪些segment，key=segment id，value=当前segment出现的次数
        map<size_t, size_t> seg_count;
        for(int& point_idx : pointSearchInd)
        {
            // 同一个点可能属于多个segment
            for(const size_t& seg_idx : ref_lidar.point_to_segment[point_idx])
                seg_count[seg_idx]++;
        }

        for(map<size_t, size_t>::const_iterator it = seg_count.begin(); it != seg_count.end(); it++)
        {
            // 只有当大部分的近邻点都属于当前的segment时，才认为这种关联是可靠的
            if(it->second < k_search_size - 2)
                continue;
            int valid_seg_id = it->first;

            for(const size_t& seg_idx : nei_lidar.point_to_segment[idx])
                line_association(seg_idx, valid_seg_id) += 1;
        }
    }

    associations = FindAssociations(ref_lidar, nei_lidar, line_coeffs_ref_world, line_coeffs_nei_world, line_association, visualization);
    return associations;
}

vector<Line2Line> AssociateLine2Line(const Velodyne& ref_lidar, const Velodyne& nei_lidar, const float dist_threshold,
                                                bool visualization)
{
    vector<Line2Line> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar) || 
        !CheckLidarSegment(ref_lidar) || !CheckLidarSegment(nei_lidar))
    {
        return associations;
    }
    const float sq_dist_threshold = dist_threshold * dist_threshold;
    const eigen_vector<Vector6d>& line_coeffs_ref = ref_lidar.segment_coeffs;
    // 把neighbor和reference的直线变换到世界坐标系下
    eigen_vector<Vector6d> line_coeffs_nei_world = TransformLines(nei_lidar.segment_coeffs, nei_lidar.GetPose());
    eigen_vector<Vector6d> line_coeffs_ref_world = TransformLines(ref_lidar.segment_coeffs, ref_lidar.GetPose());

    Eigen::MatrixXi line_association(nei_lidar.edge_segmented.size(), ref_lidar.edge_segmented.size());
    line_association.fill(0);

    for(size_t idx = 0; idx < nei_lidar.cornerLessSharp.size(); idx++)
    {
        Eigen::Vector3d point = PclPonit2EigenVecd(nei_lidar.cornerLessSharp.points[idx]);
        for(int seg_id = 0; seg_id < line_coeffs_ref_world.size(); seg_id++)
        {
            double distance = PointToLineDistance3D(point.data(), line_coeffs_ref_world[seg_id].data());
            if(distance > dist_threshold)
                continue;
            for(const size_t& nei_seg_id : nei_lidar.point_to_segment[idx])
            {
                line_association(nei_seg_id, seg_id) += 1;
            }
        }
    }
    associations = FindAssociations(ref_lidar, nei_lidar, line_coeffs_ref_world, line_coeffs_nei_world, line_association, visualization);
    return associations;
}

vector<Point2Line> AssociatePoint2Line(const Velodyne& ref_lidar, const Velodyne& nei_lidar, const float dist_threshold,
                                                bool visualization)
{
    vector<Point2Line> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar))
    {
        return associations;
    }    
    const float sq_dist_threshold = dist_threshold * dist_threshold;
    pcl::KdTreeFLANN<PointType>::Ptr kd_corner(new pcl::KdTreeFLANN<PointType>());
    kd_corner->setInputCloud(ref_lidar.cornerLessSharp.makeShared());
    for(size_t idx = 0; idx < nei_lidar.cornerLessSharp.size(); idx++)
    {
        PointType point = nei_lidar.cornerLessSharp.points[idx];
        
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        const int k_search_size = 5;
        kd_corner->nearestKSearch(point, k_search_size, pointSearchInd, pointSearchSqDis);
        if(pointSearchSqDis[k_search_size - 1] > sq_dist_threshold)
            continue;

        // 判断五个点是否为直线
        eigen_vector<Eigen::Vector3d> nearCorners;
        for (int j = 0; j < k_search_size; j++)
        {
            nearCorners.push_back(PclPonit2EigenVecd(ref_lidar.cornerLessSharp.points[pointSearchInd[j]]));
        }
        Vector6d line_coeff = FormLine(nearCorners, 10.0, 0.05);

        if(line_coeff.isZero())
            continue;
        
        Eigen::Vector3d point_on_line(line_coeff[0], line_coeff[1], line_coeff[2]);
        Eigen::Vector3d unit_direction(line_coeff[3], line_coeff[4], line_coeff[5]);
        Eigen::Vector3d point_a, point_b;
        point_a = 0.1 * unit_direction + point_on_line;
        point_b = -0.1 * unit_direction + point_on_line;
        Eigen::Vector3d lidar_point(point.x, point.y, point.z);
        lidar_point = nei_lidar.World2Local(lidar_point);   // 变回雷达坐标系
        point_a = ref_lidar.World2Local(point_a);
        point_b = ref_lidar.World2Local(point_b);
        associations.push_back(Point2Line(lidar_point, point_a, point_b));    

        // 这是用来显示当前匹配的特征,用于debug
        if(visualization)
        {
            pcl::PointCloud<pcl::PointXYZRGB> cloud;
            // 当前neighbor点用白色表示
            pcl::PointXYZRGB pt;
            pt.r = pt.g = pt.b = 255;
            pt.x = point.x; pt.y = point.y; pt.z = point.z;
            cloud.push_back(pt);
            // 当前点所匹配的直线点用红色表示
            pt.g = pt.b = 0;
            for(const Eigen::Vector3d& ref_point : nearCorners)
            {
                EigenVec2PclPoint(ref_point, pt);
                cloud.push_back(pt);
            }
            // 当前点匹配直线的两个点用绿色表示
            pt.r = 0; pt.g = 255;
            EigenVec2PclPoint(ref_lidar.Local2World(point_a), pt);
            cloud.push_back(pt);
            EigenVec2PclPoint(ref_lidar.Local2World(point_b), pt);
            cloud.push_back(pt);
            pcl::io::savePCDFileASCII(num2str(ref_lidar.id) + "_" + num2str(nei_lidar.id) + "_point2line_" + num2str(associations.size()-1) + ".pcd", cloud);
        }
    }
    return associations;
}

vector<Point2Plane> AssociatePoint2Plane(const Velodyne& ref_lidar, const Velodyne& nei_lidar, 
                            const double plane_tolerance, const float dist_threshold, bool visualization)                                                      
{
    vector<Point2Plane> associations;
    if(!CheckLidarCoordinate(ref_lidar) || !CheckLidarCoordinate(nei_lidar))
    {
        return associations;
    } 
    const float sq_dist_threshold = dist_threshold * dist_threshold;
    // 对当前surf点进行一下过滤和降采样, 减少噪声和数目
    pcl::VoxelGrid<PointType> downsize_surf;
    downsize_surf.setLeafSize(0.2, 0.2, 0.2);
    pcl::PointCloud<PointType> surf_down;
    // downsize_surf.setInputCloud(ref_lidar.surfLessFlat.makeShared());
    // downsize_surf.filter(surf_down);
    surf_down = ref_lidar.surfLessFlat;
    pcl::KdTreeFLANN<PointType>::Ptr kd_surf(new pcl::KdTreeFLANN<PointType>());
    kd_surf->setInputCloud(surf_down.makeShared());

    for(int idx = 0; idx < nei_lidar.surfFlat.points.size(); idx++)
    {
        PointType point = nei_lidar.surfFlat.points[idx];
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        const int k_search_size = 10;
        kd_surf->nearestKSearch(point, k_search_size, pointSearchInd, pointSearchSqDis);

        if (pointSearchSqDis[k_search_size - 1] > sq_dist_threshold)
            continue;
        
        eigen_vector<Eigen::Vector3d> point_local;    // kdtree搜到的近邻点要变回雷达坐标系
        size_t same_point_type = 0;
        // 判断近邻点是否能形成一个平面
        for (int j = 0; j < pointSearchSqDis.size(); j++)
        {
            const PointType& pt = surf_down.points[pointSearchInd[j]];
            same_point_type += (pt.intensity == point.intensity);
            point_local.push_back(ref_lidar.World2Local(PclPonit2EigenVecd(pt)));
        }
        // 要求搜索到的近邻点和当前的点都是同一类，也就是都是地面点或者都不是地面点
        if(same_point_type < k_search_size)
            continue;
        Eigen::Vector4d plane = FormPlane(point_local, plane_tolerance);
        // 形成平面的几个点不是共线的
        Vector6d line_coeff = FormLine(point_local, 3.0);
        if (plane.isZero() || !line_coeff.isZero())
            continue;

        Eigen::Vector3d lidar_point(point.x, point.y, point.z);
        lidar_point = nei_lidar.World2Local(lidar_point);   // 变回雷达坐标系

        associations.push_back(Point2Plane(lidar_point, plane));            
        
        if(visualization)
        {
            pcl::PointCloud<pcl::PointXYZRGB> cloud;
            pcl::PointXYZRGB pt;
            // 当前的neighbor点用白色表示
            pt.r = pt.g = pt.b = 255;
            pt.x = point.x; pt.y = point.y; pt.z = point.z;
            cloud.push_back(pt);
            // 相匹配的reference平面点用红色表示, 这里少用了一个，少的那个在后面用
            pt.g = pt.b = 0;
            for(int i = 0; i < point_local.size() - 1; i++)
            {
                EigenVec2PclPoint(ref_lidar.Local2World(point_local[i]), pt);
                cloud.push_back(pt);
            }
            // 匹配平面的法向量用绿色表示
            pt.r = 0; pt.g = 255;
            EigenVec2PclPoint(ref_lidar.Local2World(point_local[k_search_size - 1]), pt);
            cloud.push_back(pt);
            Eigen::Vector3d norm_point = point_local[k_search_size - 1] + 0.1 * plane.block<3,1>(0,0);
            EigenVec2PclPoint(ref_lidar.Local2World(norm_point), pt);
            cloud.push_back(pt);
            pcl::io::savePCDFileASCII(num2str(ref_lidar.id) + "_" + num2str(nei_lidar.id) + "_point2plane_" + num2str(associations.size()-1) + ".pcd", cloud);

        }
    }
    return associations;
}