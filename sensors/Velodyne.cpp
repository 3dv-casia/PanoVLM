
#include "Velodyne.h"
#include "../util/Visualization.h"

using std::atan2;
using std::cos;
using std::sin;
using namespace std;


// 计算两个序列的最长公共子序列的长度
template <typename T>
int LongestCommonSubsequence(const vector<T>& str1, const vector<T>& str2)
{
    int len1 = str1.size(), len2 = str2.size();
    // 动态规划表，大小(m+1)*(n+1)
	vector<vector<int>> table(len1 + 1, vector<int>(len2 + 1, 0));  
	for(int i = 1; i <= len1; i++)
	{
		for(int j = 1; j <= len2; j++)
		{
			// 第一行和第一列置0
			if (i == 0 || j == 0)
				table[i][j] = 0;
			else if(str1[i-1] == str2[j-1])
				table[i][j] = table[i-1][j-1] + 1;
			else
				table[i][j] = max(table[i-1][j], table[i][j-1]);
		}
	}
	return table[len1][len2];
}


Velodyne::Velodyne(int scan, int _id, int _horizon_scan): 
                N_SCANS(scan), world(false),scanPeriod(0.1),id(_id),valid(true),horizon_scans(_horizon_scan)
{
    cloud.clear();
    cloud_scan.clear();
    cornerSharp.clear();
    cornerLessSharp.clear();
    surfFlat.clear();
    surfLessFlat.clear();
    edge_segmented.clear();
    cornerBeforeFilter.clear();
    
    scanStartInd.resize(N_SCANS);
    scanEndInd.resize(N_SCANS);

    R_wl = Eigen::Matrix3d::Zero();
    t_wl = std::numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
    T_wc_wl = Eigen::Matrix4d::Identity();      // kitti
    
    cloudCurvature = NULL;
    cloudSortInd = NULL;
    cloudState = NULL;
    left_neighbor = NULL;
    right_neighbor = NULL;
}

Velodyne::Velodyne(): world(false),scanPeriod(0.1),valid(true),N_SCANS(0),id(-1)
{
    cloud.clear();
    cloud_scan.clear();
    cornerSharp.clear();
    cornerLessSharp.clear();
    surfFlat.clear();
    surfLessFlat.clear();
    scanStartInd.resize(N_SCANS);
    scanEndInd.resize(N_SCANS);
    R_wl = Eigen::Matrix3d::Zero();
    t_wl = std::numeric_limits<double>::infinity() * Eigen::Vector3d::Ones();
    T_wc_wl = Eigen::Matrix4d::Identity();  // kitti
    cloudCurvature = NULL;
    cloudSortInd = NULL;
    cloudState = NULL;
    left_neighbor = NULL;
    right_neighbor = NULL;
}

Velodyne::~Velodyne()
{
    cornerSharp.clear();
    cornerLessSharp.clear();
    surfFlat.clear();
    surfLessFlat.clear();
    cloud.clear();
    cloud_scan.clear();
}


void Velodyne::LoadLidar(string file_path)
{
    if(file_path.empty())
        file_path = name;
    string::size_type pos = file_path.rfind('.');
    string type = file_path.substr(pos);
    if(type == ".ply")
    {
        if(pcl::io::loadPLYFile(file_path, cloud) == -1)
        {
            LOG(ERROR) << "Fail to load lidar data at " << file_path << endl;
            return;
        }
    }
    else if(type == ".pcd")
    {    
        if(pcl::io::loadPCDFile(file_path, cloud) == -1)
        {
            LOG(ERROR) << "Fail to load lidar data at " << file_path << endl;
            return;
        }
    }
    else
    {
        LOG(ERROR) << "unknown point cloud format, only .ply or .pcd" << endl;
        return;
    }
    name = file_path;
    //从点云中移除NAN点也就是无效点
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud, cloud, indices);
    cloud = removeClosedPointCloud(0.5);

    // 改变LiDAR的坐标系，原本是X向右，Y向前，Z向上
    // 变成 X向右，Y向下，Z向前，这种顺序适应相机的坐标系
    Eigen::Matrix4f T_cam_lidar;
    T_cam_lidar << 1, 0, 0, 0 ,
                   0, 0, -1, 0,
                   0, 1, 0, 0,
                   0, 0, 0, 1;
    pcl::transformPointCloud(cloud, cloud, T_cam_lidar);

    if(cloud.size() < 4000)
    {
        LOG(ERROR) << "lidar " << id << " is invalid, only " << cloud.size() << " points in point cloud";
        valid = false;
    }

}

// 移除距离坐标系原点太近的点
pcl::PointCloud<pcl::PointXYZI> Velodyne::removeClosedPointCloud(float threshold)
{
    pcl::PointCloud<pcl::PointXYZI> cloud_out;
    float sq_threshold = threshold * threshold;
    cloud_out.points.resize(cloud.points.size());
    size_t j = 0;

    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        float dis = cloud.points[i].x * cloud.points[i].x + cloud.points[i].y * cloud.points[i].y + cloud.points[i].z * cloud.points[i].z;
        if (dis < sq_threshold)
            continue;
        cloud_out.points[j] = cloud.points[i];
        j++;
    }
    if (j != cloud.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;

    return cloud_out;
}

int Velodyne::VerticalAngleToScanID(const float& vertical_angle, const int& max_scan)
{
    int scanID = -1;
    if (max_scan == 16)
    {
        #if 1
        scanID = int((vertical_angle + 15) / 2 + 0.5);
        if (scanID > (max_scan - 1) || scanID < 0)
            scanID = -1;
        #else
        for(scanID = 0; scanID < 16; scanID++)
        {
            if(vertical_degree_VLP16[scanID] >= vertical_angle)
                break;
        }
        int near_ID = scanID == 0 ? 1 : scanID - 1;
        if(abs(vertical_degree_VLP16[scanID] - vertical_angle) > abs(vertical_degree_VLP16[near_ID] - vertical_angle))
            scanID = near_ID;
        #endif
    }
    else if (max_scan == 32)
    {
        scanID = int((vertical_angle + 92.0/3.0) * 3.0 / 4.0);
        if (scanID > (max_scan - 1) || scanID < 0)
            scanID = -1;
    }
    else if (max_scan == 64)
    {   
        if (vertical_angle >= -8.83)
            scanID = int((2 - vertical_angle) * 3.0 + 0.5);
        else
            scanID = max_scan / 2 + int((-8.83 - vertical_angle) * 2.0 + 0.5);

        // use [0 50]  > 50 remove outlies 
        if (vertical_angle > 2 || vertical_angle < -24.33 || scanID > 50 || scanID < 0)
        
            scanID = -1;
    }
    else
        LOG(ERROR) << "Only support 16, 32, 64 scan lidar" << endl;
    return scanID;
}

void Velodyne::ReOrderVLP2()
{
    point_idx_to_image.clear();
    image_to_point_idx.clear();

    image_to_point_idx.resize(N_SCANS, vector<int>(horizon_scans, -1));


    if(world)
        LOG(INFO) << "data now in world coordinate, reorder is not accurate" << endl;
    
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        LOG(ERROR) << "only support velodyne with 16, 32 or 64 scan line!\n";
        return;
    }

    double horizon_resolution = 2.0 * M_PI / horizon_scans;
    range_image.resize(N_SCANS, horizon_scans);
    range_image.fill(0);

    // 相当于把点云投影到一个二维的图像上，每个点对应一个像素, 每个像素的值是点云中的点的索引
    vector<vector<int>> scan_point_idx(N_SCANS, vector<int>(horizon_scans, -1));
    
    // atan2 范围是 (-pi, pi]
    double start_ori = atan2(cloud.points[0].x, cloud.points[0].z);
    if(start_ori < 0)
        start_ori += 2 * M_PI;
    
    // VLP雷达点云的存储顺序并不是第0线，第1线，第2线，第3线......这样的顺序，而是跳着存储的
    // 第0线，第8线，第1线，第9线，第2线，第10线.......
    // 这样就可以根据线数确定当前是这一列扫描的第几根, 
    // 例如当前根据角度计算得到为第0线，那么第0线对应着是第0次扫描，
    // 如果根据角度得到第13线，那么第13线对应着是第11次扫描
    vector<int> scanID_to_scan_order(N_SCANS);
    if(N_SCANS == 16)
    {
        for(int i = 0; i <= 7; i++)
            scanID_to_scan_order[i] = 2 * i;
        for(int i = 8; i <= 15; i++)
            scanID_to_scan_order[i] = 2 * i - 15;
    }
    vector<int> scan_order_to_scanID(N_SCANS);
    for(int i = 0; i < N_SCANS; i++)
        scan_order_to_scanID[scanID_to_scan_order[i]] = i;

   
    PointType point;
    int last_scan = -1;
    vector<int> scan_order_current_col;
    vector<int> point_idx_current_col;
    vector<int> col_id_each_point;
    for (int i = 0; i < cloud.points.size(); i++)
    {
        point = cloud.points[i];

        float vertical_angle = atan(-point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
        int scanID = VerticalAngleToScanID(vertical_angle, N_SCANS);
        if(scanID == -1)
            continue;
        // 防止数组越界
        if(i == 0)
            last_scan = scanID;
        
        // 通过当前的scan来判断当前点是否已经属于新的一列了
        // 开始新的一列后，就要处理上一列的数据
        if(scanID_to_scan_order[scanID] < scanID_to_scan_order[last_scan])
        {
            // 当前列的点所在的扫描线数应该是互不相同的，如果有两个点在同一根线上，那么就说明某一个点错了
            vector<int> scan_occupation(N_SCANS, 0);
            for(int& scan : scan_order_current_col)
                scan_occupation[scan] += 1;
            for(int j = 0; j < scan_occupation.size(); j++)
            {
                if(scan_occupation[j] <= 1)
                    continue;
                int conflict_scan = j;
                // 找到冲突的扫描线所在的索引
                vector<int> conflict_idx;
                for(int k = 0; k < scan_order_current_col.size(); k++)
                    if(scan_order_current_col[k] == conflict_scan)
                        conflict_idx.push_back(k);
                // 当出现两个点在同一个扫描线上时,需要判断这两个点中是哪个点错误
                // 可以通过改变两个点的所在的扫描线得到新的扫描顺序,然后判断哪个扫描顺序是正确的
                // 这里会产生两个新的扫描顺序,第一种是把第一个 conflict scan 的点的扫描线改为 conflict scan - 1
                // 第二种是把第二个 conflict scan 的点的扫描线改为 conflict scan + 1
                // 注意:这里只考虑了比较简单的冲突情况,即只有两个点冲突,并且只有一个扫描线上有两个点冲突,而且这两个冲突点中有一个是正确的
                // 这里比较两种新的扫描序列哪个更好，使用的方法是最长公共子序列，即LCS，常见的动态规划问题
                scan_order_current_col[conflict_idx[0]] = conflict_scan - 1;
                int similiarity1 = conflict_scan - 1 >= 0 ? LongestCommonSubsequence(scan_order_current_col, scan_order_to_scanID) : 0;
                scan_order_current_col[conflict_idx[0]] = conflict_scan;        // 恢复原来的扫描顺序
                scan_order_current_col[conflict_idx[1]] = conflict_scan + 1;
                int similiarity2 = conflict_scan + 1 < N_SCANS ? LongestCommonSubsequence(scan_order_current_col, scan_order_to_scanID) : 0;
                scan_order_current_col[conflict_idx[1]] = conflict_scan;        // 恢复原来的扫描顺序
                if(similiarity1 > similiarity2)
                    scan_order_current_col[conflict_idx[0]] = conflict_scan - 1;
                else
                    scan_order_current_col[conflict_idx[1]] = conflict_scan + 1;
            }

            pair<int, int> max_col = MaxNum(col_id_each_point);
            int col_idx = max_col.first;
            
            
            for(int j = 0; j < scan_order_current_col.size(); j++)
            {
                int scan = scan_order_current_col[j];
                int point_idx = point_idx_current_col[j];
                scan_point_idx[scan][col_idx] = point_idx;
                
            }
            scan_order_current_col.clear();
            point_idx_current_col.clear();
            col_id_each_point.clear();
        }

        // 把角度都变换到 0-2pi
        double ori = atan2(point.x, point.z);
        if(ori < 0)
            ori += 2 * M_PI;
        
        double diff_ori = ori - start_ori;
        if(diff_ori < 0)
            diff_ori += 2 * M_PI;
        int row_index = scanID;
        int col_index = round(diff_ori / horizon_resolution);
        scan_order_current_col.push_back(row_index);
        col_id_each_point.push_back(col_index);
        point_idx_current_col.push_back(i);
        last_scan = scanID;
    }
    
    
    // 每个scan的前5个点和后5个点都不算
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = cloud_scan.size() + 5;
        for(int j = 0; j < horizon_scans; j++)
        {
            int point_idx = scan_point_idx[i][j];
            if(point_idx == -1)
                continue;
            PointType point(cloud.points[point_idx]);
            point.intensity = i;
            
            range_image(i,j) = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            point_idx_to_image.push_back({i,j});
            image_to_point_idx[i][j] = cloud_scan.size();
            cloud_scan.push_back(point);
        }
        
        scanEndInd[i] = cloud_scan.size() - 6;
    }

    // 可视化range image，只用于debug
    // VisualizeRangeImage("range_image.png", range_image, 15, 0.5);
}

void Velodyne::ReOrderVLP()
{
    if(!valid)
        return;
    // 如果已经进行过reorder了，就不用第二次进行了
    if(!cloud_scan.empty())
        return;
    point_idx_to_image.clear();
    image_to_point_idx.clear();

    image_to_point_idx.resize(N_SCANS, vector<int>(horizon_scans, -1));

    if(world)
        LOG(INFO) << "data now in world coordinate, reorder is not accurate" << endl;
    // cout << "scanRengistratione \n";
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        LOG(ERROR) << "only support velodyne with 16, 32 or 64 scan line!\n";
        return;
    }

    double horizon_resolution = 2.0 * M_PI / horizon_scans;
    range_image.resize(N_SCANS, horizon_scans);
    range_image.fill(0);
    
    int cloudSize = cloud.points.size();
    
    // atan2 范围是 (-pi, pi]
    double start_ori = atan2(cloud.points[0].x, cloud.points[0].z);
    if(start_ori < 0)
        start_ori += 2 * M_PI;
    
    // VLP雷达点云的存储顺序并不是第0线，第1线，第2线，第3线......这样的顺序，而是跳着存储的
    // 第0线，第8线，第1线，第9线，第2线，第10线.......
    // 这样就可以根据线数确定当前是这一列扫描的第几根, 例如当前根据角度计算得到为第0线，那么第0线对应着是第0次扫描
    // 如果根据角度得到第13线，那么第13线对应着是第11次扫描
    map<int, int> scan_order;
    if(N_SCANS == 16)
    {
        for(int i = 0; i <= 7; i++)
            scan_order[i] = 2 * i;
        for(int i = 8; i <= 15; i++)
            scan_order[i] = 2 * i - 15;
    }

    // 当前扫描是否已经通过z轴正方向了，在一个点云里，只能通过z轴正方向一次
    bool cross_z_axis = false;
    // 上一个点与z轴的夹角
    double last_ori = -1;
    
    int count = cloudSize;
    PointType point;
    // 记录每一个scan的点云，后期要把它合起来成为一个完整的点云
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // 记录每个scan中第i个点属于range image的哪一列，因为可能第一行只有1750个点，但是理论上应该有1800个点，
    // 因此第i个点不一定是在第i列，中间是有间隔的
    // {point idx in current scan, col index}
    vector<vector<pair<size_t, size_t>>> point_idx_to_col(N_SCANS);
    int col_offset = 0;

    int last_col = 0, last_scan = -1;
    size_t cloud_scan_count = 0;
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = cloud.points[i].x;
        point.y = cloud.points[i].y;
        point.z = cloud.points[i].z;
        point.intensity = cloud.points[i].intensity;

        float vertical_angle = atan(-point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
        int scanID = VerticalAngleToScanID(vertical_angle, N_SCANS);
        if(scanID == -1)
        {
            count--;
            continue;
        }

        // 把角度都变换到 0-2pi
        double ori = atan2(point.x, point.z);
        if(ori < 0)
            ori += 2 * M_PI;
        // 因为所有的夹角都是0-2pi，因此如果当前的夹角小于了上一个点的夹角，就说明当前点跨过了z轴
        // 但这一切的前提是LiDAR的测量都是准确无误的，如果出现一些小概率情况导致测量错误，那么这显然
        // 就会导致后面的所有点出现错误，所以需要一定的保险，确定真的是跨过了z轴而非测量噪声
        // 保险措施就是当检测到跨过z轴后，要连续判断后面的N个点（N=雷达线数），如果这后面的N个点里有一半以上的点
        // 都是不满足跨过z轴的条件的，那么就认为当前跨过z轴的判断是错误的，一切恢复原状。
        if(ori < last_ori  && cross_z_axis == false)
        {
            int reliable = 0;
            int reliable_threshold = N_SCANS ;
            for(int idx = i + 1; idx < i + N_SCANS + 1 && idx < cloudSize; idx++)
            {
                double angle = atan2(cloud.points[idx].x, cloud.points[idx].z);
                if(angle < 0)
                    angle += 2 * M_PI;
                reliable += (angle < last_ori);
                if(reliable >= reliable_threshold)
                    break;
            }
            cross_z_axis = (reliable >= reliable_threshold);
        }
        ori += 2 * M_PI * cross_z_axis;

        double time = ori - start_ori;

        int row_index = scanID;
        int col_index = round((ori - start_ori) / horizon_resolution);
        
        // 通过当前的scan来判断当前点是否已经属于新的一列了
        // 开始新的一列后，判断当前的列数和上一列的列数是否相同，如果相同就说明当前的列数算错了
        // 需要手动增加1, 然后更新当前列数并保存在last_col里，准备下次再来用
        if(scan_order[scanID] < scan_order[last_scan])
        {
            col_offset = (last_col == col_index);
            last_col = col_index + col_offset;
        }
        last_scan = scanID;
        col_index += col_offset;

        // 通过实验发现，使用PCL库读取的VLP雷达数据含有一些“噪声”，也就是说会出现属于同一列的点有的向左偏一些，
        // 有的向右偏一些，造成根据旋转角度确定该点在range image的列数出现一点点误差，尤其是在上面判断是否跨过z轴的时候，
        // 会出现在某些奇怪的地方跨过了z轴，在之后又真的跨过了z轴，这就导致整体上旋转角度会额外增加720度，那么相应的，
        // 列数也会增加3600列，因此这里要用一个while语句来循环，让该点永远在正确范围内
        while(col_index >= horizon_scans)
            col_index -= horizon_scans;
        if(col_index < 0)       // 偶尔雷达出错的时候会出现这种情况
            continue;

        point.intensity = scanID;
        range_image(row_index, col_index) = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        point_idx_to_col[scanID].push_back(pair<size_t, size_t>(laserCloudScans[scanID].size(), col_index));
        
        laserCloudScans[scanID].push_back(point); 
        cloud_scan_count++;
        last_ori = ori;
    }
    
    point_idx_to_image.resize(cloud_scan_count);
    
    // 每个scan的前5个点和后5个点都不算
    for (int i = 0; i < N_SCANS; i++)
    { 
        for(const pair<size_t, size_t>& idx_col : point_idx_to_col[i])
        {
            // 目前已有的点的数量+当前点在自己的scan的顺序 = 当前点在整体点云中的顺序
            const size_t point_idx = cloud_scan.size() + idx_col.first;
            point_idx_to_image[point_idx] = pair<size_t, size_t>(i, idx_col.second);
            image_to_point_idx[i][idx_col.second] = point_idx;
        }
        scanStartInd[i] = cloud_scan.size() + 5;
        cloud_scan += laserCloudScans[i];
        scanEndInd[i] = cloud_scan.size() - 6;
    }
    // 可视化range image，只用于debug
    // VisualizeRangeImage("range_image.png", range_image, 15, 0.5);
}


// 提取特征点，完成后cornerSharp  cornerLessSharp  surfFlat  surfLessFlat 就都有值了
// 这段是从 A-LOAM 里抄来的
void Velodyne::ExtractFeatures(float max_curvature, float intersect_angle_threshold, 
                                int method, bool segment )
{
    if(!valid)
        return;
    if(cloud_scan.empty())
    {
        LOG(ERROR) << "cloud_scan is empty, call Reorder first" << endl;
        return;
    }
    // 如果已经提取过特征点了，就不用再提取了
    if(!cornerLessSharp.empty() || !surfLessFlat.empty())
        return;
    
    int cloudSize = cloud_scan.points.size();
    
    if(segment)
        Segmentation();
    
    if(cloud_scan.points.size() < cloudSize * 0.1)
    {
        LOG(WARNING) << "LiDAR data " << id << " has something wrong";
        valid = false;
        return;
    }
    cloudSize = cloud_scan.points.size();
    size_t neighbor_size = 5;

    cloudCurvature = (float*)malloc(cloudSize * sizeof(float));
    cloudState = (int*)malloc(cloudSize * sizeof(int));
    cloudSortInd = (int*)malloc(cloudSize * sizeof(int));
    left_neighbor = (int*)malloc(cloudSize * sizeof(int));
    right_neighbor = (int*)malloc(cloudSize * sizeof(int));
    // 把曲率初始化成-1
    fill(cloudCurvature, cloudCurvature + cloudSize, -1);     
    fill(cloudState, cloudState + cloudSize, POINT_NORMAL);
    fill(left_neighbor, left_neighbor + cloudSize, -1);
    fill(right_neighbor, right_neighbor + cloudSize, -1);
    for(int idx = 0; idx < cloudSize; idx ++)
    {
        cloudSortInd[idx] = idx;
    }
    
    // 把地面点分割出来
    // SegmentGround();

    // MarkOccludedPoints();
    int* cloudState_preserve = (int*)malloc(cloudSize * sizeof(int));
    memcpy(cloudState_preserve, cloudState, cloudSize * sizeof(int));

    // 计算每个点到原点的距离, 保存成一个数组是为了后面方便计算曲率
    float* cloudDistance;
    cloudDistance = (float*)malloc(cloudSize * sizeof(float));
    for(size_t i = 0; i < cloudSize; i++)
    {
        cloudDistance[i] = range_image(point_idx_to_image[i].first, point_idx_to_image[i].second);
    }
    
    // 计算每个点的弯曲度
    // lego loam 论文版本
    // for(int idx = neighbor_size; idx < cloudSize - neighbor_size; idx ++)
    // {
    //     float diff_depth = 0;
    //     for(int i = 1; i <= neighbor_size; i++)
    //     {
    //         diff_depth += (cloudDistance[idx - i] + cloudDistance[idx + i]);
    //     }
    //     diff_depth -= 2 * neighbor_size * cloudDistance[idx];
    //     // diff_depth /= 2 * neighbor_size;
    //     diff_depth /= cloudDistance[idx];
    //     diff_depth = diff_depth * diff_depth;

    //     cloudCurvature[idx] = abs(diff_depth);
        
    // }
    // 计算每个点的曲率，有不同的方法
    // 原版（lego loam代码版本）
    if(method == LOAM || method == DOUBLE_EXTRACTION)
    {
        for(int idx = neighbor_size; idx < cloudSize - neighbor_size; idx ++)
        {
            float diff_depth = 0;
            for(int i = 1; i <= neighbor_size; i++)
            {
                diff_depth += (cloudDistance[idx - i] + cloudDistance[idx + i]);
            }
            diff_depth -= 2 * neighbor_size * cloudDistance[idx];
            // diff_depth /= 2 * neighbor_size;
            diff_depth = diff_depth * diff_depth;
            cloudCurvature[idx] = abs(diff_depth);  
        }
    }
    else if(method == ADAPTIVE)
    {
        for(int scan_idx = 0; scan_idx < N_SCANS; scan_idx ++)
        {
            if(scanEndInd[scan_idx] - scanStartInd[scan_idx] < neighbor_size)
                continue;
            for(int idx = scanStartInd[scan_idx]; idx <= scanEndInd[scan_idx]; idx ++)
            {
                int left_idx = idx - neighbor_size;
                int right_idx = idx + neighbor_size;
                // 当前点左侧的点距离当前点至少要有10cm，因为在小场景下雷达扫描会比较密集，导致边缘处有很多个点，
                // 如果仅仅用左侧n个点，右侧n个点这种方式，不容易计算出边缘处的曲率，因为点与点之间离得太近了，导致曲率很小
                // 0.08 * 0.08 = 0.0064
                while(left_idx >= scanStartInd[scan_idx] && PointDistanceSquare(cloud_scan.points[left_idx], cloud_scan.points[idx]) < 0.0064)
                    left_idx --;
                while(left_idx <= scanEndInd[scan_idx] && PointDistanceSquare(cloud_scan.points[right_idx], cloud_scan.points[idx]) < 0.0064)
                    right_idx ++;
                int max_diff_idx = max(idx - left_idx, right_idx - idx);
                left_idx = idx - max_diff_idx;
                right_idx = idx + max_diff_idx;
                if(!IsInside(left_idx, scanStartInd[scan_idx] - 5, scanEndInd[scan_idx] + 5) || !IsInside(left_idx, scanStartInd[scan_idx] - 5, scanEndInd[scan_idx] + 5))
                    continue;
                float diff_depth = 0;
                for(int i = left_idx; i <= right_idx; i++)
                    diff_depth += cloudDistance[i];
                // 这里要有个+1，因为上面计算的diff depth 是包含了一个当前点的深度的，也就是 i = idx的时候，要把这个给减掉
                diff_depth -= (right_idx - left_idx + 1) * cloudDistance[idx];
                diff_depth /= (right_idx - left_idx);
                // diff_depth = diff_depth * diff_depth;
                cloudCurvature[idx] = abs(diff_depth); 
                left_neighbor[idx] = left_idx;
                right_neighbor[idx] = right_idx;
            }
        }
    }

    #if 0
    // 保存每个点的弯曲度
    pcl::PointCloud<pcl::PointXYZI> cloud_curvature(cloud_scan);
    for(int idx = 0; idx < cloudSize ; idx ++)
    {
        cloud_curvature.points[idx].intensity = cloudCurvature[idx];
    }
    string base_name = num2str(id);
    if(IsPoseValid())
        pcl::transformPointCloud(cloud_curvature, cloud_curvature, GetPose());
    pcl::io::savePCDFileASCII(base_name + "_curvature.pcd", cloud_curvature);
    
    // 显示每个点的入射角，用于debug
    pcl::PointCloud<pcl::PointXYZI> cloud_intersect(cloud_scan);
    for(pcl::PointXYZI& p : cloud_intersect.points)
        p.intensity = 0;
    for(int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        for(int ind = scanStartInd[i]; ind <= scanEndInd[i]; ind++)
        {
            // 计算入射角，也就是入射激光和激光落点所在的局部平面的夹角
            // 详情见 livox loam论文里的公式(4)
            Eigen::Vector3f vec_a = PclPonit2EigenVecf(cloud_scan.points[ind]);
            Eigen::Vector3f vec_b = PclPonit2EigenVecf(cloud_scan.points[left_neighbor[ind]]) - 
                                    PclPonit2EigenVecf(cloud_scan.points[right_neighbor[ind]]);
            
            float view_angle = acos(abs(vec_a.dot(vec_b)) / ( cloudDistance[ind] * vec_b.norm()));
            view_angle *= (180.0 / M_PI);  
            cloud_intersect.points[ind].intensity = view_angle;
        }
    }

    if(IsPoseValid())
        pcl::transformPointCloud(cloud_curvature, cloud_curvature, GetPose());
    pcl::io::savePCDFileASCII(base_name + "_intersect.pcd", cloud_intersect);
    #endif

    // 对所有点按照曲率排序，这个排序是先把每个scan分成六份，然后在每一份内按照曲率排序
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 把一个scan分成连续的6段，每次只遍历其中一段  sp=start point  ep=end point
        // j = 0  => sp = start                     ep = start + 1/6 * length - 1
        // j = 1  => sp = start +  1/6 * length     ep = start + 2/6 * length - 1
        // j = 2  => sp = start +  2/6 * length     ep = start + 3/6 * length - 1
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            // 按弯曲度对这一段里的点从小到大排序，排序结果存在cloudSortInd里对应于sp-ep这一段里
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, 
                    [this](int a, int b)->bool{return (cloudCurvature[a]<cloudCurvature[b]);}
                    );
        }
    }

    // 如果要提取两次，那就用当前传递的参数作为宽松版参数，然后计算得到一个严格版参数
    // 为了保证提取的效果，最好还是需要传递的参数宽松一些，能提出更多的特征点
    if(method == DOUBLE_EXTRACTION)
    {
        // curvature 越小， intersect angle越大，选取点就越严格
        Eigen::MatrixXf strict_image;
        ExtractEdgeFeatures(strict_image, max_curvature / 10.0, intersect_angle_threshold * 2);
        // pcl::io::savePCDFileASCII("corner_strict.pcd", cornerLessSharp);
        // VisualizeRangeImage("image_strict.png", strict_image, 10, 0.5);
        memcpy(cloudState, cloudState_preserve, cloudSize * sizeof(int));
        
        Eigen::MatrixXf loose_image;
        ExtractEdgeFeatures(loose_image, max_curvature, intersect_angle_threshold);
        // pcl::io::savePCDFileASCII("corner_loose.pcd", cornerLessSharp);
        // VisualizeRangeImage("image_loose.png", loose_image, 10, 0.5);

        CombineEdgeFeatures(strict_image, loose_image);
        // 最后提取平面特征
        ExtractPlaneFeatures();
    }
    else if(method == LOAM)
    {
        Eigen::MatrixXf loose_image;
        ExtractEdgeFeatures(loose_image, max_curvature, intersect_angle_threshold);
        // 最后提取平面特征
        ExtractPlaneFeatures();
    }
    else if(method == ADAPTIVE)
    {
        Eigen::MatrixXf loose_image;
        ExtractEdgeFeatures2(loose_image, max_curvature, intersect_angle_threshold);
        EdgeToLine();
        // 最后提取平面特征
        ExtractPlaneFeatures2();
    }
    

    // 释放内存
    free(cloudCurvature);
    cloudCurvature = NULL;
    free(cloudDistance);
    cloudDistance = NULL;
    free(cloudState_preserve);
    cloudState_preserve = NULL;
    free(cloudState);
    cloudState = NULL;
    free(left_neighbor);
    left_neighbor = NULL;
    free(right_neighbor);
    right_neighbor = NULL;
}

void Velodyne::ExtractEdgeFeatures(Eigen::MatrixXf& picked_image, float max_curvature, float intersect_angle_threshold)
{
    int cloudSize = cloud_scan.points.size();
    const int neighbor_size = 5;

    cornerSharp.clear();
    cornerLessSharp.clear();
    vector<size_t> corner_picked;
    // 遍历所有scan
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 把一个scan分成连续的6段，每次只遍历其中一段  sp=start point  ep=end point
        // j = 0  => sp = start                     ep = start + 1/6 * length - 1
        // j = 1  => sp = start +  1/6 * length     ep = start + 2/6 * length - 1
        // j = 2  => sp = start +  2/6 * length     ep = start + 3/6 * length - 1
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
            int largestPickedNum = 0;   // 选取的弯曲度最大的点的数量
            // 从ep开始选，因为cloudSortInd是按弯曲度从小到大排列的，最后一个点弯曲度最大
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; 
                if(cloudState[ind] != POINT_NORMAL)
                    continue;
                // 弯曲度太大的也不要，很可能是测量错误的点
                if(cloudCurvature[ind] > max_curvature || cloudCurvature[ind] < 0.1)     
                    continue;
                // 计算入射角，也就是入射激光和激光落点所在的局部平面的夹角
                // 详情见 livox loam论文里的公式(4)
                Eigen::Vector3f vec_a = PclPonit2EigenVecf(cloud_scan.points[ind]);
                Eigen::Vector3f vec_b = PclPonit2EigenVecf(cloud_scan.points[ind + neighbor_size]) - 
                                        PclPonit2EigenVecf(cloud_scan.points[ind - neighbor_size]);
                
                float view_angle = acos(abs(vec_a.dot(vec_b)) / 
                                        ( range_image(point_idx_to_image[ind].first, point_idx_to_image[ind].second) * vec_b.norm()));
                view_angle *= (180.0 / M_PI);   
                if(view_angle < intersect_angle_threshold || view_angle > 180 - intersect_angle_threshold)
                    continue;

                // 弯曲度最大的两个点放入sharp和lessSharp里，剩下的最大的28个放入lessSharp里
                largestPickedNum++;
                if (largestPickedNum <= 3)
                {                        
                    cloudState[ind] = POINT_SHARP;
                    // 把每个点的intensity改为这个点在cloud_scan的索引
                    // 这个是在后面有用的
                    PointType p(cloud_scan.points[ind]);
                    p.intensity = ind;
                    cornerSharp.emplace_back(p);
                    cornerLessSharp.emplace_back(p);
                    corner_picked.push_back(ind);
                }
                else if (largestPickedNum <= 30)
                {                        
                    cloudState[ind] = POINT_LESS_SHARP; 
                    PointType p(cloud_scan.points[ind]);
                    p.intensity = ind;
                    cornerLessSharp.emplace_back(p);
                    corner_picked.push_back(ind);
                }
                else
                    break;
                
                // 遍历当前点之后的连续5个点，计算相邻的两个点之间的距离，大于0.22就认为产生了间断，不能被
                // 选择了，即对应点的 cloudState=0
                // 0.2236 * 0.2236 = 0.05
                for (int l = 1; l <= 5; l++)
                {
                    float diffX = cloud_scan.points[ind + l].x - cloud_scan.points[ind + l - 1].x;
                    float diffY = cloud_scan.points[ind + l].y - cloud_scan.points[ind + l - 1].y;
                    float diffZ = cloud_scan.points[ind + l].z - cloud_scan.points[ind + l - 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudState[ind + l] |= POINT_DISABLE;
                }
                // 和上面一样，只是遍历之前的5个点
                for (int l = -1; l >= -5; l--)
                {
                    float diffX = cloud_scan.points[ind + l].x - cloud_scan.points[ind + l + 1].x;
                    float diffY = cloud_scan.points[ind + l].y - cloud_scan.points[ind + l + 1].y;
                    float diffZ = cloud_scan.points[ind + l].z - cloud_scan.points[ind + l + 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudState[ind + l] |= POINT_DISABLE;
                }
                
            }
        }

    }

    // LOG(INFO) << "pick " << corner_picked.size() << " points with loose condition";
    picked_image.resize(N_SCANS, horizon_scans);
    picked_image.fill(0);
    for(const size_t& idx : corner_picked)
    {
        const size_t row = point_idx_to_image[idx].first;
        const size_t col = point_idx_to_image[idx].second;
        picked_image(row, col) = range_image(row, col);
    }
}

void Velodyne::ExtractEdgeFeatures2(Eigen::MatrixXf& picked_image, float max_curvature, float intersect_angle_threshold)
{
    int cloudSize = cloud_scan.points.size();
    const int neighbor_size = 5;

    cornerSharp.clear();
    cornerLessSharp.clear();
    vector<size_t> corner_picked;
    // 遍历所有scan
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 把一个scan分成连续的6段，每次只遍历其中一段  sp=start point  ep=end point
        // j = 0  => sp = start                     ep = start + 1/6 * length - 1
        // j = 1  => sp = start +  1/6 * length     ep = start + 2/6 * length - 1
        // j = 2  => sp = start +  2/6 * length     ep = start + 3/6 * length - 1
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
            int largestPickedNum = 0;   // 选取的弯曲度最大的点的数量
            // 从ep开始选，因为cloudSortInd是按弯曲度从小到大排列的，最后一个点弯曲度最大
            for (int k = ep; k >= sp; k--)
            {
                int ind = cloudSortInd[k]; 
                if(cloudState[ind] != POINT_NORMAL)
                    continue;
                // 弯曲度太大的也不要，很可能是测量错误的点
                if(cloudCurvature[ind] > max_curvature || cloudCurvature[ind] < 0.1)    
                    continue;
                // 计算入射角，也就是入射激光和激光落点所在的局部平面的夹角
                // 详情见 livox loam论文里的公式(4)
                Eigen::Vector3f vec_a = PclPonit2EigenVecf(cloud_scan.points[ind]);
                Eigen::Vector3f vec_b = PclPonit2EigenVecf(cloud_scan.points[left_neighbor[ind]]) - 
                                        PclPonit2EigenVecf(cloud_scan.points[right_neighbor[ind]]);
                
                float view_angle = acos(abs(vec_a.dot(vec_b)) / ( range_image(point_idx_to_image[ind].first, point_idx_to_image[ind].second) * vec_b.norm()));
                view_angle *= (180.0 / M_PI);   
                if(view_angle < intersect_angle_threshold || view_angle > 180 - intersect_angle_threshold)
                    continue;
                
                // 找到当前点左右的近邻点，判断这些近邻点能否形成一条直线，如果可以形成，就说明当前点不是边缘特征点
                // 出现这种情况的原因是LiDAR附近可能有一些大平面，距离LiDAR比较近，那么平面上的点就会产生很大的曲率
                // 但这种情况实际上是不应该提取出边缘特征的，因此要进行判断，如果近邻的点都在同一条直线上，那么就说明这个点
                // 是在一个平面上
                // eigen_vector<Eigen::Vector3d> neighbor_points;
                // for(int left_idx = left_neighbor[ind]; left_idx <= right_neighbor[ind]; left_idx++)
                // {
                //     neighbor_points.push_back(PclPonit2EigenVecd(cloud_scan.points[left_idx]));
                // }
                // Vector6d line_coeff = FormLine(neighbor_points, 5.0, 0.05);
                // if(!line_coeff.isZero())
                //     continue;

                // 弯曲度最大的两个点放入sharp和lessSharp里，剩下的最大的28个放入lessSharp里
                largestPickedNum++;
                if (largestPickedNum <= 3)
                {                        
                    cloudState[ind] = POINT_SHARP;
                    // 把每个点的intensity改为这个点在cloud_scan的索引
                    // 这个是在后面有用的
                    PointType p(cloud_scan.points[ind]);
                    p.intensity = ind;
                    cornerSharp.emplace_back(p);
                    cornerLessSharp.emplace_back(p);
                    corner_picked.push_back(ind);
                }
                else if (largestPickedNum <= 30)
                {                        
                    cloudState[ind] = POINT_LESS_SHARP; 
                    PointType p(cloud_scan.points[ind]);
                    p.intensity = ind;
                    cornerLessSharp.emplace_back(p);
                    corner_picked.push_back(ind);
                }
                else
                    break;
                
                // 每选中一个点，就会对周围的点进行压制，压制表现为三个条件
                // 1. 距离当前点五个点之内的点，只有两个相邻点之间距离大于22.36cm，才会终止压制
                // 2. 距离当前点超过5个点的，如果和当前点距离大于6cm，也会终止压制
                // 3. 当前scan已经到达尽头了，那么自然终止压制
                // 0.2236 * 0.2236 = 0.05
                // 0.1 * 0.1 = 0.01
                for(int l = 1; ind + l <= scanEndInd[i]; l++ )
                {
                    if(l <= 5 && PointDistanceSquare(cloud_scan.points[ind + l], cloud_scan.points[ind + l - 1]) > 0.05)
                        break;
                    else if(l > 5 && PointDistanceSquare(cloud_scan.points[ind + l], cloud_scan.points[ind]) > 0.0036)
                        break;
                    cloudState[ind + l] |= POINT_DISABLE;
                }
                for(int l = 1; ind - l >= scanStartInd[i]; l++ )
                {
                    if(l <= 5 && PointDistanceSquare(cloud_scan.points[ind - l], cloud_scan.points[ind - l + 1]) > 0.05)
                        break;
                    else if(l > 5 && PointDistanceSquare(cloud_scan.points[ind - l], cloud_scan.points[ind]) > 0.0036)
                        break;
                    cloudState[ind - l] |= POINT_DISABLE;
                    
                }
            }
        }

    }

    // LOG(INFO) << "pick " << corner_picked.size() << " points with loose condition";
    picked_image.resize(N_SCANS, horizon_scans);
    picked_image.fill(0);
    for(const size_t& idx : corner_picked)
    {
        const size_t row = point_idx_to_image[idx].first;
        const size_t col = point_idx_to_image[idx].second;
        picked_image(row, col) = range_image(row, col);
    }
}


void Velodyne::ExtractPlaneFeatures()
{
    int cloudSize = cloud_scan.points.size();
    const int neighbor_size = 5;
    surfFlat.clear();
    surfLessFlat.clear();
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            int smallestPickedNum = 0;      // 选择的弯曲度最小的点的数量
            // 这里是从sp开始选择的
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 地面点和普通点都可以作为特别平坦点的一员
                if (cloudState[ind] != POINT_NORMAL && cloudState[ind] != POINT_GROUND)
                    continue;
                if(cloudCurvature[ind] > 0.1)
                    continue;
                // 给surfFlat里的点进行一下分类，分成地面点和非地面点。在之后的匹配的时候，地面点只能和
                // 地面点匹配，非地面点只能和非地面点匹配。使用点的intensity保存当前点是否为地面点
                PointType pt(cloud_scan.points[ind]);
                pt.intensity = cloudState[ind];
                surfFlat.push_back(pt);
                // 能进入到这个位置的点只有普通点和地面点，为了将地面点进行特殊处理，
                // surfPointsLessFlatScan 中只会保留普通点，地面点将在后面单独处理
                if(cloudState[ind] == POINT_NORMAL)
                    surfPointsLessFlatScan->push_back(pt);
                cloudState[ind] |= POINT_FLAT; 
                smallestPickedNum++;
                for (int l = 1; l <= 5; l++)
                { 
                    float diffX = cloud_scan.points[ind + l].x - cloud_scan.points[ind + l - 1].x;
                    float diffY = cloud_scan.points[ind + l].y - cloud_scan.points[ind + l - 1].y;
                    float diffZ = cloud_scan.points[ind + l].z - cloud_scan.points[ind + l - 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudState[ind + l] |= POINT_DISABLE;
                }
                for (int l = -1; l >= -5; l--)
                {
                    float diffX = cloud_scan.points[ind + l].x - cloud_scan.points[ind + l + 1].x;
                    float diffY = cloud_scan.points[ind + l].y - cloud_scan.points[ind + l + 1].y;
                    float diffZ = cloud_scan.points[ind + l].z - cloud_scan.points[ind + l + 1].z;
                    if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                    {
                        break;
                    }

                    cloudState[ind + l] |= POINT_DISABLE;
                }
                
            }
            
            for (int k = sp; k <= ep; k++)
            {
                if((cloudState[k] & POINT_NORMAL) > 0 )
                {
                    surfPointsLessFlatScan->push_back(cloud_scan.points[k]);
                }
            }
        }
        pcl::PointCloud<PointType> surfLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfLessFlatScanDS);

        surfLessFlat += surfLessFlatScanDS;
    }
    for(PointType& p : surfLessFlat.points)
        p.intensity = POINT_NORMAL;
    // 在这里单独处理地面点
    pcl::PointCloud<PointType> ground_cloud, ground_cloud_ds;
    for(size_t i = 0; i < cloudSize; i++)
        if((cloudState[i] & POINT_GROUND) > 0)
            ground_cloud.push_back(cloud_scan.points[i]);
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(ground_cloud.makeShared());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(ground_cloud_ds);
    for(PointType& p : ground_cloud_ds.points)
        p.intensity = POINT_GROUND;
    surfLessFlat += ground_cloud_ds;
}

void Velodyne::ExtractPlaneFeatures2()
{
    int cloudSize = cloud_scan.points.size();
    const int neighbor_size = 5;
    surfFlat.clear();
    surfLessFlat.clear();
    for (int i = 0; i < N_SCANS; i++)
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            int smallestPickedNum = 0;      // 选择的弯曲度最小的点的数量
            // 这里是从sp开始选择的
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 地面点和普通点都可以作为特别平坦点的一员
                if (cloudState[ind] != POINT_NORMAL && cloudState[ind] != POINT_GROUND)
                    continue;
                if(cloudCurvature[ind] > 0.02)
                    continue;
                // 给surfFlat里的点进行一下分类，分成地面点和非地面点。在之后的匹配的时候，地面点只能和
                // 地面点匹配，非地面点只能和非地面点匹配。使用点的intensity保存当前点是否为地面点
                PointType pt(cloud_scan.points[ind]);
                pt.intensity = cloudState[ind];
                surfFlat.push_back(pt);
                // 能进入到这个位置的点只有普通点和地面点，为了将地面点进行特殊处理，
                // surfPointsLessFlatScan 中只会保留普通点，地面点将在后面单独处理
                if(cloudState[ind] == POINT_NORMAL)
                    surfPointsLessFlatScan->push_back(pt);
                cloudState[ind] |= POINT_FLAT; 
                smallestPickedNum++;
                for(int l = 1; ind + l <= scanEndInd[i]; l++ )
                {
                    if(l <= 5 && PointDistanceSquare(cloud_scan.points[ind + l], cloud_scan.points[ind + l - 1]) > 0.05)
                        break;
                    else if(l > 5 && PointDistanceSquare(cloud_scan.points[ind + l], cloud_scan.points[ind]) > 0.0036)
                        break;
                    cloudState[ind + l] |= POINT_DISABLE;
                }
                for(int l = 1; ind - l >= scanStartInd[i]; l++ )
                {
                    if(l <= 5 && PointDistanceSquare(cloud_scan.points[ind - l], cloud_scan.points[ind - l + 1]) > 0.05)
                        break;
                    else if(l > 5 && PointDistanceSquare(cloud_scan.points[ind - l], cloud_scan.points[ind]) > 0.0036)
                        break;
                    cloudState[ind - l] |= POINT_DISABLE;
                }
                if (smallestPickedNum >= 4)
                    break;
            }
            
            for (int k = sp; k <= ep; k++)
            {
                // 必须满足三个条件才能认为是 “不那么平坦”的点
                // 1. 当前点得是个普通点 2. 当前点附近没有其他特征点 3. 当前点的曲率比较小
                if((cloudState[k] & POINT_NORMAL) > 0  &&
                    (cloudState[k] & POINT_DISABLE) == 0 && 
                    cloudCurvature[k] < 0.3)
                {
                    surfPointsLessFlatScan->push_back(cloud_scan.points[k]);
                }
            }
        }
        pcl::PointCloud<PointType> surfLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfLessFlatScanDS);

        surfLessFlat += surfLessFlatScanDS;
    }
    for(PointType& p : surfLessFlat.points)
        p.intensity = POINT_NORMAL;
    // 在这里单独处理地面点
    pcl::PointCloud<PointType> ground_cloud, ground_cloud_ds;
    for(size_t i = 0; i < cloudSize; i++)
        if((cloudState[i] & POINT_GROUND) > 0)
            ground_cloud.push_back(cloud_scan.points[i]);
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(ground_cloud.makeShared());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(ground_cloud_ds);
    for(PointType& p : ground_cloud_ds.points)
        p.intensity = POINT_GROUND;
    surfLessFlat += ground_cloud_ds;
}

void Velodyne::CombineEdgeFeatures(const Eigen::MatrixXf& strict_image, const Eigen::MatrixXf& loose_image)
{
    cornerLessSharp.clear();
    const int h_size = 3;   // 水平方向的邻域半径
    const int v_size = 2;   // 竖直方向的邻域半径
    stack<pair<size_t, size_t>> s;
    vector<pair<size_t, size_t>> segments;
    size_t seg_count = 0;

    // 保存哪些点被访问过，0代表没被访问，1代表访问过
    Eigen::MatrixXi visited(N_SCANS, horizon_scans);
    visited.fill(0);
    // 保存所有被保留下来的边缘点，只是用来可视化的，删去也无妨
    Eigen::MatrixXi corner(N_SCANS, horizon_scans);
    corner.fill(0);
    // BFS算法遍历两张图，得到较为稳定的边缘点，具体方法受Canny算子提取边缘启发
    // 首先默认strict里的点都是边缘点，loose里大部分是边缘点，少部分不是，最终的目标就是提取出最可靠的边缘
    // 1. 从strict里开始，找到第一个点然后遍历它的邻域，这里的邻域既包括strict也包括loose，直到邻域所有的邻域都遍历了，形成了一个
    // 小区域（segment），就可以认为这个segment里的点都是属于同一个边缘的
    // 2. 如果segment里有足够多的点，就认为它确实是一个边缘，保留下来，否则就舍弃
    // 3. 重复步骤1和2直到strict里所有的点都被遍历了
    for(int i = 0; i < strict_image.rows(); i++)
    {
        for(int j = 0; j < strict_image.cols(); j++)
        {
            if(visited(i, j) > 0 || strict_image(i,j) == 0)
                continue;
            s.push(pair<size_t,size_t>(i,j));
            while(!s.empty())
            {
                const size_t row = s.top().first;
                const size_t col = s.top().second;
                if(visited(row, col) > 0)
                {
                    s.pop();
                    continue;
                }
                visited(row, col) = 1;
                segments.emplace_back(s.top());
                s.pop();
                for(int h = -h_size; h <= h_size; h++)
                {
                    for(int v = -v_size; v <= v_size; v++)
                    {
                        const int curr_row = row + v;
                        const int curr_col = col + h;
                        if(curr_row < 0 || curr_row > N_SCANS - 1 || curr_col < 0 || curr_col > horizon_scans - 1)
                            continue;
                        if(strict_image(curr_row, curr_col) > 0 || loose_image(curr_row, curr_col) > 0)
                            s.emplace(pair<size_t, size_t>(size_t(curr_row), size_t(curr_col)));
                    }
                }
            }
            // LOG(INFO) << "segment size: " << segments.size();
            // 如果某个区域里的点足够多，就保留当前区域的点作为最终提取出的点, 这个segment里的点会保存在两个点云里
            // 一个是统一的点云 corner less sharp，另一个则是每个segment独立的点云，edge_segmented
            if(segments.size() > 4)
            {
                pcl::PointCloud<pcl::PointXYZI> seg;
                for(const pair<size_t,size_t>& p : segments)
                {
                    const size_t& point_idx = image_to_point_idx[p.first][p.second];
                    pcl::PointXYZI point = cloud_scan.points[point_idx];
                    cornerLessSharp.push_back(point);
                    point.intensity = seg_count;
                    seg.push_back(point);
                    corner(p.first, p.second) = 1;
                }
                edge_segmented.push_back(seg);
                seg_count++;
            }
            segments.clear();
        }
    }
    // VisualizeRangeImage("image_visited.png",visited.cast<float>(), 1, 0.5);
    // VisualizeRangeImage("image_corner.png",corner.cast<float>(), 1, 0.5);
}

void Velodyne::EdgeToLine()
{
    cornerBeforeFilter = cornerLessSharp;
    eigen_vector<Vector6d> line_coeffs;
    ExtractLineFeatures(cornerLessSharp, point_idx_to_image, edge_segmented, segment_coeffs);
    assert(edge_segmented.size() == segment_coeffs.size());
    for(size_t i = 0; i < edge_segmented.size(); i++)
    {
        int start, end;
        double distance;
        FurthestPoints(edge_segmented[i], start, end, distance);
        end_points.push_back(ProjectPoint2Line3D(PclPonit2EigenVecd(edge_segmented[i].points[start]), segment_coeffs[i].data()));
        end_points.push_back(ProjectPoint2Line3D(PclPonit2EigenVecd(edge_segmented[i].points[end]), segment_coeffs[i].data()));
    }
    cornerLessSharp.clear();
    // 在这里要进行两个任务
    // 1. 把不同的segment都放到cornerLessSharp里，而且不能有重复的点。
    // 也就是说点p在多个segment里同时存在，但是在cornerLessSharp里只能有一个
    // 2. 记录CornerLessSharp里的点都属于哪个segment，如果某个点属于多个segment，
    // 也要记录在point_to_segment里
    map<int, int> id_to_idx;       // 保存了当前cornerLessSharp里所有点的索引以及对应的id，key=id，value=索引
    map<int, int>::const_iterator it;
    for(size_t seg_id = 0; seg_id < edge_segmented.size(); seg_id++)
    {
        const pcl::PointCloud<PointType>& cloud = edge_segmented[seg_id];
        for(const PointType& p : cloud)
        {
            it = id_to_idx.find(int(p.intensity));
            if(it != id_to_idx.end())
                point_to_segment[it->second].insert(seg_id);
            else 
            {
                id_to_idx[int(p.intensity)] = cornerLessSharp.size();
                cornerLessSharp.push_back(p);
                set<int> tmp = {(int)seg_id};
                point_to_segment.push_back(tmp);
            }
        }
    }
    
    assert(point_to_segment.size() == cornerLessSharp.size());
    assert(point_to_segment.size() == id_to_idx.size());

    // 对corner sharp进行过滤，方法是把原本的sharp和现在的less sharp做交集，得到的就是新的sharp
    pcl::PointCloud<PointType> tmp = cornerSharp;
    cornerSharp.clear();
    set<int> idx;
    for(const PointType& p : cornerLessSharp.points)
        idx.insert(int(p.intensity));
    for(const PointType& p : tmp.points)
    {
        if(idx.count(p.intensity) == 0)
            continue;
        cornerSharp.emplace_back(p);
    }
}

Eigen::Vector4f Velodyne::SegmentGround(bool compute_coeff)
{
    GroundSegmentationParams params;
	params.visualize = false;
	params.n_threads = 1;
	params.n_bins = 120 * 5;
	params.n_segments = 360;
	params.max_dist_to_line = 0.1;
	params.max_slope = 0.3;
	params.long_threshold = 1.0*6;
	params.max_long_height = 0.4;
	params.max_start_height = 0.3;
	params.sensor_height = 1.6;
	params.line_search_angle = 0.1;
	params.r_min_square = 0.5 * 0.5;
	params.r_max_square = 50 * 50;
	params.max_error_square = 0.05 * 0.05;

    GroundSegmentation segmenter(params);

    // 改变LiDAR的坐标系，原本是X向右，Y向前，Z向上
    // 变成 X向右，Y向下，Z向前，这种顺序适应相机的坐标系
    Eigen::Matrix4f T_cam_lidar;
    T_cam_lidar << 1, 0, 0, 0 ,
                   0, 0, -1, 0,
                   0, 1, 0, 0,
                   0, 0, 0, 1;
    pcl::PointCloud<PointType> cloud_tmp;
    pcl::transformPointCloud(cloud_scan, cloud_tmp, T_cam_lidar.inverse());


    std::vector<int> ground_inliers;
    segmenter.segment(cloud_tmp, ground_inliers);
    Eigen::Vector4f ground_coeff(0,0,0,0);
    if(ground_inliers.size() < 100)
        return ground_coeff;
    #if 0
    for(const int& idx : ground_inliers)
        cloudState[idx] = POINT_GROUND;
    if(compute_coeff)
    {
        pcl::PointCloud<PointType> cloud_ground_init;
        for(const int& idx : ground_inliers)
            cloud_ground_init.push_back(cloud_scan[idx]);
        FormPlane(cloud_ground_init.points, ground_coeff, -1.f);
    }
    #else 
    // 用RANSAC得到一个更好的结果
    pcl::PointCloud<PointType> cloud_ground_init;
    for(const int& idx : ground_inliers)
        cloud_ground_init.push_back(cloud_scan[idx]);
    pcl::SampleConsensusModelPlane<PointType>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointType>(cloud_ground_init.makeShared()));
    pcl::RandomSampleConsensus<PointType> ransac(model_p);
    ransac.setDistanceThreshold(0.1);
    ransac.computeModel();
    std::vector<int> rac_inliers;
    ransac.getInliers(rac_inliers);
    for(const int& idx : rac_inliers)
        cloudState[ground_inliers[idx]] = POINT_GROUND;
    Eigen::VectorXf plane_coeff;
    ransac.getModelCoefficients(plane_coeff);
    ground_coeff = plane_coeff;
    /*
    pcl::PointCloud<PointType> cloud_ground;
    for(size_t i = 0; i < cloud_scan.size(); i++)
    {
        if(cloudState[i] == POINT_GROUND)
            cloud_ground.push_back(cloud_scan[i]);
    }
    pcl::io::savePCDFileBinary(num2str(id) + "_ground.pcd", cloud_ground);
    pcl::io::savePCDFileBinary(num2str(id) + "_gournd_init.pcd", cloud_ground_init);
    */
    #endif
    return ground_coeff;

}

bool Velodyne::ExtractGroundPointCloud(pcl::PointCloud<PointType>& ground_cloud, pcl::PointCloud<PointType>& other_cloud, Eigen::Vector4f& ground_coeff)
{
    if(cloud_scan.empty())
    {
        LOG(ERROR) << "cloud_scan is empty";
        return false;
    }
    ground_cloud.clear();
    cloudState = (int*)malloc(cloud_scan.size() * sizeof(int));
    fill(cloudState, cloudState + cloud_scan.size(), POINT_NORMAL);
    ground_coeff = SegmentGround(true);
    if(ground_coeff.isZero())
    {
        LOG(ERROR) << "Extract ground failed";
        return false;
    }
    for(size_t i = 0; i < cloud_scan.size(); i++)
    {
        if(cloudState[i] == POINT_GROUND)
            ground_cloud.push_back(cloud_scan[i]);
        else 
            other_cloud.push_back(cloud_scan[i]);
    }
    free(cloudState);
    return true;
}

bool Velodyne::ExtractPlanes()
{
    // PlaneSegmentation(cloud_scan.makeShared(), point_idx_to_image, image_to_point_idx, 20, 10000);
    PlaneSegmentation2(cloud_scan.makeShared(), range_image, point_idx_to_image, image_to_point_idx, 100, 10000);
    return true;
}

// 对点云进行分割，分割的算法来自于 LEGO-LOAM
int Velodyne::Segmentation()
{
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> label_image(N_SCANS, horizon_scans);
    label_image.fill(0);

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;
    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;
    queueIndX = new uint16_t[N_SCANS*horizon_scans];
    queueIndY = new uint16_t[N_SCANS*horizon_scans];
    allPushedIndX = new uint16_t[N_SCANS*horizon_scans];
    allPushedIndY = new uint16_t[N_SCANS*horizon_scans];
    // neighbor iterator for segmentaiton process
    vector<pair<int8_t, int8_t> > neighborIterator = {
        pair<int8_t, int8_t>(-1, 0),
        pair<int8_t, int8_t>(0, 1),
        pair<int8_t, int8_t>(0, -1),
        pair<int8_t, int8_t>(1, 0)
    } ;
    float segmentAlphaX = 0.2 / 180.0 * M_PI;
    float segmentAlphaY = 2.0 / 180.0 * M_PI;
    // float segmentTheta = 60.0 / 180.0 * M_PI; // decrese this value may improve accuracy     竖直放置
    float segmentTheta = 20.0 / 180.0 * M_PI;       // 倾斜放置
    int label_count = 1;
    for (size_t row = 0; row < N_SCANS; ++row)
        for (size_t col = 0; col < horizon_scans; ++col)
            if (label_image(row,col) == 0)
            {
                // use std::queue std::vector std::deque will slow the program down greatly
                float d1, d2, alpha, angle;
                int fromIndX, fromIndY, thisIndX, thisIndY; 
                bool lineCountFlag[N_SCANS] = {false};

                queueIndX[0] = row;
                queueIndY[0] = col;
                int queueSize = 1;
                int queueStartInd = 0;
                int queueEndInd = 1;

                allPushedIndX[0] = row;
                allPushedIndY[0] = col;
                int allPushedIndSize = 1;
                
                while(queueSize > 0){
                    // Pop point
                    fromIndX = queueIndX[queueStartInd];
                    fromIndY = queueIndY[queueStartInd];
                    --queueSize;
                    ++queueStartInd;
                    // Mark popped point
                    label_image(fromIndX, fromIndY) = label_count;
                    // Loop through all the neighboring grids of popped grid
                    for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                        // new index
                        thisIndX = fromIndX + (*iter).first;
                        thisIndY = fromIndY + (*iter).second;
                        // index should be within the boundary
                        if (thisIndX < 0 || thisIndX >= N_SCANS)
                            continue;
                        // at range image margin (left or right side)
                        if (thisIndY < 0)
                            thisIndY = horizon_scans - 1;
                        if (thisIndY >= horizon_scans)
                            thisIndY = 0;
                        // prevent infinite loop (caused by put already examined point back)
                        if (label_image(thisIndX, thisIndY) != 0)
                            continue;

                        d1 = std::max(range_image(fromIndX, fromIndY), range_image(thisIndX, thisIndY));
                                    
                        d2 = std::min(range_image(fromIndX, fromIndY), range_image(thisIndX, thisIndY));
                                    

                        if ((*iter).first == 0)
                            alpha = segmentAlphaX;
                        else
                            alpha = segmentAlphaY;

                        angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                        if (angle > segmentTheta){

                            queueIndX[queueEndInd] = thisIndX;
                            queueIndY[queueEndInd] = thisIndY;
                            ++queueSize;
                            ++queueEndInd;

                            label_image(thisIndX, thisIndY) = label_count;
                            lineCountFlag[thisIndX] = true;

                            allPushedIndX[allPushedIndSize] = thisIndX;
                            allPushedIndY[allPushedIndSize] = thisIndY;
                            ++allPushedIndSize;
                        }
                    }
                }

                // check if this segment is valid
                bool feasibleSegment = false;
                if (allPushedIndSize >= 30)
                    feasibleSegment = true;
                else if (allPushedIndSize >= 5){
                    int lineCount = 0;
                    for (size_t i = 0; i < N_SCANS; ++i)
                        if (lineCountFlag[i] == true)
                            ++lineCount;
                    if (lineCount >= 3)
                        feasibleSegment = true;            
                }
                // segment is valid, mark these points
                if (feasibleSegment == true){
                    ++label_count;
                }else{ // segment is invalid, mark these points
                    for (size_t i = 0; i < allPushedIndSize; ++i){
                        label_image(allPushedIndX[i], allPushedIndY[i]) = INT16_MAX;
                    }
                }
            }
    
    // 根据分割结果重新组织点云，除去不可靠点
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // map<size_t, pair<size_t, size_t>> point_idx_to_image_new;
    // map<pair<size_t, size_t>, size_t> image_to_point_idx_new;
    std::vector<std::pair<size_t, size_t> > point_idx_to_image_new;
    std::vector<std::vector<int> > image_to_point_idx_new;
    image_to_point_idx_new.resize(N_SCANS, vector<int>(horizon_scans, -1));

    size_t count = 0;
    
    for(size_t idx = 0; idx < cloud_scan.size(); idx++)
    {
        if(label_image(point_idx_to_image[idx].first, point_idx_to_image[idx].second) == INT16_MAX)
            continue;
        else 
        {
            point_idx_to_image_new.push_back(point_idx_to_image[idx]);
            image_to_point_idx_new[point_idx_to_image[idx].first][point_idx_to_image[idx].second] = count;
            count ++;
            laserCloudScans[cloud_scan.points[idx].intensity].push_back(cloud_scan.points[idx]);
        }
        
    }
    cloud_scan.clear();
    point_idx_to_image.swap(point_idx_to_image_new);
    point_idx_to_image_new.clear();
    image_to_point_idx_new.swap(image_to_point_idx);
    image_to_point_idx_new.clear();
    // 每个scan的前5个点和后6个点都不算
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = cloud_scan.size() + 5;
        cloud_scan += laserCloudScans[i];
        scanEndInd[i] = cloud_scan.size() - 6;
    }

    return 1;
}

void Velodyne::MarkOccludedPoints()
{
    int cloudSize = cloud_scan.points.size();

    for (int i = 5; i < cloudSize - 6; ++i){

        float depth1 = range_image(point_idx_to_image[i].first, point_idx_to_image[i].second);
        float depth2 = range_image(point_idx_to_image[i + 1].first, point_idx_to_image[i + 1].second);
        int columnDiff = std::abs(int( point_idx_to_image[i].second -  point_idx_to_image[i+1].second));

        if (columnDiff < 10){

            if (depth1 - depth2 > 0.3){
                cloudState[i - 5] = POINT_OCCLUDED;
                cloudState[i - 4] = POINT_OCCLUDED;
                cloudState[i - 3] = POINT_OCCLUDED;
                cloudState[i - 2] = POINT_OCCLUDED;
                cloudState[i - 1] = POINT_OCCLUDED;
                cloudState[i] = POINT_OCCLUDED;
            }else if (depth2 - depth1 > 0.3){
                cloudState[i + 1] = POINT_OCCLUDED;
                cloudState[i + 2] = POINT_OCCLUDED;
                cloudState[i + 3] = POINT_OCCLUDED;
                cloudState[i + 4] = POINT_OCCLUDED;
                cloudState[i + 5] = POINT_OCCLUDED;
                cloudState[i + 6] = POINT_OCCLUDED;
            }
        }

        float depth3 = range_image(point_idx_to_image[i - 1].first, point_idx_to_image[i - 1].second);
        float diff1 = std::abs(float(depth3 - depth1));
        float diff2 = std::abs(float(depth2 - depth1));

        if (diff1 > 0.02 * depth1 && diff2 > 0.02 * depth1)
            cloudState[i] = POINT_OCCLUDED;
    }
}

bool Velodyne::UndistortCloud(const Eigen::Matrix4d& T_we)
{
    Eigen::Matrix3d R_we = T_we.block<3,3>(0,0);
    Eigen::Vector3d t_we = T_we.block<3,1>(0,3);
    return UndistortCloud(R_we, t_we);
}

bool Velodyne::UndistortCloud(const Eigen::Matrix3d& R_we, const Eigen::Vector3d& t_we)
{
    if(!IsPoseValid())
        return false;
    // 计算从终止（end）到起始（start）的变换
    const Eigen::Matrix3d R_se = R_wl.transpose() * R_we;
    const Eigen::Vector3d t_se = R_wl.transpose() * (t_we - t_wl);
    const Eigen::Quaterniond q_se(R_se);
    if(cloud.empty())
        LoadLidar(name);
    if(!cloud.empty())
    {
        for(size_t i = 0; i < cloud.points.size(); i++)
        {
            double ratio = 1.f * i / cloud.points.size();
            Eigen::Quaterniond q_sc = Eigen::Quaterniond::Identity().slerp(ratio, q_se);
            Eigen::Vector3d t_sc = ratio * t_se;
            Eigen::Vector3d point = PclPonit2EigenVecd(cloud.points[i]);
            point = q_sc * point + t_sc;
            EigenVec2PclPoint(point, cloud.points[i]);
        }
        cloud_scan.clear();
        cornerLessSharp.clear();
        cornerSharp.clear();
        surfFlat.clear();
        surfLessFlat.clear();
        return true;
    }
    else 
    {
        return false;
    }
}

void Velodyne::Reset()
{
    cloud_scan.swap(*(new pcl::PointCloud<PointType>()));
    cornerLessSharp.swap(*(new pcl::PointCloud<PointType>()));
    cornerSharp.swap(*(new pcl::PointCloud<PointType>()));
    surfLessFlat.swap(*(new pcl::PointCloud<PointType>()));
    surfFlat.swap(*(new pcl::PointCloud<PointType>()));
    edge_segmented.swap(*(new vector<pcl::PointCloud<PointType>>()));
    point_to_segment.swap(*(new vector<set<int>>()));
    segment_coeffs.swap(*(new eigen_vector<Vector6d>()));
    end_points.swap(*(new eigen_vector<Eigen::Vector3d>()));
    cornerBeforeFilter.swap(*(new pcl::PointCloud<PointType>()));
    scanStartInd.resize(N_SCANS, -1);
    scanEndInd.resize(N_SCANS, -1);
    range_image.resize(0,0);
    point_idx_to_image.swap(*(new vector<pair<size_t, size_t>>()));
    image_to_point_idx.swap(*(new vector<vector<int>>()));
    if(cloudCurvature)
    {
        free(cloudCurvature);
        cloudCurvature = NULL;
    }
    if(cloudSortInd)
    {
        free(cloudSortInd);
        cloudSortInd = NULL;
    }
    if(cloudState)
    {
        free(cloudState);
        cloudState = NULL;
    }
    if(left_neighbor)
    {
        free(left_neighbor);
        left_neighbor = NULL;
    }
    if(right_neighbor)
    {
        free(right_neighbor);
        right_neighbor = NULL;
    }
}

const bool Velodyne::SaveFeatures(string path) const
{
    string base_name = num2str(id);
    if(!cornerLessSharp.empty())
        pcl::io::savePCDFile(path + base_name + "_corner_less_sharp.pcd", cornerLessSharp);
    if(!cornerSharp.empty())
        pcl::io::savePCDFile(path + base_name + "_corner_sharp.pcd", cornerSharp);
    if(!surfLessFlat.empty())
        pcl::io::savePCDFile(path + base_name + "_surf_less_flat.pcd", surfLessFlat);
    if(!surfFlat.empty())
        pcl::io::savePCDFile(path + base_name + "_surf_flat.pcd", surfFlat);
    if(!cloud_scan.empty())
        pcl::io::savePCDFile(path + base_name + "_cloud_scan.pcd", cloud_scan);
    if(!cloud.empty())
        pcl::io::savePCDFile(path + base_name + "_cloud.pcd", cloud);
    if(!edge_segmented.empty())
    {
        pcl::PointCloud<pcl::PointXYZI> segment;
        for(const pcl::PointCloud<pcl::PointXYZI>& s : edge_segmented)
            segment += s;
        pcl::io::savePCDFileASCII(path + base_name + "_edge_seg.pcd", segment);
        for(int i = 0; i < edge_segmented.size(); i++)
            pcl::io::savePCDFileASCII(path + base_name + "_segment_" + num2str(i) + ".pcd", edge_segmented[i]);
    }
    if(!cornerBeforeFilter.empty())
        pcl::io::savePCDFileASCII(path + base_name + "_before_filter.pcd", cornerBeforeFilter);
    return true;
}

void Velodyne::VisualizeRangeImage(std::string file_name, const Eigen::MatrixXf& _range_image, 
                                const float max_range, const float min_range)
{
    cv::Mat img_depth = cv::Mat::zeros(_range_image.rows(), _range_image.cols(), CV_8UC3);
    const float range = max_range - min_range;
    for(int i = 0; i < img_depth.rows; i++)
        for(int j = 0; j < img_depth.cols; j++)
        {
            float real_depth = _range_image(i,j);
            if(real_depth == 0)
            {
                img_depth.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                continue;
            }
            if(real_depth > max_range)
                real_depth = max_range;
            if(real_depth < min_range)
                real_depth = min_range;
            uchar relative_depth = static_cast<uchar>((real_depth - min_range) / range * 255.0);
            img_depth.at<cv::Vec3b>(i,j) = Gray2Color(relative_depth);
        }
    cv::imwrite(file_name, img_depth);
}

void Velodyne::Transform2LidarWorld()
{
    if(world)
    {
        cout << "already in world coordinate" << endl;
        return;
    }
    if(!IsPoseValid())
    {
        LOG(ERROR) << "lidar pose is invalid, unable to transform to world coordinate";
        return;
    }
    Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
    T_wl.block<3,3>(0,0) = R_wl;
    T_wl.block<3,1>(0,3) = t_wl;

    // 从雷达坐标系变换到雷达的世界坐标系
    if(!cloud_scan.empty())
        pcl::transformPointCloud(cloud_scan, cloud_scan, T_wl);
    if(!surfFlat.empty())
        pcl::transformPointCloud(surfFlat, surfFlat, T_wl);
    if(!surfLessFlat.empty())
        pcl::transformPointCloud(surfLessFlat, surfLessFlat, T_wl);
    if(!cornerSharp.empty())
        pcl::transformPointCloud(cornerSharp, cornerSharp, T_wl);
    if(!cornerLessSharp.empty())
        pcl::transformPointCloud(cornerLessSharp, cornerLessSharp, T_wl);
    if(!cloud.empty())
        pcl::transformPointCloud(cloud, cloud, T_wl);
    if(!edge_segmented.empty())
        for(pcl::PointCloud<PointType>& s : edge_segmented)
            pcl::transformPointCloud(s, s, T_wl);
    if(!cornerBeforeFilter.empty())
        pcl::transformPointCloud(cornerBeforeFilter, cornerBeforeFilter, T_wl);
    world = true;
}

void Velodyne::Transform2Local()
{
    if(!world)
    {
        LOG(ERROR) << "lidar points in not in world coordinate" << endl;
        return;
    }
    if(!IsPoseValid())
    {
        LOG(ERROR) << "lidar pose is invalid, unable to transform to local coordinate";
        return;
    }
    Eigen::Matrix3d R_lw = R_wl.transpose();
    Eigen::Vector3d t_lw = -R_lw * t_wl;
    Eigen::Matrix4d T_lw = Eigen::Matrix4d::Identity();
    T_lw.block<3,3>(0,0) = R_lw;
    T_lw.block<3,1>(0,3) = t_lw;

    // 从雷达世界坐标系变回雷达坐标系
    if(!cloud_scan.empty())
        pcl::transformPointCloud(cloud_scan, cloud_scan, T_lw);
    if(!surfFlat.empty())
        pcl::transformPointCloud(surfFlat, surfFlat, T_lw);
    if(!surfLessFlat.empty())
        pcl::transformPointCloud(surfLessFlat, surfLessFlat, T_lw);
    if(!cornerSharp.empty())
        pcl::transformPointCloud(cornerSharp, cornerSharp, T_lw);
    if(!cornerLessSharp.empty())
        pcl::transformPointCloud(cornerLessSharp, cornerLessSharp, T_lw);
    if(!cloud.empty())
        pcl::transformPointCloud(cloud, cloud, T_lw);
    if(!edge_segmented.empty())
        for(pcl::PointCloud<PointType>& s : edge_segmented)
            pcl::transformPointCloud(s, s, T_lw);
    if(!cornerBeforeFilter.empty())
        pcl::transformPointCloud(cornerBeforeFilter, cornerBeforeFilter, T_lw);

    world = false;
}

const Eigen::Vector3d Velodyne::World2Local(Eigen::Vector3d point_w) const
{
    return R_wl.transpose() * point_w - R_wl.transpose() * t_wl;
} 

// 把单个点从雷达坐标系变换到世界坐标系
const Eigen::Vector3d Velodyne::Local2World(Eigen::Vector3d point_local) const
{
    return R_wl * point_local + t_wl;
}


void Velodyne::SetName(std::string _name)
{
    name = _name;
}

void Velodyne::SetPose(const Eigen::Matrix3d _R_wl, const Eigen::Vector3d _t_wl)
{
    R_wl = _R_wl;
    t_wl = _t_wl;
}
void Velodyne::SetPose(const Eigen::Matrix4d T_wl)
{
    R_wl = T_wl.block<3,3>(0,0);
    t_wl = T_wl.block<3,1>(0,3);
}
void Velodyne::SetRotation(const Eigen::Matrix3d _R_wl)
{
    R_wl = _R_wl;
}
void Velodyne::SetTranslation(const Eigen::Vector3d _t_wl)
{
    t_wl = _t_wl;
}

const Eigen::Matrix4d Velodyne::GetPose() const
{
    Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
    T_wl.block<3,3>(0,0) = R_wl;
    T_wl.block<3,1>(0,3) = t_wl;
    return T_wl;
}

const bool Velodyne::IsPoseValid() const
{
    if(!isinf(t_wl(0)) && !isnan(t_wl(0)) && !isinf(t_wl(1)) && !isnan(t_wl(1)) && !isinf(t_wl(2)) && !isnan(t_wl(2)) && !R_wl.isZero())
        return true;
    return false;
}

const bool Velodyne::IsInWorldCoordinate() const
{
    return world;
}

void Velodyne::test()
{
    LoadLidar();
    ReOrderVLP();
    size_t cloudSize = cloud_scan.size();
    cloudState = (int*)malloc(cloudSize * sizeof(int));
    fill(cloudState, cloudState + cloudSize, POINT_NORMAL);
    SegmentGround();
   
    pcl::io::savePCDFileASCII(num2str(id) + "_cloud.pcd", cloud_scan);
    free(cloudState);
    cloudState = NULL;

}



