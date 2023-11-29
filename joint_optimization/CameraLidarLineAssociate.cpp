/*
 * @Author: Diantao Tu
 * @Date: 2021-11-12 17:26:50
 */

#include "CameraLidarLineAssociate.h"


using namespace std;

CameraLidarLineAssociate::CameraLidarLineAssociate(int _rows, int _cols, cv::Mat _img_gray):rows(_rows), cols(_cols), img_gray(_img_gray)
{
}

CameraLidarLineAssociate::CameraLidarLineAssociate(int _rows, int _cols):rows(_rows), cols(_cols)
{
}
const vector<CameraLidarLinePair> CameraLidarLineAssociate::GetAssociatedPairs()
{
    return line_pairs;
}
void CameraLidarLineAssociate::Associate(const std::vector<cv::Vec4f>& lines, const pcl::PointCloud<pcl::PointXYZI> point_cloud, 
                            const Eigen::Matrix4d T_cl)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::transformPointCloud(point_cloud, cloud, T_cl);
    Equirectangular eq(rows, cols);
    // 一条直线在球型投影下会成为一个曲线，在LiDAR和直线相关联的时候，曲线不好操作，因此把一个长直线变成一个个的短直线
    // 用短直线来近似长直线在球型投影下得到的曲线
    vector<cv::Point2f> sub_lines;
    size_t sub_line_count = 0;
    // 记录每个小直线所属于的原本的直线的index
    map<size_t, size_t> subline_to_line;

    for(size_t line_idx = 0; line_idx < lines.size(); line_idx++)
    {
        // 由于一条直线在全景图像上是曲线，首先把这条直线分成一段段的小直线来近似
        vector<cv::Point2f> segments = eq.BreakToSegments(lines[line_idx], 70);
        // 把每一小段直线都用这段直线的中点表示
        for(size_t i = 0; i < segments.size() - 1; i++)
        {
            if(abs(segments[i].x - segments[i+1].x) > 0.8 * cols)       
                continue;
            cv::Point2f center;
            center.x = (segments[i+1].x + segments[i].x) / 2.0; 
            center.y = (segments[i+1].y + segments[i].y) / 2.0; 
            sub_lines.push_back(center);
            subline_to_line[sub_line_count] = line_idx;
            sub_line_count++;
        }
    }

    cv::Mat points(sub_line_count, 2, CV_32F);
    for(int i = 0; i < sub_line_count; i++)
    {
        points.at<float>(i, 0) = sub_lines[i].x;
        points.at<float>(i, 1) = sub_lines[i].y;
    }

    // 建立一棵kd树，衡量距离默认使用L2距离
    cv::flann::Index kdtree(points, cv::flann::KDTreeIndexParams(1)); //此部分建立kd-tree索引同上例，故不做详细叙述

    unsigned queryNum = 3;//用于设置返回邻近点的个数
    vector<float> curr_point(2);//存放 查询点 的容器（本例都是vector类型）
    vector<int> vecIndex(queryNum);//存放返回的点索引
    vector<float> vecDist(queryNum);//存放距离
    cv::flann::SearchParams params(32);//设置knnSearch搜索参数
    // 保存雷达点和图像直线之间的对应关系，由于位姿可能不准确，所以这种对应关系不是唯一的，也就是说，一个雷达点可以对应多个
    // 图像直线，
    std::map<int, vector<size_t> > lidar_line;      // lidar点对应的直线的index      lidar -> line
    std::map<int, vector<size_t> > line_lidar;      // 直线对应的lidar点   line -> lidar

    for(size_t lidar_idx = 0; lidar_idx < cloud.size(); lidar_idx++)
    {
        cv::Point3f point(cloud.points[lidar_idx].x, cloud.points[lidar_idx].y, cloud.points[lidar_idx].z);
        cv::Point2f pixel = eq.CamToImage(point);
        curr_point = {pixel.x, pixel.y};
        kdtree.knnSearch(curr_point, vecIndex, vecDist, queryNum, cv::flann::SearchParams(-1));   
        for(size_t i = 0; i < vecIndex.size(); i++)
        {
            if(vecDist[i] > 60 * 60)
                continue;
            lidar_line[lidar_idx].push_back(subline_to_line[vecIndex[i]]);
            line_lidar[subline_to_line[vecIndex[i]]].push_back(lidar_idx);
        }
    }

    // 遍历每一根直线以及其近邻的雷达投影点
    for(map<int, vector<size_t>>::iterator it = line_lidar.begin(); it != line_lidar.end(); it++)
    {
        pcl::PointCloud<pcl::PointXYZI> line_points;
        // 如果直线对应的雷达点少于10个，就跳过
        if(it->second.size() < 6)
        {
            it->second.clear();
            continue;
        }
        for(size_t lidar_idx : it->second)
        {    
            line_points.push_back(cloud.points[lidar_idx]);
        }

        // 用于debug，显示图像直线以及相关联的雷达点
        // pcl::io::savePCDFileASCII("lidar_points_" + num2str(it->first) + ".pcd", line_points);
        // cv::Mat img_line_cloud = ProjectLidar2PanoramaGray(line_points, img_gray, Eigen::Matrix4f::Identity(), 5);
        // DrawLine(img_line_cloud, lines[it->first], cv::Scalar(0, 0, 255), 3, 2);
        // cv::imwrite("line_and_points_" + num2str(it->first) + ".jpg", img_line_cloud);

        // 用占比最大的segment里的点拟合一个直线
        pcl::ModelCoefficients line_coeff;
        vector<size_t> inliers;
        if(!FitLineRANSAC(line_points, line_coeff, inliers))
        {
            it->second.clear();
            continue;
        }
        
        // 找到所有的内点中距离最远的两个点
        size_t start = 0, end = 0;
        float max_distance = -1;
        for(size_t i = 0; i < inliers.size(); i++)
        {
            for(size_t j = i + 1; j < inliers.size(); j++)
            {
                float distance = PointDistanceSquare(line_points[inliers[i]], line_points[inliers[j]]);
                if( distance > max_distance)
                {
                    max_distance = distance;
                    start = i;
                    end = j;
                }
            }
        }
        
        // 把相距最远的两个内点投影到直线上
        Eigen::Vector3d p1 = ProjectPoint2Line3D(PclPonit2EigenVecd(line_points[start]), line_coeff);
        Eigen::Vector3d p2 = ProjectPoint2Line3D(PclPonit2EigenVecd(line_points[end]), line_coeff);
        
        CameraLidarLinePair line_pair;
        line_pair.image_line = lines[it->first];
        line_pair.lidar_line_start = p1;    // 注意这些点都是在相机坐标系下的
        line_pair.lidar_line_end = p2;
        line_pair.angle = FLT_MAX;
        line_pairs.push_back(line_pair);

        // 之后的都是可视化处理，只在debug使用
        continue;
        Eigen::Vector2d p1_pixel = eq.SphereToImage(eq.CamToSphere(p1));
        Eigen::Vector2d p2_pixel = eq.SphereToImage(eq.CamToSphere(p2));

        vector<cv::Vec4f> tmp_lines = {lines[it->first], cv::Vec4f(p1_pixel.x(), p1_pixel.y(), p2_pixel.x(), p2_pixel.y())};
        vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                cv::Scalar(255,0,255)}; 
        cv::Mat img_line = DrawLinesOnImage(img_gray, tmp_lines, colors, 4, true);
        
        pcl::PointCloud<pcl::PointXYZI> tmp_cloud;
        tmp_cloud.push_back(line_points[start]);
        tmp_cloud.push_back(line_points[end]);
        img_line = ProjectLidar2PanoramaRGB(tmp_cloud, img_line, Eigen::Matrix4d::Identity(), 0.5, 5, 5);
        cv::imwrite("lidar_line_" + num2str(it->first) + ".jpg", img_line);
    }

    // 把所有相关联的直线画在图像上, 用于debug
    // {
    //     cv::Mat img_lidar_line;
    //     img_lidar_line = DrawLinePairsOnImage(img_gray, line_pairs, Eigen::Matrix4f::Identity());
    //     cv::imwrite("lidar_line_all.jpg", img_lidar_line);
    // }
    
    Filter(true, true);
    
    // 把经过过滤的直线画在图像上，用于debug
    // {
    //     cv::Mat img_lidar_line;
    //     img_lidar_line = DrawLinePairsOnImage(img_gray, line_pairs, Eigen::Matrix4f::Identity());
    //     cv::imwrite("lidar_line_all_filtered.jpg", img_lidar_line);
    // }

    // 把匹配的直线重新变回LiDAR坐标系下
    Eigen::Matrix4d T_lc = T_cl.inverse();
    for(CameraLidarLinePair& lp : line_pairs)
    {
        lp.lidar_line_start = (T_lc * lp.lidar_line_start.homogeneous()).hnormalized();
        lp.lidar_line_end = (T_lc * lp.lidar_line_end.homogeneous()).hnormalized();
    }
}


void CameraLidarLineAssociate::Associate(const std::vector<cv::Vec4f>& lines, const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const Eigen::Matrix4d T_cl)
{
    // 把segment cloud都放到一个点云里，同时用每个点的intensity来表示这个点所属的segment
    const size_t seg_size = segmented_cloud.size();
    pcl::PointCloud<pcl::PointXYZI> cloud;
    for(size_t i = 0; i < seg_size; i++)
    {
        const pcl::PointCloud<pcl::PointXYZI>& seg = segmented_cloud[i];
        for(pcl::PointXYZI p : seg.points)
        {
            p.intensity = i;
            cloud.push_back(p);
        }
    }
    pcl::transformPointCloud(cloud, cloud, T_cl);
    Equirectangular eq(rows, cols);
    // 一条直线在球型投影下会成为一个曲线，在LiDAR和直线相关联的时候，曲线不好操作，因此把一个长直线变成一个个的短直线
    // 用短直线来近似长直线在球型投影下得到的曲线
    vector<cv::Point2f> sub_lines;
    size_t sub_line_count = 0;
    // 记录每个小直线所属于的原本的直线的index
    map<size_t, size_t> subline_to_line;

    for(size_t line_idx = 0; line_idx < lines.size(); line_idx++)
    {
        // 由于一条直线在全景图像上是曲线，首先把这条直线分成一段段的小直线来近似
        vector<cv::Point2f> segments = eq.BreakToSegments(lines[line_idx], 70);
        // 把每一小段直线都用这段直线的中点表示
        for(size_t i = 0; i < segments.size() - 1; i++)
        {
            if(abs(segments[i].x - segments[i+1].x) > 0.8 * cols)       
                continue;
            cv::Point2f center;
            center.x = (segments[i+1].x + segments[i].x) / 2.0; 
            center.y = (segments[i+1].y + segments[i].y) / 2.0; 
            sub_lines.push_back(center);
            subline_to_line[sub_line_count] = line_idx;
            sub_line_count++;
        }
    }

    cv::Mat points(sub_line_count, 2, CV_32F);
    for(int i = 0; i < sub_line_count; i++)
    {
        points.at<float>(i, 0) = sub_lines[i].x;
        points.at<float>(i, 1) = sub_lines[i].y;
    }

    // 建立一棵kd树，衡量距离默认使用L2距离
    cv::flann::Index kdtree(points, cv::flann::KDTreeIndexParams(1)); //此部分建立kd-tree索引同上例，故不做详细叙述

    unsigned queryNum = 3;//用于设置返回邻近点的个数
    vector<float> curr_point(2);//存放 查询点 的容器（本例都是vector类型）
    vector<int> vecIndex(queryNum);//存放返回的点索引
    vector<float> vecDist(queryNum);//存放距离
    cv::flann::SearchParams params(32);//设置knnSearch搜索参数
    std::map<int, int> lidar_line;      // lidar点对应的直线的index      lidar -> line
    std::map<int, vector<size_t> > line_lidar;      // 直线对应的lidar点   line -> lidar

    for(size_t lidar_idx = 0; lidar_idx < cloud.size(); lidar_idx++)
    {
        cv::Point3f point(cloud.points[lidar_idx].x, cloud.points[lidar_idx].y, cloud.points[lidar_idx].z);
        cv::Point2f pixel = eq.CamToImage(point);
        curr_point = {pixel.x, pixel.y};
        kdtree.knnSearch(curr_point, vecIndex, vecDist, queryNum, cv::flann::SearchParams(-1));   
        if(vecDist[0] > 60 * 60)
        {
            lidar_line[lidar_idx] = -1;
            continue;
        }
        lidar_line[lidar_idx] = subline_to_line[vecIndex[0]];
        line_lidar[subline_to_line[vecIndex[0]]].push_back(lidar_idx);
    }

    // 遍历每一根直线以及其近邻的雷达投影点
    for(map<int, vector<size_t>>::iterator it = line_lidar.begin(); it != line_lidar.end(); it++)
    {
        pcl::PointCloud<pcl::PointXYZI> line_points;
        // 如果直线对应的雷达点少于10个，就跳过
        if(it->second.size() < 6)
        {
            it->second.clear();
            continue;
        }
        // 找到当前直线对应的雷达点，并且统计雷达点所属的segment，只有大部分雷达点都属于同一个segment才认为
        // 这个直线匹配是正确的
        int point_each_seg[seg_size] = {0};
        for(size_t lidar_idx : it->second)
        {    
            const pcl::PointXYZI& p = cloud.points[lidar_idx];
            line_points.push_back(p);
            point_each_seg[int(p.intensity)]++;
        }
        int max_position = max_element(point_each_seg, point_each_seg + seg_size) - point_each_seg;
        if(point_each_seg[max_position] < 0.7 * it->second.size())
        {
            it->second.clear();
            continue;
        }
        // 用占比最大的segment里的点拟合一个直线
        pcl::ModelCoefficients line_coeff;
        vector<size_t> inliers;
        if(!FitLineRANSAC(segmented_cloud[max_position], line_coeff, inliers))
        {
            it->second.clear();
            continue;
        }

        // 找到所有的内点中距离最远的两个点
        size_t start = 0, end = 0;
        float max_distance = -1;
        for(size_t i = 0; i < inliers.size(); i++)
        {
            for(size_t j = i + 1; j < inliers.size(); j++)
            {
                float distance = PointDistanceSquare(segmented_cloud[max_position][inliers[i]], segmented_cloud[max_position][inliers[j]]);
                if( distance > max_distance)
                {
                    max_distance = distance;
                    start = i;
                    end = j;
                }
            }
        }
        
        // 把相距最远的两个内点投影到直线上
        Eigen::Vector3d p1 = ProjectPoint2Line3D(PclPonit2EigenVecd(segmented_cloud[max_position][start]), line_coeff);
        Eigen::Vector3d p2 = ProjectPoint2Line3D(PclPonit2EigenVecd(segmented_cloud[max_position][end]), line_coeff);
        
        CameraLidarLinePair line_pair;
        line_pair.image_line = lines[it->first];
        line_pair.lidar_line_start = p1;    // 注意这些点都是在相机坐标系下的
        line_pair.lidar_line_end = p2;
        line_pair.angle = FLT_MAX;
        line_pairs.push_back(line_pair);
    }

    Filter(true, true);
    
    // 把匹配的直线重新变回LiDAR坐标系下
    Eigen::Matrix4d T_lc = T_cl.inverse();
    for(CameraLidarLinePair& lp : line_pairs)
    {
        lp.lidar_line_start = (T_lc * lp.lidar_line_start.homogeneous()).hnormalized();
        lp.lidar_line_end = (T_lc * lp.lidar_line_end.homogeneous()).hnormalized();
    }
}

void CameraLidarLineAssociate::AssociateByAngle(const std::vector<cv::Vec4f>& lines, 
                    const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const eigen_vector<Vector6d>& segment_coeffs,
                    const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
                    const std::vector<std::set<int>>& point_to_segment,
                    const eigen_vector<Eigen::Vector3d>& line_end_points,
                    const Eigen::Matrix4d T_cl, const bool multiple_association,
                    const std::vector<bool>& image_line_mask,
                    const std::vector<bool>& lidar_line_mask)
{
    assert(segment_coeffs.size() == segmented_cloud.size());
    // 判断一下mask是否存在，如果存在则要求mask大小和直线大小相同，如果不存在就新建一个全是true的mask，代表所有直线都参与匹配
    vector<bool> image_mask, lidar_mask;
    if(!image_line_mask.empty())
    {
        assert(image_line_mask.size() == lines.size());
        image_mask = image_line_mask;
    }
    else 
        image_mask = vector<bool>(lines.size(), true);
    if(!lidar_line_mask.empty())
    {
        assert(lidar_line_mask.size() == segmented_cloud.size());
        lidar_mask = lidar_line_mask;
    }
    else 
        lidar_mask = vector<bool>(segmented_cloud.size(), true);

    line_pairs.clear();
    // 记录一下每个点到中心的距离
    vector<float> range;
    for(const pcl::PointXYZI& point : point_cloud)
        range.push_back(point.x * point.x + point.y * point.y + point.z * point.z);
    // 把雷达点云变换到相机坐标系下
    pcl::PointCloud<pcl::PointXYZI> cloud(point_cloud);
    pcl::transformPointCloud(cloud, cloud, T_cl);
    eigen_vector<Eigen::Vector3d> lidar_lines_endpoint;     // 这是在相机坐标系下的
    eigen_vector<Eigen::Vector4d> lidar_plane;              // 这是在相机坐标系下的
    for(size_t i = 0; i < segmented_cloud.size(); i++)
    {
        // 把相距最远的两个内点投影到直线上
        Eigen::Vector3d p1 = line_end_points[i * 2];
        Eigen::Vector3d p2 = line_end_points[i * 2 + 1];
        p1 = (T_cl * p1.homogeneous()).hnormalized();
        p2 = (T_cl * p2.homogeneous()).hnormalized();
        lidar_lines_endpoint.push_back(p1);
        lidar_lines_endpoint.push_back(p2);
        Eigen::Vector4d plane = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));
        plane.normalize();      // 因为平面是过原点的，所以d=0，因此可以直接用normalize
        lidar_plane.push_back(plane);
    }
    // 角度的阈值是3度
    const double angle_threshold = 3.0 / 180.0 * M_PI;
    Equirectangular eq(rows, cols);
    for(size_t image_line_idx = 0; image_line_idx < lines.size(); image_line_idx++)
    {
        if(!image_mask[image_line_idx])
            continue;
        const cv::Vec4f& l = lines[image_line_idx];
        // 把起始点和终止点都变换成单位圆上的XYZ坐标
        Eigen::Vector3d p1 = eq.ImageToCam(Eigen::Vector2d(l[0], l[1]));
        Eigen::Vector3d p2 = eq.ImageToCam(Eigen::Vector2d(l[2], l[3]));
        Eigen::Vector4d image_plane = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));
        image_plane.normalize();        // 变成单位向量
        Eigen::Vector3d p4 = (p1 + p2) / 2.0;
        // 计算图像直线在球面上所形成的弧所对应的圆心角
        const double image_line_scope = VectorAngle3D(p1.data(), p4.data());
        // 记录LiDAR的每个segment分别有多少点是属于当前图像直线的，key=seg id , value=segment里点的数量
        map<size_t, size_t> seg_count;
        for(size_t i = 0; i < cloud.size(); i++)
        {
            const Eigen::Vector3d p = PclPonit2EigenVecd(cloud.points[i]);
            // 只考虑在15米范围内的点
            if(range[i] > 15*15)
                continue;
            Eigen::Vector3d point_projected;
            ProjectPointToPlane(p.data(), image_plane.data(), point_projected.data(), true);
            // 计算从球心出发分别射向p和point_projected的两条射线的夹角,要求射线夹角不能大于阈值，这就是要求
            // LiDAR点不能离图像直线所在的平面太远
            if(VectorAngle3D(p.data(), point_projected.data()) >= angle_threshold)
                continue;
            // 这是要求雷达点基本在图像直线的范围内，也就是说如果雷达点投影到图像上，那么投影点应该在图像直线附近
            if(VectorAngle3D(p4.data(), point_projected.data()) >= image_line_scope + angle_threshold)
                continue;
            for(const int& seg_id : point_to_segment[i])
                seg_count[seg_id]++;
        }
        for(map<size_t, size_t>::const_iterator it = seg_count.begin(); it != seg_count.end(); it++)
        {
            if(it->second < segmented_cloud[it->first].size() / 2)
                continue;
            const size_t& lidar_line_idx = it->first;
            if(!lidar_mask[lidar_line_idx])
                continue;
            double angle = PlaneAngle(image_plane.data(), lidar_plane[lidar_line_idx].data(), true);
            if(angle > angle_threshold)
                continue;
            // 把雷达的起始点和终止点投影到图像上，要求这两个点是在图像直线的范围内的
            // if(VectorAngle3D(ProjectPointToPlane(lidar_lines_endpoint[lidar_line_idx * 2], image_plane, true), p4) > image_line_scope + angle_threshold)
            //     continue;
            // if(VectorAngle3D(ProjectPointToPlane(lidar_lines_endpoint[lidar_line_idx * 2 + 1], image_plane, true), p4) > image_line_scope + angle_threshold)
            //     continue;
            Eigen::Vector3d lidar_middle = (lidar_lines_endpoint[lidar_line_idx * 2] + lidar_lines_endpoint[lidar_line_idx * 2 + 1]) / 2.f;
            Eigen::Vector3d lidar_middle_projected;
            ProjectPointToPlane(lidar_middle.data(), image_plane.data(), lidar_middle_projected.data(), true);
            if(VectorAngle3D(lidar_middle_projected.data(), p4.data()) > image_line_scope)
                continue;
            float angle2 = VectorAngle3D(lidar_middle.data(), lidar_middle_projected.data());
            if(angle2 > angle_threshold / 2.0)
                continue;
            float score = angle + angle2;

            CameraLidarLinePair line_pair;
            line_pair.image_line = lines[image_line_idx];
            line_pair.lidar_line_start = lidar_lines_endpoint[it->first * 2];
            line_pair.lidar_line_end = lidar_lines_endpoint[it->first * 2 + 1];
            line_pair.image_line_id = image_line_idx;
            line_pair.lidar_line_id = it->first;
            line_pair.angle = score;
            line_pairs.push_back(line_pair);
        }
    }

    Filter(false, true);

    if(!multiple_association)
        UniqueLinePair(lines, lidar_lines_endpoint);

    // 把匹配的直线重新变回LiDAR坐标系下
    Eigen::Matrix4d T_lc = T_cl.cast<double>().inverse();
    for(CameraLidarLinePair& lp : line_pairs)
    {
        lp.lidar_line_start = (T_lc * lp.lidar_line_start.homogeneous()).hnormalized();
        lp.lidar_line_end = (T_lc * lp.lidar_line_end.homogeneous()).hnormalized();
    }
}

Eigen::Matrix4d CameraLidarLineAssociate::AssociateRandomDisturbance(const std::vector<cv::Vec4f>& lines, 
                    const std::vector<pcl::PointCloud<pcl::PointXYZI>>& segmented_cloud,
                    const eigen_vector<Vector6d>& segment_coeffs,
                    const pcl::PointCloud<pcl::PointXYZI>& point_cloud,
                    const std::vector<std::set<int>>& point_to_segment,
                    const eigen_vector<Eigen::Vector3d>& line_end_points,
                    const Eigen::Matrix4d T_cl, const bool multiple_association,
                    const std::vector<bool>& image_line_mask,
                    const std::vector<bool>& lidar_line_mask )
{
    float rotation_step = 1.0;
    float translation_step = 0.05;
    float scale = 1;
    vector<CameraLidarLinePair> best_line_pair;
    float best_average_score = -1;
    Eigen::Matrix4d curr_T_cl = T_cl;
    for(int iter = 0; iter < 1; iter++)
    {
        bool find_better_result = false;
        // 对当前的外参进行扰动，得到扰动后的结果
        eigen_vector<Eigen::Matrix4d> perturbed_T_cl = PerturbCalibration(curr_T_cl, scale * rotation_step, scale * translation_step);
        // 使用扰动后的结果进行直线关联
        for(size_t i = 0; i < perturbed_T_cl.size(); i++)
        {
            AssociateByAngle(lines, segmented_cloud, segment_coeffs, point_cloud, point_to_segment, line_end_points, perturbed_T_cl[i], 
                            multiple_association, image_line_mask, lidar_line_mask);
            float average_score = 0;
            for(const CameraLidarLinePair& p : line_pairs)
                average_score += p.angle;
            average_score /= line_pairs.size();
            // 只有两种情况下会更新最终的匹配信息：
            // 1. 当前匹配的直线数更多  2. 匹配的直线数量一样，但是平均误差更小
            if(line_pairs.size() > best_line_pair.size() || 
                (line_pairs.size() == best_line_pair.size() && average_score < best_average_score))
            {
                best_line_pair = line_pairs;
                best_average_score = average_score;
                curr_T_cl = perturbed_T_cl[i];
                // i=0的时候得到的结果就是初始外参得到的结果，不能认为是找到了更好的外参
                if(i > 0)
                    find_better_result = true;
            }
        }
        // 如果经过扰动后没能找到更好的结果，那就把扰动的程度降低，然后重新迭代一次
        if(!find_better_result && scale == 1)
        {
            scale *= 2;
            iter--;
        }
        // 如果降低扰动程度后，还是没能找到更好的结果，那就认为当前已经是最优了，直接break即可
        else if(!find_better_result && scale != 1)
            break;
        // 这种情况是在缩小扰动范围后找到了更好的结果，那么就恢复到原本的扰动程度，继续迭代
        // 不然之后的迭代都是缩小范围的扰动（其实我也不知道到底应不应该恢复到scale=1）
        else if(scale != 1)
            scale = 1;
    }
    line_pairs = best_line_pair;
    return curr_T_cl;
}

Eigen::Matrix4d CameraLidarLineAssociate::AssociateRandomDisturbance(const std::vector<cv::Vec4f>& lines,
                    const Frame& frame, const Velodyne& lidar,
                    const Eigen::Matrix4d T_cl, const bool multiple_association,
                    const std::vector<bool>& image_line_mask,
                    const std::vector<bool>& lidar_line_mask)
{
    const int max_iteration = 15;
    float rotation_step = 0.5;
    float translation_step = 0.05;
    float scale = 1;
    vector<CameraLidarLinePair> best_line_pair;
    float best_average_score = -1;
    cv::Mat img_gray = frame.GetImageGray();
    Eigen::Matrix4d curr_T_cl = T_cl;
    vector<float> score_each_iteration(max_iteration);     // 每次迭代过程中平均的误差
    vector<int> pairs_each_iteration(max_iteration);       // 每次迭代后得到的匹配的直线数
    for(int iter = 0; iter < max_iteration; iter++)
    {
        bool find_better_result = false;
        // 对当前的外参进行扰动，得到扰动后的结果
        eigen_vector<Eigen::Matrix4d> perturbed_T_cl = PerturbCalibration(curr_T_cl, scale * rotation_step, scale * translation_step);
        // 使用扰动后的结果进行直线关联
        for(size_t i = 0; i < perturbed_T_cl.size(); i++)
        {
            AssociateByAngle(lines, lidar.edge_segmented, lidar.segment_coeffs, lidar.cornerLessSharp, lidar.point_to_segment, 
                            lidar.end_points, perturbed_T_cl[i], 
                            multiple_association, image_line_mask, lidar_line_mask);
            float average_score = 0;
            for(const CameraLidarLinePair& p : line_pairs)
                average_score += p.angle;
            average_score /= line_pairs.size();
            // 只有两种情况下会更新最终的匹配信息：
            // 1. 当前匹配的直线数更多  2. 匹配的直线数量一样，但是平均误差更小
            if(line_pairs.size() > best_line_pair.size() || 
                (line_pairs.size() == best_line_pair.size() && average_score < best_average_score))
            {
                best_line_pair = line_pairs;
                best_average_score = average_score;
                curr_T_cl = perturbed_T_cl[i];
                // i=0的时候得到的结果就是初始外参得到的结果，不能认为是找到了更好的外参
                if(i > 0)
                    find_better_result = true;
            }
        }
        
        cv::Mat img_line = DrawLinePairsOnImage(img_gray, best_line_pair, curr_T_cl, 7, true);                
        cv::imwrite("./line_pair_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(iter+2) + ".jpg", img_line);
        
        cv::Mat img_cloud = ProjectLidar2PanoramaRGB(lidar.cloud, img_gray,
                    curr_T_cl, 1.5, 20, 5);
        cv::imwrite("./cloud_project_" + num2str(frame.id) + "_" + num2str(lidar.id) + "_" + num2str(iter+2) + ".jpg", img_cloud);


        LOG(INFO) << "iter " << iter + 2 << " ================";
        LOG(INFO) << "line pairs : " << best_line_pair.size();
        LOG(INFO) << "average angle: " << best_average_score;
        pairs_each_iteration[iter] = best_line_pair.size();
        score_each_iteration[iter] = best_average_score;
        // 已经迭代了至少5次，如果最后几次都没能使得匹配的直线数量增加，那么就认为无法找到更好的结果了
        // 把find_better_result变为false
        if(iter >= 4)
        {
            int count = 0;
            for(size_t i = iter; i > iter - 4; i--)
                count += (pairs_each_iteration[i] == pairs_each_iteration[i-1]);
            find_better_result = ((count < 4) && find_better_result);
        }
        // 如果经过扰动后没能找到更好的结果，那就把扰动的程度降低，然后重新迭代一次
        if(!find_better_result && scale == 1)
        {
            scale /= 2;
            iter--;
        }
        // 如果降低扰动程度后，还是没能找到更好的结果，那就认为当前已经是最优了，直接break即可
        else if(!find_better_result && scale != 1)
            break;
        // 这种情况是在缩小扰动范围后找到了更好的结果，那么就恢复到原本的扰动程度，继续迭代
        // 不然之后的迭代都是缩小范围的扰动（其实我也不知道到底应不应该恢复到scale=1）
        else if(scale != 1)
            scale = 1;
    }
    line_pairs = best_line_pair;
    return curr_T_cl;

}


// 对相关联的直线进行过滤，过滤的方法是两个
// 1. 计算两个直线的夹角，夹角太大的过滤掉
// 2. 把每个直线都分为一个个的小段，用于统计在图像上的长度，如果长度太短也直接过滤掉
void CameraLidarLineAssociate::Filter(bool filter_by_angle, bool filter_by_length)
{
    float min_length_threshold = 100;
    float max_length_threshold = 2000;
    vector<CameraLidarLinePair> good_pair;
    Equirectangular eq(rows, cols);
    for(CameraLidarLinePair& p : line_pairs)
    {
        if(filter_by_angle)
        {
            // 雷达直线投影到球面后得到的弧与球心形成一个平面，计算这个平面的法向量
            Eigen::Vector3d p1 = p.lidar_line_start;
            Eigen::Vector3d p2 = p.lidar_line_end;
            Eigen::Vector4d plane_lidar = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));
            plane_lidar.normalize();
            
            // 图像直线投影到球面后和球心形成一个平面，计算这个平面的法向量
            p1 = eq.ImageToCam(Eigen::Vector2d(p.image_line[0], p.image_line[1]));
            p2 = eq.ImageToCam(Eigen::Vector2d(p.image_line[2], p.image_line[3]));
            Eigen::Vector4d plane_img = FormPlane(p1, p2, Eigen::Vector3d(0,0,0));
            plane_img.normalize();
            // 计算两个平面的夹角，如果太大就过滤掉
            double plane_angle = PlaneAngle(plane_lidar.data(), plane_img.data(), true) * 180.0 / M_PI;
            if(plane_angle > 5)
                continue;
            p.angle = plane_angle;
            // 要求雷达直线要基本在图像直线内，也就是雷达的投影不能超过图像直线范围
            double image_line_angle = VectorAngle3D(p1.data(), p2.data()) / 2.0;
            Eigen::Vector3d lidar_start_projected;
            ProjectPointToPlane(p.lidar_line_start.data(), plane_img.data(), lidar_start_projected.data(), true);
            Eigen::Vector3d lidar_end_projected;
            ProjectPointToPlane(p.lidar_line_end.data(), plane_img.data(), lidar_end_projected.data(), true);
            Eigen::Vector3d image_middle = (p1 + p2) / 2.0;
            if(VectorAngle3D(lidar_start_projected.data(), image_middle.data()) > image_line_angle)
                continue;
            if(VectorAngle3D(lidar_end_projected.data(), image_middle.data()) > image_line_angle)
                continue;
            // 如果雷达点到平面距离太大，过滤掉
            Eigen::Vector3d p1_cam = p.lidar_line_start.normalized() * 5;
            Eigen::Vector3d p2_cam = p.lidar_line_end.normalized() * 5;
            float distance = min(PointToPlaneDistance(plane_img.data(), p1_cam.data(),true), 
                                PointToPlaneDistance(plane_img.data(), p2_cam.data(),true));
            if(distance > 0.4)
                continue;
        }
        
        if(filter_by_length)
        {
            // 把雷达直线都分成一段一段的
            cv::Point2f p1_pixel = eq.CamToImage(cv::Point3f(p.lidar_line_start.x(), p.lidar_line_start.y(), p.lidar_line_start.z()));
            cv::Point2f p2_pixel = eq.CamToImage(cv::Point3f(p.lidar_line_end.x(), p.lidar_line_end.y(), p.lidar_line_end.z()));

            vector<cv::Point2f> lidar_line_seg = eq.BreakToSegments(p1_pixel, p2_pixel, 100);
    
            float lidar_line_length = 0;    // 雷达投影直线的总长度
            for(size_t i = 0; i < lidar_line_seg.size() - 1; i++)
            {
                if(abs(lidar_line_seg[i].x - lidar_line_seg[i+1].x) > 0.8 * cols)       
                    continue;
                lidar_line_length += sqrt(PointDistanceSquare(lidar_line_seg[i], lidar_line_seg[i+1]));
            }
            // 如果雷达在图像上投影的线太短，过滤掉
            if(lidar_line_length < min_length_threshold)
                continue;
            if(lidar_line_length > max_length_threshold)
                continue;
            
            /*
            // 把图像直线分成一段一段的
            vector<cv::Point2f> image_line_seg = eq.BreakToSegments(p.image_line, 100);
            float image_line_length = 0;    // 图像直线长度
            for(size_t i = 0; i < image_line_seg.size() - 1; i++)
            {
                if(abs(image_line_seg[i].x - image_line_seg[i+1].x) > 0.8 * cols)       
                    continue;
                image_line_length += sqrt(PointDistanceSquare(image_line_seg[i], image_line_seg[i+1]));
            }
            // 如果在图像直线的长度小于LiDAR直线，也认为不可靠
            if(image_line_length < lidar_line_length)
                continue;
            */
        }
        good_pair.push_back(p);
    }
    line_pairs.clear();
    good_pair.swap(line_pairs);

}

bool CameraLidarLineAssociate::FitLineRANSAC(const pcl::PointCloud<pcl::PointXYZI>& cloud, pcl::ModelCoefficients& line_coeff, 
                                vector<size_t>& inlier)
{
    //创建一个模型参数对象，用于记录结果
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  //inliers表示内点的序号
    pcl::SACSegmentation<pcl::PointXYZI> seg;     // 创建一个分割器     
    seg.setOptimizeCoefficients(false);     // 不使用最小二乘法进行优化，因为如果内点数目少于3个，会直接报错
    seg.setModelType(pcl::SACMODEL_LINE);  // Mandatory-设置目标几何形状
    seg.setMethodType(pcl::SAC_RANSAC);     //分割方法：随机采样法
    seg.setDistanceThreshold(0.1);         //设置误差容忍范围，也就是阈值
    seg.setInputCloud(cloud.makeShared());               //输入点云
    seg.segment(*inliers, line_coeff);   //分割点云，获得平面和法向量

    if(inliers->indices.size() < 3)
        return false;
    // 如果内点数足够多，就用最小二乘法计算一个更好的结果    
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(cloud, *inliers, centroid);
    Eigen::Matrix3f covariance_matrix;
    pcl::computeCovarianceMatrix (cloud, *inliers, centroid, covariance_matrix);
    line_coeff.values[0] = centroid[0];
    line_coeff.values[1] = centroid[1];
    line_coeff.values[2] = centroid[2];

    EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
    EIGEN_ALIGN16 Eigen::Vector3f eigen_vectors;
    pcl::eigen33 (covariance_matrix, eigen_values); // 计算特征值
    pcl::computeCorrespondingEigenVector (covariance_matrix, eigen_values [2], eigen_vectors);

    line_coeff.values[3] = eigen_vectors[0];
    line_coeff.values[4] = eigen_vectors[1];
    line_coeff.values[5] = eigen_vectors[2];
    for(const auto& idx : inliers->indices)
        inlier.push_back(idx);
    return true;
}

void CameraLidarLineAssociate::UniqueLinePair(const std::vector<cv::Vec4f>& lines, const eigen_vector<Eigen::Vector3d>& lidar_lines_endpoint)
{
    // 这是一个临时的结构体，为了简洁，所以就写在这里了，不放在外面
    // 这个用于记录一个评分，这样如果出现两条LiDAR直线对应于同一条图像直线的时候，
    // 或者两条图像直线对应同一条雷达的时候，就可以根据评分来确定
    // 当前直线到底和哪条LiDAR直线相关联
    struct PairScore
    {
        int idx;
        float score;        // 这个评分其实是越低越好，因为代表了角度误差，误差越小越好
        PairScore(const int& _idx, const float& _score):idx(_idx), score(_score)
        {}
    };
    // 记录每条图像直线对应的雷达直线以及这对直线的评分，
    // image2lidar   key=图像直线id   value=LiDAR直线id以及评分
    // lidar2image   key=LiDAR直线id   value=图像直线id以及评分
    map<int, PairScore> image2lidar;
    map<int, PairScore> lidar2image;
    vector<CameraLidarLinePair> unique_line_pairs;
    for(const CameraLidarLinePair& pair : line_pairs)
    {
        const int& image_line_idx = pair.image_line_id;
        const int& lidar_line_idx = pair.lidar_line_id;
        const float& score = pair.angle;
        map<int, PairScore>::iterator it_image2lidar = image2lidar.find(image_line_idx);
        map<int, PairScore>::iterator it_lidar2image = lidar2image.find(lidar_line_idx);
        // 如果当前图像直线没有和其他雷达直线相关联，而且当前LiDAR直线也没和其他图像直线相关联，那就直接保存这对匹配关系
        if(it_image2lidar == image2lidar.end() && it_lidar2image == lidar2image.end())
        {
            image2lidar.insert({image_line_idx, PairScore(lidar_line_idx, score)});
            lidar2image.insert({lidar_line_idx, PairScore(image_line_idx, score)});
        }
        
        // 如果当前图像直线已经和其他雷达直线关联了,但当前雷达直线没和其他图像直线相关联
        // 那么就判断一下当前匹配对的评分是否低于已有匹配对评分，决定是否保留下来。
        // 在保留时,要先删去原本的lidar2image中LiDAR直线,因为它已经被新的匹配对取代了
        if(it_image2lidar != image2lidar.end() && it_lidar2image == lidar2image.end())
        {
            if(score < it_image2lidar->second.score)
            {
                lidar2image.erase(lidar2image.find(it_image2lidar->second.idx));
                it_image2lidar->second = PairScore(lidar_line_idx, score);
                lidar2image.insert({lidar_line_idx, PairScore(image_line_idx, score)});
            }
        }
        // 如果当前图像直线没和其他雷达直线关联，但当前雷达直线和其他图像直线关联了，
        // 处理方法和上面的一样，就是调换一下变量
        if(it_image2lidar == image2lidar.end() && it_lidar2image != lidar2image.end())
        {
            if(score < it_lidar2image->second.score)
            {
                image2lidar.erase(image2lidar.find(it_lidar2image->second.idx));
                it_lidar2image->second = PairScore(image_line_idx, score);
                image2lidar.insert({image_line_idx, PairScore(lidar_line_idx, score)});
            }
        }
        // 如果当前的两个直线都和其他直线相关联了，那么处理会复杂一些,为了方便说明，使用图像直线L1 L2和雷达直线A B来说明
        // 假设目前已经存在的匹配关系是 L1-A  L2-B，也就是L1和A匹配，L2和B匹配，现在新出现的匹配关系是L1-B
        // 那么就会有it_image2lidar=L1-A    it_lidar2image=L2-B，以下分为四种情况
        // 1. L1-B的评分是三者中最低的，也就是说要新增L1-B，同时删去L1-A L2-B
        // 2. 评分大小为 L1-A < L1-B < L2-B，这种情况下，L2-B被L1-B取代，而L1-B又被L1-A取代，所以最终结果就是
        //      删去L2-B，其他不变
        // 3. 评分大小为 L2-B < L1-B < L1-A ，这种情况下，L1-A被L1-B取代，而L1-B又被L2-B取代，所以最终结果就是
        //      删去L1-A，其他不变
        // 4. L1-B的评分是三者中最高的，那么就什么都不变
        if(it_image2lidar != image2lidar.end() && it_lidar2image != lidar2image.end())
        {
            // 对应于情况1
            if(score < min(it_image2lidar->second.score, it_lidar2image->second.score))
            {
                image2lidar.erase(it_lidar2image->second.idx);
                lidar2image.erase(it_image2lidar->second.idx);
                image2lidar.erase(it_image2lidar);
                lidar2image.erase(it_lidar2image);
                image2lidar.insert({image_line_idx, PairScore(lidar_line_idx, score)});
                lidar2image.insert({lidar_line_idx, PairScore(image_line_idx, score)});
            }
            // 对应于情况2
            if(score > it_image2lidar->second.score && score < it_lidar2image->second.score)
            {
                // 下面这三行代码当然可以直接用一行代码代替，也就是
                // assert(image2lidar.erase(it_lidar2image->second.idx));
                // 但是我发现以上的写法在debug期间没问题，但是release模式下会偶尔出现无法从map里erase的问题
                // 也就是说，相同的输入，debug和release模式有着不同的结果，因此改写成了下面的这三行代码，
                // 这种写法就没问题了
                it_image2lidar = image2lidar.find(it_lidar2image->second.idx);
                assert(it_image2lidar != image2lidar.end());
                image2lidar.erase(it_image2lidar);
                lidar2image.erase(it_lidar2image);
            }
            // 对应于情况3
            if(score < it_image2lidar->second.score && score > it_lidar2image->second.score)
            {
                it_lidar2image = lidar2image.find(it_image2lidar->second.idx);
                assert(it_lidar2image != lidar2image.end());
                lidar2image.erase(it_lidar2image);
                image2lidar.erase(it_image2lidar);
            }
        }
    }
    // 这是对lidar2image 和 image2lidar的检查，确保两者确实是相互匹配的
    assert(image2lidar.size() == lidar2image.size());
    for(map<int, PairScore>::const_iterator it_image2lidar = image2lidar.begin(); 
            it_image2lidar != image2lidar.end(); it_image2lidar++)
    {
        map<int, PairScore>::const_iterator it_lidar2image = lidar2image.find(it_image2lidar->second.idx);
        assert(it_lidar2image != lidar2image.end());
        assert(it_image2lidar->second.score == it_lidar2image->second.score);
    }

    line_pairs.clear();
    for(map<int, PairScore>::const_iterator it = image2lidar.begin(); it != image2lidar.end(); it++)
    {
        CameraLidarLinePair line_pair;
        line_pair.image_line = lines[it->first];
        line_pair.lidar_line_start = lidar_lines_endpoint[it->second.idx * 2];
        line_pair.lidar_line_end = lidar_lines_endpoint[it->second.idx * 2 + 1];
        line_pair.image_line_id = it->first;
        line_pair.lidar_line_id = it->second.idx;
        line_pair.angle = it->second.score;
        line_pairs.push_back(line_pair);
    }
}

eigen_vector<Eigen::Matrix4d> CameraLidarLineAssociate::PerturbCalibration(const Eigen::Matrix4d& T_cl,
                const float rotation_step, const float translation_step)
{
    // 这里为了方便理解，使用了两个vector来保存经过扰动的外参，分别称为perturbed1和perturbed2，这两个是轮流用的
    // 也就是1扰动得到2，然后2扰动得到1，接着1扰动得到2，如此循环，直到在6个方向上的所有扰动都完成了
    eigen_vector<Eigen::Matrix4d> perturbed1 = {T_cl};
    eigen_vector<Eigen::Matrix4d> perturbed2;
    Eigen::Matrix4d delta_T = Eigen::Matrix4d::Identity();
    
    // 在每个轴上的旋转有三个方向，分别是不转，正向转，反向转，delta rotation就是代表着这三个旋转矩阵
    // 而且第一个矩阵永远是固定的单位阵，因为是不旋转，只有后两个在变
    eigen_vector<Eigen::Matrix3d> delta_rotation(3);
    delta_rotation[0] = Eigen::Matrix3d::Identity();
    // 在X轴上旋转
    delta_rotation[1] = Eigen::Matrix3d(Eigen::AngleAxisd(rotation_step * M_PI / 180.f ,Eigen::Vector3d(1,0,0)));
    delta_rotation[2] = Eigen::Matrix3d(Eigen::AngleAxisd(- rotation_step * M_PI / 180.f ,Eigen::Vector3d(1,0,0)));
    for(int i = 0; i < perturbed1.size(); i++)
    {
        for(const Eigen::Matrix3d& delta_r : delta_rotation)
        {
            delta_T.block<3,3>(0,0) = delta_r;
            perturbed2.push_back(delta_T * perturbed1[i]);
        }
    }
    perturbed1.clear();
    // 在y轴上旋转
    delta_rotation[1] = Eigen::Matrix3d(Eigen::AngleAxisd(rotation_step * M_PI / 180.f ,Eigen::Vector3d(0,1,0)));
    delta_rotation[2] = Eigen::Matrix3d(Eigen::AngleAxisd(- rotation_step * M_PI / 180.f ,Eigen::Vector3d(0,1,0)));
 
    for(int i = 0; i < perturbed2.size(); i++)
    {
        for(const Eigen::Matrix3d& delta_r : delta_rotation)
        {
            delta_T.block<3,3>(0,0) = delta_r;
            perturbed1.push_back(delta_T * perturbed2[i]);
        }
    }
    perturbed2.clear();
    // 在z轴上旋转
    delta_rotation[1] = Eigen::Matrix3d(Eigen::AngleAxisd(rotation_step * M_PI / 180.f ,Eigen::Vector3d(0,0,1)));
    delta_rotation[2] = Eigen::Matrix3d(Eigen::AngleAxisd(- rotation_step * M_PI / 180.f ,Eigen::Vector3d(0,0,1)));
 
    for(int i = 0; i < perturbed1.size(); i++)
    {
        for(const Eigen::Matrix3d& delta_r : delta_rotation)
        {
            delta_T.block<3,3>(0,0) = delta_r;
            perturbed2.push_back(delta_T * perturbed1[i]);
        }
    }
    perturbed1.clear();

    // 同理，每个轴上的平移也只有三个方向，不动，正向，反向
    // 因此 delta_t 的第一个永远是[0,0,0]
    eigen_vector<Eigen::Vector3d> delta_translation(3);
    delta_translation[0] = Eigen::Vector3d::Zero();
    delta_T = Eigen::Matrix4d::Identity();
    // 在x轴上平移
    delta_translation[1] = Eigen::Vector3d(translation_step, 0, 0);
    delta_translation[2] = - delta_translation[1];
 
    for(int i = 0; i < perturbed2.size(); i++)
    {
        for(const Eigen::Vector3d& delta_t : delta_translation)
        {
            delta_T.block<3,1>(0,3) = delta_t;
            perturbed1.push_back(delta_T * perturbed2[i]);
        }
    }
    perturbed2.clear();
    // 在y轴上平移
    delta_translation[1] = Eigen::Vector3d(0, translation_step * 0.5, 0);
    delta_translation[2] = - delta_translation[1];
 
    for(int i = 0; i < perturbed1.size(); i++)
    {
        for(const Eigen::Vector3d& delta_t : delta_translation)
        {
            delta_T.block<3,1>(0,3) = delta_t;
            perturbed2.push_back(delta_T * perturbed1[i]);
        }
    }
    perturbed1.clear();
    // 在z轴上平移
    delta_translation[1] = Eigen::Vector3d(0, 0, translation_step);
    delta_translation[2] = - delta_translation[1];
 
    for(int i = 0; i < perturbed2.size(); i++)
    {
        for(const Eigen::Vector3d& delta_t : delta_translation)
        {
            delta_T.block<3,1>(0,3) = delta_t;
            perturbed1.push_back(delta_T * perturbed2[i]);
        }
    }
    assert(perturbed1.size() == 729);    // 3^6 = 729
    return perturbed1;
}

CameraLidarLineAssociate::~CameraLidarLineAssociate()
{
}
