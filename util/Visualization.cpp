/*
 * @Author: Diantao Tu
 * @Date: 2021-10-24 15:25:01
 */

#include "Visualization.h"

using namespace std;

//灰度图转为彩虹图:灰度值255~0分别对应：红、橙、黄、绿、青、蓝。
cv::Vec3b Gray2Color(uchar gray)
{
    cv::Vec3b pixel;
    // if (gray == 0)
    // {
    //     pixel[0] = 0;
    //     pixel[1] = 0;
    //     pixel[2] = 0;
    // }
     if (gray <= 51)
    {
        pixel[0] = 255;
        pixel[1] = gray * 5;
        pixel[2] = 0;
    }
    else if (gray <= 102)
    {
        gray -= 51;
        pixel[0] = 255 - gray * 5;
        pixel[1] = 255;
        pixel[2] = 0;
    }
    else if (gray <= 153)
    {
        gray -= 102;
        pixel[0] = 0;
        pixel[1] = 255;
        pixel[2] = gray * 5;
    }
    else if (gray <= 204)
    {
        gray -= 153;
        pixel[0] = 0;
        pixel[1] = 255 - static_cast<unsigned char>(128.0 * gray / 51.0 + 0.5);
        pixel[2] = 255;
    }
    else
    {
        gray -= 204;
        pixel[0] = 0;
        pixel[1] = 127 - static_cast<unsigned char>(127.0 * gray / 51.0 + 0.5);
        pixel[2] = 255;
    }
    return pixel;
}

cv::Vec3f BGR2HSV(const cv::Vec3b& bgr)
{
    const float r = bgr[2] / 255.f;
    const float g = bgr[1] / 255.f;
    const float b = bgr[0] / 255.f;
    const float C_max = max(r, max(g, b));
    const float C_min = min(r, min(g, b));
    // 最大的颜色分量是0，那就肯定是黑色，直接返回
    if(C_max == 0)
        return cv::Vec3f(0,0,0);
    const float delta_C = C_max - C_min;
    float h;
    if(C_max == r)
        h = 60.f * ((g - b) / delta_C + 6 * (g < b));
    else if(C_max == g)
        h = 60.f * ((b - r) / delta_C + 2);
    else if(C_max == b)
        h = 60.f * ((r - g) / delta_C + 4);
    float s = delta_C / C_max;
    float v = C_max;
    return cv::Vec3f(h / 360.f, s, v);
}

void DrawLine(cv::Mat& img, const cv::Vec4f& line, cv::Scalar color, int thickness, bool panoramic, int id )
{
    if(!panoramic)
    {
        cv::line(img, cv::Point2f(line[0], line[1]), cv::Point2f(line[2], line[3]), color, thickness);
        return;
    }

    Equirectangular eq(img.rows, img.cols);
    vector<cv::Point2f> segments = eq.BreakToSegments(line, 70);
    for(int i = 0; i < segments.size() - 1; i++)
    {
        if(abs(segments[i].x - segments[i+1].x) > 0.8 * img.cols)       
            continue;
        cv::line(img, segments[i], segments[i+1], color, thickness);
    }
    if(id >= 0)
        cv::putText(img, num2str(id), (segments[0] + segments[1]) / 2.0, CV_FONT_HERSHEY_PLAIN, 2.5, color, 3);
    return;
}

void DrawEachLine(const string& path, const cv::Mat& img, const vector<cv::Vec4f>& lines, 
                const cv::Scalar& color, const int thickness,  const bool panoramic)
{
    cv::Mat img_line_raw;
    if(img.channels() == 1)
    {
        vector<cv::Mat> images = {img, img, img};
        cv::merge(images, img_line_raw);
    }
    else 
        img_line_raw = img.clone();

    int img_count = 0;
    
    for(size_t i = 0; i < lines.size(); i++)
    {
        cv::Mat img_line = img_line_raw.clone();
        DrawLine(img_line, lines[i], color, thickness, panoramic);
        cv::imwrite(path + "/img_line_" + num2str(img_count) + ".jpg", img_line);
        img_count++;
    }
    return;
}

cv::Mat DrawLinesOnImage(const cv::Mat& img, const vector<cv::Vec4f>& lines, const vector<cv::Scalar>& colors, 
                        const int thickness, const bool panoramic, const bool draw_id, const vector<int>& ids)
{
    cv::Mat img_line;
    if(img.channels() == 1)
        cv::cvtColor(img, img_line, CV_GRAY2BGR);
    else 
        img_line = img.clone();
    
    for(size_t i = 0; i < lines.size(); i++)
        DrawLine(img_line, lines[i], colors[i % colors.size()], thickness, panoramic, draw_id ? ids[i] : -1);
    return img_line;
}

cv::Mat DrawLinePairsOnImage(const cv::Mat& img_gray, const vector<CameraLidarLinePair>& line_pairs, 
                            const Eigen::Matrix4d& T_cl, const int thickness, const bool draw_id)
{
    Equirectangular eq(img_gray.rows, img_gray.cols);
    vector<cv::Vec4f> img_lines, lidar_lines;
    vector<int> img_line_ids, lidar_line_ids;
    for(const CameraLidarLinePair& p : line_pairs)
    {
        img_lines.push_back(p.image_line);
        img_line_ids.push_back(p.image_line_id);
        Eigen::Vector3d p1 = (T_cl * p.lidar_line_start.homogeneous()).hnormalized();
        Eigen::Vector3d p2 = (T_cl * p.lidar_line_end.homogeneous()).hnormalized();
        Eigen::Vector2d p1_pixel = eq.CamToImage(p1);
        Eigen::Vector2d p2_pixel = eq.CamToImage(p2);
        lidar_lines.push_back(cv::Vec4f(p1_pixel.x(), p1_pixel.y(), p2_pixel.x(), p2_pixel.y()));
        lidar_line_ids.push_back(p.lidar_line_id);
    }
    cv::Mat img_lidar_line;
    // 先是分别画出图像直线和雷达直线
    img_lidar_line = DrawLinesOnImage(img_gray, img_lines, vector<cv::Scalar>(1, cv::Scalar(0,0,255)), thickness, true, draw_id, img_line_ids);
    img_lidar_line = DrawLinesOnImage(img_lidar_line, lidar_lines, vector<cv::Scalar>(1, cv::Scalar(255,0,0)), thickness, true, draw_id, lidar_line_ids);
    // 在相互匹配的直线之间画一条线连接
    int half_thickness = ceil(thickness / 2.0);
    for(size_t i = 0; i < line_pairs.size(); i++)
    {
        const cv::Vec4f& line1 = img_lines[i];
        const cv::Vec4f& line2 = lidar_lines[i];
        cv::Point2f line1_middle, line2_middle;
        // 直线长度太短，那就直接用端点的中点代替
        if(PointDistanceSquare(line1) < 100 * 100)
        {
            const cv::Point2f line1_start(line1[0], line1[1]);
            const cv::Point2f line1_end(line1[2], line1[3]);
            line1_middle = (line1_start + line1_end) / 2.0;
        }
        else 
        {
            vector<cv::Point2f> segments = eq.BreakToSegments(line1, 70);
            line1_middle = segments[segments.size()/2];
        }
        if(PointDistanceSquare(line2) < 100 * 100)
        {
            const cv::Point2f line2_start = cv::Point2f(line2[0], line2[1]);
            const cv::Point2f line2_end = cv::Point2f(line2[2], line2[3]);
            line2_middle = (line2_start + line2_end) / 2.0;
        }
        else 
        {
            vector<cv::Point2f> segments = eq.BreakToSegments(line2, 70);
            line2_middle = segments[segments.size()/2];
        }
        cv::line(img_lidar_line, line1_middle, line2_middle, cv::Scalar(0,255,0), half_thickness);
    }

    return img_lidar_line;
}

cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::Vec4f>& lines1,
                            const cv::Mat& img2, const std::vector<cv::Vec4f>& lines2,
                            const std::vector<cv::DMatch>& matches, const bool panoramic)
{
    cv::Mat out = cv::Mat::zeros(img1.rows + img2.rows, max(img1.cols, img2.cols), CV_8UC3);
    cv::Mat img1_line, img2_line;
    if(img1.channels() == 1) 
        cv::cvtColor(img1, img1_line, CV_GRAY2BGR);
    else 
        img1_line = img1.clone();

    if(img2.channels() == 1)  
        cv::cvtColor(img2, img2_line, CV_GRAY2BGR);
    else 
        img2_line = img2.clone();
    cv::Point2f offset(0, img1.rows);

    vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                cv::Scalar(255,0,255)}; 
    // 先在两张图像上分别画出直线，然后再竖直拼到一起
    for(size_t i = 0; i < matches.size(); i++)
    {
        const cv::DMatch& match = matches[i];
        const cv::Vec4f& line1 = lines1[match.queryIdx];
        const cv::Vec4f& line2 = lines2[match.trainIdx];
        DrawLine(img1_line, line1, colors[i % colors.size()], 5, panoramic);
        DrawLine(img2_line, line2, colors[i % colors.size()], 5, panoramic);
    }
    img1_line.copyTo(out.rowRange(0, img1.rows).colRange(0, img1.cols));
    img2_line.copyTo(out.rowRange(img1.rows, img1.rows + img2.rows).colRange(0, img2.cols));
    // 把匹配直线的中点连接起来，这里涉及到全景图像上曲线中点的计算，为了简化计算方式（毕竟只是为了可视化，看着差不多就行）
    // 如果直线比较短，那就用端点的中点代替即可。如果直线比较长，那么就先把直线分段，然后取出中间的那个点作为中点
    Equirectangular eq(img1.rows, img1.cols);
    for(size_t i = 0; i < matches.size(); i++)
    {
        const cv::DMatch& match = matches[i];
        const cv::Vec4f& line1 = lines1[match.queryIdx];
        const cv::Vec4f& line2 = lines2[match.trainIdx];
        cv::Point2f line1_middle, line2_middle;
        // 不是全景图像，或者直线长度太短，那就直接用端点的中点代替
        if(!panoramic || PointDistanceSquare(line1) < 100 * 100)
        {
            const cv::Point2f line1_start(line1[0], line1[1]);
            const cv::Point2f line1_end(line1[2], line1[3]);
            line1_middle = (line1_start + line1_end) / 2.0;
        }
        else 
        {
            vector<cv::Point2f> segments = eq.BreakToSegments(line1, 70);
            line1_middle = segments[segments.size()/2];
        }
        if(!panoramic || PointDistanceSquare(line2) < 100 * 100)
        {
            const cv::Point2f line2_start = cv::Point2f(line2[0], line2[1]) + offset;
            const cv::Point2f line2_end = cv::Point2f(line2[2], line2[3]) + offset;
            line2_middle = (line2_start + line2_end) / 2.0;
        }
        else 
        {
            vector<cv::Point2f> segments = eq.BreakToSegments(line2, 70);
            line2_middle = segments[segments.size()/2] + offset;
        }
        cv::line(out, line1_middle, line2_middle, colors[i % colors.size()], 3);   
    }
    return out;
}

cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::line_descriptor::KeyLine>& key_lines1,
                            const cv::Mat& img2, const std::vector<cv::line_descriptor::KeyLine>& key_lines2,
                            const std::vector<cv::DMatch>& matches, const bool panoramic)
{
    vector<cv::Vec4f> lines1, lines2;
    for(const cv::line_descriptor::KeyLine& line : key_lines1)
        lines1.push_back(cv::Vec4f(line.startPointX, line.startPointY, line.endPointX, line.endPointY));
    for(const cv::line_descriptor::KeyLine& line : key_lines2)
        lines2.push_back(cv::Vec4f(line.startPointX, line.startPointY, line.endPointX, line.endPointY));
    return DrawMatchesVertical(img1, lines1, img2, lines2, matches, panoramic);
}
                        

cv::Mat DepthImageRGB(const cv::Mat& depth_map, const float max_depth, const float min_depth)
{
    assert(max_depth >= min_depth);
    cv::Mat depth_rgb = cv::Mat::zeros(depth_map.size(), CV_8UC3);
    cv::Mat depth_raw;
    if(depth_map.type() ==  CV_16U)
    {
        depth_map.convertTo(depth_raw, CV_32F);
        depth_raw /= 256.0;
    }
    else if(depth_map.type() == CV_32F)
        depth_raw = depth_map.clone();
    else 
    {
        cout << "only support CV_32F or CV_16U" << endl;
        return cv::Mat();
    }
    double max_d, min_d;
    // 找到深度图里的最大值和最小值
    cv::minMaxLoc(depth_raw, &min_d, &max_d, new cv::Point2i(), new cv::Point2i());
    max_d = max_depth < 0 ? max_d : max_depth;
    min_d = min_depth < 0 ? min_d : min_depth;
    float depth_range = static_cast<float>(max_d - min_d);
    for(size_t i = 0; i < depth_map.rows; i++)
    {
        for(size_t j = 0; j < depth_map.cols; j++)
        {
            float real_depth = depth_raw.at<float>(i,j);
            if(real_depth == 0)
                depth_rgb.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            else if(real_depth > max_d)
                depth_rgb.at<cv::Vec3b>(i,j) = Gray2Color(255);
            else if(real_depth < min_d)
                depth_rgb.at<cv::Vec3b>(i,j) = Gray2Color(0);
            else 
                depth_rgb.at<cv::Vec3b>(i,j) = Gray2Color(static_cast<uchar>((real_depth - min_d) / depth_range * 255));
        }
    }
    return depth_rgb;
}


cv::Mat DepthImageGray(const cv::Mat& depth_map, const float max_depth, const float min_depth)
{
    assert(max_depth >= min_depth);
    cv::Mat depth_gray = cv::Mat::zeros(depth_map.size(), CV_8U);
    cv::Mat depth_raw;
    if(depth_map.type() ==  CV_16U)
    {
        depth_map.convertTo(depth_raw, CV_32F);
        depth_raw /= 256.0;
    }
    else if(depth_map.type() == CV_32F)
        depth_raw = depth_map.clone();
    else 
    {
        cout << "only support CV_32F or CV_16U" << endl;
        return cv::Mat();
    }
    double max_d, min_d;
    // 找到深度图里的最大值和最小值
    cv::minMaxLoc(depth_raw, &min_d, &max_d, new cv::Point2i(), new cv::Point2i());
    max_d = max_depth < 0 ? max_d : max_depth;
    min_d = min_depth < 0 ? min_d : min_depth;
    float depth_range = static_cast<float>(max_d - min_d);
    for(size_t i = 0; i < depth_map.rows; i++)
    {
        for(size_t j = 0; j < depth_map.cols; j++)
        {
            float real_depth = depth_raw.at<float>(i,j);
            depth_gray.at<uchar>(i,j) = static_cast<uchar>((real_depth - min_d) / depth_range * 255);
        }
    }
    return depth_gray;
}

cv::Mat DepthImageGray16(const cv::Mat& depth_map)
{
    // assert(max_depth >= min_depth);
    cv::Mat depth_gray = cv::Mat::zeros(depth_map.size(), CV_16U);
    cv::Mat depth_raw;
    if(depth_map.type() ==  CV_16U)
        return depth_map;
    else if(depth_map.type() == CV_32F)
        depth_raw = depth_map.clone();
    else 
    {
        cout << "only support CV_32F or CV_16U" << endl;
        return cv::Mat();
    }

    for(size_t i = 0; i < depth_map.rows; i++)
    {
        for(size_t j = 0; j < depth_map.cols; j++)
        {
            float relative_depth = depth_raw.at<float>(i,j) * 256.f;
            relative_depth = relative_depth > 65535 ? 0 : relative_depth;
            depth_gray.at<ushort>(i,j) = static_cast<ushort>(relative_depth);
        }
    }
    return depth_gray;
}

void SaveDepthImageRaw(const cv::Mat& depth_image, const string file_path)
{
    cv::Mat depth_16;
    if(depth_image.type() ==  CV_16U)
    {
        depth_16 = depth_image.clone();
    }
    else if(depth_image.type() == CV_32F)
    {
        cv::Mat tmp = depth_image * 256.0;
        tmp.convertTo(depth_16, CV_16U);
    }
    else 
    {
        cout << "only support CV_32F or CV_16U" << endl;
        return ;
    }
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0); // 无压缩png
    cv::imwrite(file_path, depth_16, compression_params);
    return ;
}

bool CameraCenterPCD(const string& file_name,
                    const eigen_vector<Eigen::Vector3d>& t_wc_list)
{
    pcl::PointCloud<pcl::PointXYZI> cloud_center;
    for(size_t i = 0; i < t_wc_list.size(); i++)
    {
        pcl::PointXYZI point;
        point.x = t_wc_list[i].x();
        point.y = t_wc_list[i].y();
        point.z = t_wc_list[i].z();
        point.intensity = i;
        if(isinf(point.x) || isinf(point.y) || isinf(point.z))
            continue;
        if(isnan(point.x) || isnan(point.y) || isnan(point.z))
            continue;
        cloud_center.push_back(point);
    }
    pcl::io::savePCDFileASCII(file_name, cloud_center);
    return true;
}


cv::Mat CombineDepthWithRGB(const cv::Mat& depth_image, const cv::Mat& rgb_image, const float max_depth, const float min_depth)
{
    assert(depth_image.rows == rgb_image.rows && depth_image.cols == rgb_image.cols);
    cv::Mat depth_color = DepthImageRGB(depth_image, max_depth, min_depth);
    cv::Mat combined = rgb_image.clone();
    for(int i = 0; i < depth_image.rows; i++)
        for(int j = 0; j < depth_image.cols; j++)
        {
            if(depth_color.at<cv::Vec3b>(i,j) != cv::Vec3b::zeros())
                combined.at<cv::Vec3b>(i,j) = depth_color.at<cv::Vec3b>(i,j);
        }
    return combined;
}

cv::Mat DrawKeyPoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const vector<cv::Scalar>& colors)
{
    cv::Mat out;
    if(img.channels() == 1)
        cv::cvtColor(img, out, CV_GRAY2BGR);
    else if(img.channels() == 3)
        out = img.clone();
    else 
    {
        LOG(ERROR) << "only support gray image or RGB image" ;
        return cv::Mat();
    }
    for(int i = 0; i < keypoints.size(); i++)
    {
        cv::Vec3b color;
        if(!colors.empty())
        {
            color[0] = colors[i % colors.size()][0];
            color[1] = colors[i % colors.size()][1];
            color[2] = colors[i % colors.size()][2];
        }
        else 
            color = Gray2Color(static_cast<uchar>(i % 256));
        cv::circle(out, keypoints[i].pt, 10, color, 3);
    }
    return out;
}

cv::Mat DrawMatchesVertical(const cv::Mat& img1, const std::vector<cv::KeyPoint> keypoints1,
                            const cv::Mat& img2, const std::vector<cv::KeyPoint> keypoints2,
                            const std::vector<cv::DMatch>& matches, const std::vector<size_t>& inlier_idx)
{
    cv::Mat out = cv::Mat::zeros(img1.rows + img2.rows, max(img1.cols, img2.cols), CV_8UC3);
    if(img1.channels() == 1)
    {
        cv::Mat tmp;
        cv::cvtColor(img1, tmp, CV_GRAY2BGR);
        tmp.copyTo(out.rowRange(0, img1.rows).colRange(0, img1.cols));
    }
    else 
        img1.copyTo(out.rowRange(0, img1.rows).colRange(0, img1.cols));
    if(img2.channels() == 1)
    {
        cv::Mat tmp;
        cv::cvtColor(img2, tmp, CV_GRAY2BGR);
        tmp.copyTo(out.rowRange(img1.rows, img1.rows + img2.rows).colRange(0, img2.cols));
    }
    else 
        img2.copyTo(out.rowRange(img1.rows, img1.rows + img2.rows).colRange(0, img2.cols));
    cv::Point2f offset(0, img1.rows);
    vector<size_t> valid_ids;
    if(!inlier_idx.empty())
        valid_ids = inlier_idx; 
    else 
    {
        valid_ids.resize(matches.size());
        std::iota(valid_ids.begin(), valid_ids.end(), 0);
    }
    for(const size_t& i : valid_ids)
    {
        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;
        cv::Vec3b color = Gray2Color(static_cast<uchar>(idx1 % 256));
        cv::circle(out, keypoints1[idx1].pt, 10, color, 3);
        cv::circle(out, keypoints2[idx2].pt + offset, 10, color, 3);
        cv::line(out, keypoints1[idx1].pt, keypoints2[idx2].pt + offset, color, 3);
    }
    return out;
}


bool CameraPoseVisualize(const string& plyfile, const eigen_vector<Eigen::Matrix3d>& R_wc_list, 
                        const eigen_vector<Eigen::Vector3d>& t_wc_list, const int main_axis)
{
    if(R_wc_list.size() == 0){
        LOG(ERROR) << "no camera rotation" << endl;
        return false;
    }
    assert(R_wc_list.size() == t_wc_list.size());
    // 在ply文件里，每个camera用一个四棱锥表示，也就是用5个顶点
    eigen_vector<eigen_vector<Eigen::Vector3d>> cameras;
    float size = 0.1;
    int other_axis1 = (main_axis + 1) % 3, other_axis2 = (main_axis + 2) % 3;
    for(int i = 0; i < R_wc_list.size(); i++)
    { 
        const Eigen::Matrix3d& R_wc = R_wc_list[i];
        const Eigen::Vector3d& t_wc = t_wc_list[i];
        if(R_wc.isZero())
            continue;
        if(isinf(t_wc.x()) || isinf(t_wc.y()) || isinf(t_wc.z()))
            continue;
        if(isnan(t_wc.x()) || isnan(t_wc.y()) || isnan(t_wc.z()))
            continue;

        eigen_vector<Eigen::Vector3d> vertex;   

        Eigen::Vector3d v1(size, 2 * size, size);
        Eigen::Vector3d v2(size, 2 * size, -size);
        Eigen::Vector3d v3(-size, 2 * size, size);
        Eigen::Vector3d v4(-size, 2 * size, -size);

        v1(main_axis) = 2 * size;
        v2(main_axis) = 2 * size;
        v3(main_axis) = 2 * size;
        v4(main_axis) = 2 * size;

        v1(other_axis1) = size;
        v1(other_axis2) = size;
        v2(other_axis1) = size;
        v2(other_axis2) = -size;
        v3(other_axis1) = -size;
        v3(other_axis2) = size;
        v4(other_axis1) = -size;
        v4(other_axis2) = -size;

        vertex.push_back(t_wc);
        vertex.push_back(R_wc * v1 + t_wc);
        vertex.push_back(R_wc * v2 + t_wc);
        vertex.push_back(R_wc * v3 + t_wc);
        vertex.push_back(R_wc * v4 + t_wc);
        
        cameras.push_back(vertex);
    }

    FILE *fp;
    fp = fopen(plyfile.c_str(), "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %ld\n", cameras.size() * cameras[0].size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "element edge %ld\n", cameras.size() * 8);
    fprintf(fp, "property int vertex1\n");
    fprintf(fp, "property int vertex2\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");
    for(int i = 0; i < cameras.size(); i++)
    {
        fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][0].x(),cameras[i][0].y(),cameras[i][0].z(), 255, 0, 0);
        for (int j = 1; j < cameras[i].size(); j++)
        {
            fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][j].x(),cameras[i][j].y(),cameras[i][j].z(), 0, 0, 255);
        }
    }
    for(int i = 0; i < cameras.size(); i++)
    {
        cv::Vec3b bgr = Gray2Color(static_cast<uchar>(i % 256));
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 1, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 2, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 3, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 4, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size() + 1, i*cameras[i].size() + 2, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size() + 2, i*cameras[i].size() + 4, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size() + 4, i*cameras[i].size() + 3, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size() + 3, i*cameras[i].size() + 1, bgr[0], bgr[1], bgr[2]);
    }
    fclose(fp);
    return true;
}

bool CameraPoseVisualizeCoord(const string& plyfile, const eigen_vector<Eigen::Matrix3d>& R_wc_list, 
                        const eigen_vector<Eigen::Vector3d>& t_wc_list)
{
    if(R_wc_list.empty())
    {
        LOG(ERROR) << "no camera rotation" << endl;
        return false;
    }
    assert(R_wc_list.size() == t_wc_list.size());
    // 每一个相机位姿用一个小的坐标系表示，有四个点，原点+三个坐标轴上的端点
    eigen_vector<eigen_vector<Eigen::Vector3d>> cameras;
    float size = 0.1;

    for(int i = 0; i < R_wc_list.size(); i++)
    {
        eigen_vector<Eigen::Vector3d> vertex;
        Eigen::Matrix3d R_wc = R_wc_list[i];
        Eigen::Vector3d t_wc = t_wc_list[i];

        Eigen::Vector3d v1(size, 0, 0);
        Eigen::Vector3d v2(0, size, 0);
        Eigen::Vector3d v3(0, 0, size);

        vertex.push_back(t_wc);
        vertex.push_back(R_wc * v1 + t_wc);
        vertex.push_back(R_wc * v2 + t_wc);
        vertex.push_back(R_wc * v3 + t_wc);
        
        cameras.push_back(vertex);
    }
    FILE *fp;
    fp = fopen(plyfile.c_str(), "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %ld\n", cameras.size() * cameras[0].size());
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "element edge %ld\n", cameras.size() * 3);
    fprintf(fp, "property int vertex1\n");
    fprintf(fp, "property int vertex2\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");
    for(int i = 0; i < cameras.size(); i++)
    {
        fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][0].x(),cameras[i][0].y(),cameras[i][0].z(), 255, 255, 255);
        fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][1].x(),cameras[i][1].y(),cameras[i][1].z(), 255, 0, 0);
        fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][2].x(),cameras[i][2].y(),cameras[i][2].z(), 0, 255, 0);
        fprintf(fp, "%f %f %f %d %d %d\n",cameras[i][3].x(),cameras[i][3].y(),cameras[i][3].z(), 0, 0, 255);
    }
    for(int i = 0; i < cameras.size(); i++)
    {
        cv::Vec3b bgr = Gray2Color(static_cast<unsigned char>(i % 256));
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 1, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 2, bgr[0], bgr[1], bgr[2]);
        fprintf(fp, "%ld %ld %d %d %d\n",i*cameras[i].size(), i*cameras[i].size() + 3, bgr[0], bgr[1], bgr[2]);
    }
    fclose(fp);
    return true;
}

cv::Mat DrawNormalImage(const cv::Mat& normal_image, bool normalized)
{
    cv::Mat img_color = cv::Mat::zeros(normal_image.size(), CV_8UC3);
    cv::MatConstIterator_<cv::Vec3f> it_normal = normal_image.begin<cv::Vec3f>();
    cv::MatIterator_<cv::Vec3b> it_color = img_color.begin<cv::Vec3b>();
    for( ;it_normal != normal_image.end<cv::Vec3f>() && it_color != img_color.end<cv::Vec3b>(); it_normal++, it_color++)
    {
        if((*it_normal) == cv::Vec3f::zeros())
            continue;
        cv::Vec3f normal = *it_normal;
        if(!normalized)
            normal /= cv::norm(normal);
        int b = normal[0] * 128 + 128;
        b -= (b > 255);
        int g = normal[1] * 128 + 128;
        g -= (g > 255);
        int r = normal[2] * 128 + 128;
        r -= (r > 255);
        // uchar b = min(max(static_cast<uchar>(round((1.f - normal(0))) * 127.5f ), uchar(0)), uchar(255));
        // uchar g = min(max(static_cast<uchar>(round((1.f - normal(1))) * 127.5f ), uchar(0)), uchar(255));
        // uchar r = min(max(static_cast<uchar>(round(-normal(2) * 255.f)), uchar(0)), uchar(255));
        *it_color = cv::Vec3b(b,g,r);
    }
    return img_color;
}