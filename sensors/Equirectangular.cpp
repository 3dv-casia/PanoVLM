/*
 * @Author: Diantao Tu
 * @Date: 2021-10-30 12:03:39
 */


#include "Equirectangular.h"


using namespace std;

void Equirectangular::PreComputeI2C()
{
    image_to_unit_sphere = cv::Mat::zeros(rows, cols, CV_32FC3);
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++)
            image_to_unit_sphere.at<cv::Vec3f>(i,j) = ImageToCam(cv::Point2f(j, i), 1.f);
}

std::vector<cv::Point2f> Equirectangular::BreakToSegments(const cv::Point2f start, const cv::Point2f end, const float seg_length)
{
    // 把起始点和终止点都变换成单位圆上的XYZ坐标
    cv::Point3f p1 = ImageToCam((start), float(5.0));
    cv::Point3f p2 = ImageToCam((end), float(5.0));
    cv::Point3f sphere_line = p2 - p1;
    // 把提取出的直线按seg_length像素一段进行粗略划分，每一段结束后就要有一个点
    float length = sqrt((start.x - end.x)*(start.x - end.x) + (start.y - end.y)*(start.y - end.y));
    int count = length / seg_length + 1;
    vector<cv::Point2f> segments = {start};
    for(int i = 1; i < count; i++)
    {
        cv::Point3f p = p1 + i * 1.f / count * sphere_line;
        cv::Point2f pixel = CamToImage(p);
        // 如果连续的两个点之间的u轴坐标的差异特别大，说明当前直线跨过了球体的YZ平面，从图像的左边一下边到了右边，
        // 或者从右边变到了左边，那么此时就要处理一下，不能相连
        // 方法就是找到线段在图像边缘处的两个点，分别为 left 和 right，这两个点之间互相不连接，其他的照样连上
        if(abs(pixel.x - segments[segments.size() - 1].x) > 0.8 * cols)
        {
            p = p1 + p1.x / (p1.x - p2.x) * sphere_line;    // 让p的x坐标恰好为0，这样p就会处于分界线上
            cv::Point2f left = CamToImage(p);
            left.x = 0;
            cv::Point2f right(cols - 1, left.y);
            if(pixel.x > segments[segments.size() - 1].x)
            {
                segments.push_back(left);
                segments.push_back(right);
            }
            else 
            {
                segments.push_back(right);
                segments.push_back(left);
            }
        }
        segments.push_back(pixel);
    }
    segments.push_back(end);
    return segments;
}

std::vector<cv::Point2f> Equirectangular::BreakToSegments(const cv::Vec4f line, const float length)
{
    cv::Point2f p1 = cv::Point2f(line[0], line[1]);
    cv::Point2f p2 = cv::Point2f(line[2], line[3]);
    return BreakToSegments(p1, p2, length);
}
