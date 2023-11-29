/*
 * @Author: Diantao Tu
 * @Date: 2021-10-30 11:40:20
 */

#define USE_FAST_ATAN2

#ifndef EQUIRECTANGULAR_H
#define EQUIRECTANGULAR_H

#include <opencv2/core.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../base/Math.h"

class Equirectangular
{
private:
    int cols, rows;
    cv::Mat image_to_unit_sphere;
public:
    Equirectangular(int _rows, int _cols):cols(_cols), rows(_rows){};

    void PreComputeI2C();

    // 一条空间中的直线映射到球面模型上会成为一条曲线，这个曲线有起点和终点，
    // 但在图像上曲线不好表示，因此把它分段用直线来表示， 可以用于画图以及LiDAR和图像上直线的匹配
    // length -- 每一段直线的长度，像素为单位
    std::vector<cv::Point2f> BreakToSegments(const cv::Point2f start, const cv::Point2f end, const float length);

    // 和上一个函数一样，只是输入的参数变了
    std::vector<cv::Point2f> BreakToSegments(const cv::Vec4f line, const float length);

    /**
     * @description: 点从相机坐标系变换到球坐标系
     * @param T float double
     * @param point {Point3f} 输入的点坐标 
     * @return {Point2f} 球坐标系下的两个角 \theta \phi , 或者称为 longitude 和 latitude
     */
    template <typename T>
    inline cv::Point_<T> CamToSphere(const cv::Point3_<T>& point) const
    {
        // 这个变换公式是 X-右  Y-前   Z-上
        // angle.x = atan2(p[0], p[1]);    // \theta  longitude
        // angle.y = asin(p[2]);           // \phi    latitude
        // 这个变换公式是 X-右  Y-下   Z-前
        // angle.x = atan2(point.x, point.z);    // longtitude
        // angle.y = -asin(point.y);          // latitude
        #ifdef USE_FAST_ATAN2
        return cv::Point_<T>(FastAtan2(point.x, point.z), -FastAtan2(point.y, (T)sqrt(Square(point.x) + Square(point.z))));
        #else 
        return cv::Point_<T>(atan2(point.x, point.z), -atan2(point.y, (T)sqrt(Square(point.x) + Square(point.z))));
        #endif
    }

    // 同样的变换，只是支持了Eigen
    template <typename T>
    inline Eigen::Matrix<T, 2, 1> CamToSphere(const Eigen::Matrix<T, 3, 1>& point) const
    {
        // 这个变换公式是 X-右  Y-前   Z-上
        // angle.x = atan2(p[0], p[1]);    // \theta  longitude
        // angle.y = asin(p[2]);           // \phi    latitude
        // 这个变换公式是 X-右  Y-下   Z-前
        // angle(0) = atan2(p[0], p[2]);    // longtitude
        // angle(1) = -asin(p[1]);          // latitude
        #ifdef USE_FAST_ATAN2 
        return Eigen::Matrix<T, 2, 1>(FastAtan2(point[0], point[2]), -FastAtan2(point[1], (T)sqrt(Square(point[0]) + Square(point[2]))));
        #else 
        return Eigen::Matrix<T, 2, 1>(atan2(point[0], point[2]), -atan2(point[1], (T)sqrt(Square(point[0]) + Square(point[2]))));
        #endif
    }

    /**
     * @description: 从球坐标系变换到图像坐标系，图像坐标系的原点在左上角
     * @param T float double
     * @param sphere {Point2f} 球坐标系下的点
     * @return {Point2f} 图像坐标系的点
     */
    template <typename T>
    inline cv::Point_<T> SphereToImage(const cv::Point_<T>& sphere) const
    {
        cv::Point_<T> pixel;
        pixel.x = cols * (0.5 + sphere.x / (2.0 * M_PI));
        pixel.y = rows * (0.5 - sphere.y / M_PI);
        return pixel;
    }

    template <typename T>
    inline Eigen::Matrix<T, 2, 1> SphereToImage(const Eigen::Matrix<T, 2, 1>& sphere) const
    {
        Eigen::Matrix<T, 2, 1> pixel;
        pixel(0) = cols * (0.5 + sphere(0) / (2.0 * M_PI));
        pixel(1) = rows * (0.5 - sphere(1) / M_PI);
        return pixel;
    }

    template <typename T>
    inline cv::Point_<T> ImageToSphere(const cv::Point_<T>& image) const
    {
        cv::Point_<T> sphere;
        sphere.x = (2 * image.x / cols - 1) * M_PI;
        sphere.y = (0.5 - image.y / rows) * M_PI;
        return sphere;
    }

    template <typename T>
    inline Eigen::Matrix<T, 2, 1> ImageToSphere(const Eigen::Matrix<T, 2, 1>& image) const
    {
        Eigen::Matrix<T, 2, 1> sphere;
        sphere(0) = (2 * image(0) / cols - 1) * M_PI;
        sphere(1) = (0.5 - image(1) / rows) * M_PI;
        return sphere;
    }

    template <typename T>
    inline cv::Point3_<T> SphereToCam(const cv::Point_<T>& sphere, const T r) const
    {
        cv::Point3_<T> cam;
        // 这个变换公式是 X-右  Y-前   Z-上
        // cam.x = r * cos(sphere.y) * sin(sphere.x);
        // cam.y = r * cos(sphere.y) * cos(sphere.x);
        // cam.z = r * sin(sphere.y);
        // 这个变换公式是 X-右  Y-下   Z-前
        T cy = cos(sphere.y);
        cam.x = r * cy * sin(sphere.x);
        cam.y = - r * sin(sphere.y);
        cam.z = r * cy * cos(sphere.x);
        return cam;
    }

    template <typename T>
    inline Eigen::Matrix<T, 3, 1> SphereToCam(const Eigen::Matrix<T, 2, 1>& sphere, const T r = 1.0) const
    {
        Eigen::Matrix<T, 3, 1> cam;
        // 这个变换公式是 X-右  Y-前   Z-上
        // cam.x = r * cos(sphere.y) * sin(sphere.x);
        // cam.y = r * cos(sphere.y) * cos(sphere.x);
        // cam.z = r * sin(sphere.y);
        // 这个变换公式是 X-右  Y-下   Z-前
        T cy = cos(sphere(1));
        cam(0) = r * cy * sin(sphere(0));
        cam(1) = - r * sin(sphere(1));
        cam(2) = r * cy * cos(sphere(0));
        return cam;
    }

    template <typename T>
    inline cv::Point3_<T> ImageToCam(const cv::Point_<T>& pixel, const T r) const
    {
        return SphereToCam(ImageToSphere(pixel), r);
    }

    // 这是专门为 cv::Point2i 优化过，因为图像上的像素数量是有限的，可以提前先计算好每个
    // 二维像素点到三维空间点之间的对应关系，然后直接查表即可
    inline cv::Point3f ImageToCam(const cv::Point2i& pixel) const
    {
        if(!image_to_unit_sphere.empty())
        {
            return cv::Point3f(image_to_unit_sphere.at<cv::Vec3f>(pixel));
        }
        else 
            return ImageToCam(cv::Point2f(pixel), 1.f);
    }

    template <typename T>
    inline Eigen::Matrix<T, 3, 1> ImageToCam(const Eigen::Matrix<T, 2, 1>& pixel, const T r = 1.0) const
    {
        return SphereToCam(ImageToSphere(pixel), r);
    }

    template <typename T>
    inline cv::Point_<T> CamToImage(const cv::Point3_<T>& cam) const
    {
        return SphereToImage(CamToSphere(cam));
    }

    template <typename T>
    inline Eigen::Matrix<T, 2, 1> CamToImage(const Eigen::Matrix<T, 3, 1> cam) const
    {
        return SphereToImage(CamToSphere(cam));
    }

    inline bool IsInside(const cv::Point2i& pt) const
    {
        return pt.x >= 0 && pt.y >= 0 && pt.x + 1 <= cols && pt.y + 1 <= rows;
    }

    inline bool IsInside(const cv::Point2f& pt) const
    {
        return pt.x >= 0 && pt.y >= 0 && pt.x < cols && pt.y < rows;
    }

    template<typename T>
    inline bool IsInside(const cv::Point_<T>& pt, int row_marin, int col_margin)
    {
        return pt.x >= col_margin && pt.y >= row_marin && pt.x < cols - col_margin && pt.y < rows - row_marin;
    }

    template<typename T>
    inline bool IsInside(const Eigen::Matrix<T, 2, 1>& pt) const 
    {
        return pt.x() >= 0 && pt.y() >= 0 && pt.x() < cols && pt.y() < rows;
    }

    ~Equirectangular(){};
};


#endif