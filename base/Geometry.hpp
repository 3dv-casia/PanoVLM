/*
 * @Author: Diantao Tu
 * @Date: 2021-11-10 20:33:55
 */

#ifndef _GEOMETRY_H
#define _GEOMETRY_H

#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <glog/logging.h>
#include "common.h"
#include "Math.h"

// 二维两点距离的平方
template<typename T>
inline T PointDistanceSquare(const T* p1, const T* p2)
{
    return Square(p1[0] - p2[0]) + Square(p1[1] - p2[1]) ;
}

template<typename T>
inline T PointDistanceSquare(const cv::Point_<T>& p1, const cv::Point_<T>& p2){
    return Square(p1.x - p2.x) + Square(p1.y - p2.y);
}

template<typename T>
inline T PointDistanceSquare(const cv::Vec<T,4>& line)
{
    return Square(line[0] - line[2]) + Square(line[1] - line[3]) ;
}

inline float PointDistanceSquare(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2){
    return Square(p1.x - p2.x) + Square(p1.y - p2.y) + Square(p1.z - p2.z);
}

/**
 * @description: 用点斜式表示一条直线
 * @param T cv::poin2f  cv::point2d
 * @param sp 直线的起点
 * @param ep 直线的终点
 * @return 直线的表达式 k,b, 如果返回的b=infinite，那么就说明这条直线垂直于x轴
 */
template<typename T>
inline T FormLine(const T& sp, const T& ep )
{
    if(abs(sp.x - ep.x) < 1e-2)
        return T(sp.x, std::numeric_limits<float>::infinity());
    float k = (sp.y - ep.y)/(sp.x - ep.x);
    float b = sp.y - k * sp.x;
    return T(k, b);
}

// 和上面的一样，只是用一个cv::vec 来存储起点和终点，其中前两维是起点，后两维是终点
template<typename T>
inline cv::Point_<T> FormLine(const cv::Vec<T,4>& line)
{
    if(abs(line[0] - line[2]) < 1e-2)
        return cv::Point_<T>(line[0], std::numeric_limits<T>::infinity());
    T k = (line[1] - line[3])/(line[0] - line[2]);
    T b = line[1] - k * line[0];
    return cv::Point_<T>(k, b);
}


/**
 * @description: 投影二维点到直线上
 * @param T 可以是 cv::point2f  cv::point2d
 * @param sp 直线的起始点
 * @param ep 直线的终止点
 * @param point 要投影的二维点
 * @return 投影点
 */
template<typename T>
inline T ProjectPoint2Line2D(const T& sp, const T& ep, const T& point)
{
    if(abs(sp.x - ep.x) < 1e-2)
        return T(sp.x, point.y);
    float k = (sp.y - ep.y)/(sp.x - ep.x);
    float b = sp.y - k * sp.x;
    float x_proj = (k * (point.y - b) + point.x) / (k * k + 1);
    return T(x_proj, k * x_proj + b);
}

// 和上面的函数一样，只是这里的直线用cv::vec来表示，其中前两维是起点，后两维是终点
template<typename T>
inline cv::Point_<T> ProjectPoint2Line2D(const cv::Vec<T,4>& line, const cv::Point_<T>& point)
{
    if(abs(line[0] - line[2]) < 1e-2)
        return cv::Point_<T>(line[0], point.y);
    T k = (line[1] - line[3])/(line[0] - line[2]);
    T b = line[1] - k * line[0];
    T x_proj = (k * (point.y - b) + point.x) / (k * k + 1);
    return cv::Point_<T>(x_proj, k * x_proj + b);
}

/**
 * @description: 点到二维直线距离
 * @param line 直线，用起点和终点表示
 * @param point 点
 * @return 点到直线距离的平方
 */
template<typename T>
inline T PointToLineDistance2DSquare(const cv::Vec<T,4>& line, const cv::Point_<T>& point)
{
    cv::Point_<T> point_projected = ProjectPoint2Line2D(line, point);
    return PointDistanceSquare(point, point_projected);
}

/**
 * @description: 计算多个点到同一条直线的二维距离
 * @param line 直线，用起点和终点表示
 * @param points 多个点
 * @return 每个点到直线距离的平方
 */
template<typename T>
inline std::vector<T> PointToLineDistance2DSquare(const cv::Vec<T,4>& line, const std::vector<cv::Point_<T>>& points)
{
    vector<T> sq_dist;
    if(abs(line[0] - line[2]) < 1e-2)
    {
        for(const cv::Point_<T> p : points)
            sq_dist.push_back((p.x - line[0]) * (p.x - line[0]));
    }
    else 
    {
        T k = (line[1] - line[3])/(line[0] - line[2]);
        T b = line[1] - k * line[0];
        for(const cv::Point_<T> p : points)
        {
            T x_proj = (k * (p.y - b) + p.x) / (k * k + 1);
            sq_dist.push_back(PointDistanceSquare(p, cv::Point_<T>(x_proj, k * x_proj + b)));
        }
    }
    return sq_dist;
}

/**
 * @description: 投影三维点到直线上
 * @param T 可以是float double
 * @param point  三维点的坐标
 * @param line  直线的表示形式 a,b,c,d,e,f   (x-a)/d=(y-b)/e=(z-c)/f 
 * @return 投影点的坐标
 */  
template<typename T>
inline Eigen::Matrix<T, 3, 1> ProjectPoint2Line3D(const Eigen::Matrix<T, 3, 1>& point, const pcl::ModelCoefficients line)
{
    T x0 = line.values[0];
    T y0 = line.values[1];
    T z0 = line.values[2];
    T nx = line.values[3];
    T ny = line.values[4];
    T nz = line.values[5];
    T k = (nx * (point.x() - x0) + ny * (point.y() - y0) + nz * (point.z() - z0)) / (Square(nx)  + Square(ny)  + Square(nz));
    return Eigen::Matrix<T, 3, 1>(k * nx + x0, k * ny + y0, k * nz + z0);
}

// 和上面的方法一样，只是参数不同罢了
// T - float double 
template<typename T>
inline cv::Point3_<T> ProjectPoint2Line3D(const cv::Point3_<T>& point, const double* line)
{
    float x0 = line[0];
    float y0 = line[1];
    float z0 = line[2];
    float nx = line[3];
    float ny = line[4];
    float nz = line[5];
    float k = (nx * (point.x - x0) + ny * (point.y - y0) + nz * (point.z - z0)) / (Square(nx)  + Square(ny)  + Square(nz));
    return cv::Point3_<T>(k * nx + x0, k * ny + y0, k * nz + z0);
}

template<typename T>
inline Eigen::Matrix<T, 3, 1> ProjectPoint2Line3D(const Eigen::Matrix<T, 3, 1>& point, const double* line)
{
    T x0 = line[0];
    T y0 = line[1];
    T z0 = line[2];
    T nx = line[3];
    T ny = line[4];
    T nz = line[5];
    T k = (nx * (point(0) - x0) + ny * (point(1) - y0) + nz * (point(2) - z0)) / (Square(nx)  + Square(ny)  + Square(nz));
    return Eigen::Matrix<T, 3, 1>(k * nx + x0, k * ny + y0, k * nz + z0);
}

/**
 * @description: 计算三维点到空间直线的距离，就是先把点投影到直线上，然后计算点和投影点的距离
 * @param point 三维点
 * @param line 空间直线的表达，前三个数是经过的点，后三个数是方向向量（不需要是单位向量）
 * @param T double float
 * @return 点到直线距离
 */
template<typename T>
inline T PointToLineDistance3D(const T* point, const T* line)
{
    T x0 = line[0];
    T y0 = line[1];
    T z0 = line[2];
    T nx = line[3];
    T ny = line[4];
    T nz = line[5];
    T k = (nx * (point[0] - x0) + ny * (point[1] - y0) + nz * (point[2] - z0)) / (Square(nx)  + Square(ny)  + Square(nz));
    T project_point[3] = {k * nx + x0, k * ny + y0, k * nz + z0};
    T distance = sqrt(Square(project_point[0] - point[0]) + Square(project_point[1] - point[1]) + Square(project_point[2] - point[2]));
    return distance;
}

/**
 * @description: 判断一系列三维点是否能形成一条空间直线，根据的是这些点的PCA方向，详情见 公式推导.md
 * @param points 三维点
 * @param tolerance 这个值越大，对直线的要求就越严苛，也就是这些点要更“直”一些
 * @param dis_threshold 点到直线距离，超过这个距离就认为没有形成直线
 * @return 形成的直线的参数 a b c d e f , 其中a b c是直线经过的点, d e f是直线的法向量(单位向量)
 */
template<typename T>
Eigen::Matrix<T,6,1> FormLine(const eigen_vector<Eigen::Matrix<T, 3, 1>>& points, T tolerance, T dis_threshold = 0)
{
    Eigen::Matrix<T,3,1> center(0, 0, 0);
    for (int i = 0; i < points.size(); i++)
    {
        center = center + points[i]; 
    }
    center = center / T(points.size());

    Eigen::Matrix<T,3,3> covMat = Eigen::Matrix<T,3,3>::Zero();
    for (int i = 0; i < points.size(); i++)
    {
        Eigen::Matrix<T, 3, 1> tmpZeroMean = points[i] - center;
        covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,3,3>> saes(covMat);

    // if is indeed line feature
    // note Eigen library sort eigenvalues in increasing order
    if(saes.eigenvalues()[2] > tolerance * saes.eigenvalues()[1])
    {
        Eigen::Matrix<T,3,1> unit_direction = saes.eigenvectors().col(2).normalized();
        Eigen::Matrix<T, 6, 1> line_coeff;
        line_coeff << center[0], center[1], center[2], unit_direction[0], unit_direction[1], unit_direction[2];
        if(dis_threshold > T(0.0))
        {
            T distance = 0;
            for(const Eigen::Matrix<T,3,1>& point : points)
            {
                distance = PointToLineDistance3D(point.data(), line_coeff.data());
                if(distance > dis_threshold)
                    return Eigen::Matrix<T,6,1>::Zero();
            }
        }
        return line_coeff;
    }
    else 
        return Eigen::Matrix<T,6,1>::Zero();
}


// 点到平面距离  plane -- 平面系数a b c d
// normalized 平面系数中的(a,b,c)是否已经经过归一化，也就是说(a,b,c)是否为单位向量
inline float PointToPlaneDistance(const cv::Vec4f& plane, const cv::Point3f& point, const bool normalized=false)
{
    if(!normalized)
        return abs(plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3]) / 
            sqrt(Square(plane[0]) + Square(plane[1]) + Square(plane[2]));
    else 
        return abs(plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3]);
}

// T = float  double
template<typename T>
inline T PointToPlaneDistance(const T* plane, const T* point, const bool normalized=false)
{
    if(!normalized)
        return abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) /
            sqrt(Square(plane[0]) + Square(plane[1]) + Square(plane[2]));
    else 
        return abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]);
}

// 点投影到平面
inline cv::Point3f ProjectPointToPlane(const cv::Point3f& point, const cv::Vec4f& plane, const bool normalized=false)
{
    float dis = PointToPlaneDistance(plane, point, normalized);
    float t = normalized ? dis : dis / sqrt(Square(plane[0]) + Square(plane[1]) + Square(plane[2]));
    cv::Point3f p;
    p.x = point.x - t * plane[0];
    p.y = point.y - t * plane[1];
    p.z = point.z - t * plane[2];
    // 判断一下得到的投影点是否在平面上，如果不在平面上那就是方向错了，应该是另一个方向
    if(abs(plane[0] * p.x + plane[1] * p.y + plane[2] * p.z + plane[3]) < 1e-4)
        return p;
    else 
        return cv::Point3f(point.x + t * plane[0], point.y + t * plane[1], point.z + t * plane[2]);
}

template<typename T>
inline void ProjectPointToPlane(const T* point, const T* plane, T* point_projected, const bool normalized = false)
{
    T dis = PointToPlaneDistance(plane, point, normalized);
    T t = normalized ? dis : dis / sqrt(Square(plane[0]) + Square(plane[1]) + Square(plane[2]));
    point_projected[0] = point[0] - t * plane[0];
    point_projected[1] = point[1] - t * plane[1];
    point_projected[2] = point[2] - t * plane[2];
    // 判断一下得到的投影点是否在平面上，如果不在平面上那就是方向错了，应该是另一个方向
    if(abs(plane[0] * point_projected[0] + plane[1] * point_projected[1] + plane[2] * point_projected[2] + plane[3]) > 1e-4)
    {
        point_projected[0] = point[0] + t * plane[0];
        point_projected[1] = point[1] + t * plane[1];
        point_projected[2] = point[2] + t * plane[2];
    }
}

template<typename T>
cv::Vec<T,4> FormPlane(const cv::Point3_<T>& p1, const cv::Point3_<T>& p2, const cv::Point3_<T>& p3)
{
    T a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
    T b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
    T c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );
    T d = -(a * p1.x + b * p1.y + c * p1.z);
    return cv::Vec<T,4>(a,b,c,d);
} 

template<typename T>
Eigen::Matrix<T, 4, 1> FormPlane(const Eigen::Matrix<T, 3, 1>& p1, const Eigen::Matrix<T, 3, 1>& p2, const Eigen::Matrix<T, 3, 1>& p3)
{
    T a = ( (p2.y()-p1.y())*(p3.z()-p1.z())-(p2.z()-p1.z())*(p3.y()-p1.y()) );
    T b = ( (p2.z()-p1.z())*(p3.x()-p1.x())-(p2.x()-p1.x())*(p3.z()-p1.z()) );
    T c = ( (p2.x()-p1.x())*(p3.y()-p1.y())-(p2.y()-p1.y())*(p3.x()-p1.x()) );
    T d = -(a * p1.x() + b * p1.y() + c * p1.z());
    return Eigen::Matrix<T, 4, 1>(a,b,c,d);
} 

// T-double float
/**
 * @description: 判断 n 个点能否形成一个平面
 * @param points n个点组成的vector
 * @param tolerance 点到平面距离的阈值，如果超出这个阈值，就认为这些点无法形成平面
 * @return 形成的平面参数，如果无法形成平面则返回零
 */
template<typename T>
Eigen::Matrix<T, 4, 1> FormPlane(const eigen_vector<Eigen::Matrix<T, 3, 1>>& points, T tolerance)
{
    const int size = points.size();
    Eigen::Matrix<T, Eigen::Dynamic, 3> matA0;
    matA0.resize(size, 3);
    Eigen::Matrix<T, Eigen::Dynamic, 1> matB0;
    matB0.resize(size, 1);
    matB0.fill(-1);
    for(int i = 0; i < size; i++)
    {
        matA0(i,0) = points[i].x();
        matA0(i,1) = points[i].y();
        matA0(i,2) = points[i].z();
    }
    // find the norm of plane
    Eigen::Matrix<T, 3, 1> norm = matA0.colPivHouseholderQr().solve(matB0);
    T negative_OA_dot_norm = 1 / norm.norm();
    norm.normalize();
    if(tolerance > 0)
    {
        for(const Eigen::Matrix<T, 3, 1>& p : points)
        {
            if(abs(norm.dot(p) + negative_OA_dot_norm) > tolerance)
                return Eigen::Matrix<T, 4, 1>(0,0,0,0);
        }
    }
    return  Eigen::Matrix<T, 4, 1>(norm.x(), norm.y(), norm.z(), negative_OA_dot_norm);
}

template<typename T>
Eigen::Matrix<float, 4, 1> FormPlane(const std::vector<T>& points, float tolerance)
{
    const int size = points.size();
    Eigen::Matrix<float, Eigen::Dynamic, 3> matA0;
    matA0.resize(size, 3);
    Eigen::Matrix<float, Eigen::Dynamic, 1> matB0;
    matB0.resize(size, 1);
    matB0.fill(-1);
    for(int i = 0; i < size; i++)
    {
        matA0(i,0) = points[i].x;
        matA0(i,1) = points[i].y;
        matA0(i,2) = points[i].z;
    }
    // find the norm of plane
    Eigen::Matrix<float, 3, 1> norm = matA0.colPivHouseholderQr().solve(matB0);
    float negative_OA_dot_norm = 1 / norm.norm();
    norm.normalize();
    if(tolerance > 0)
    {
        for(const T& p : points)
        {
            if(abs(norm.dot(Eigen::Matrix<float, 3, 1>(p.x, p.y, p.z)) + negative_OA_dot_norm) > tolerance)
                return Eigen::Matrix<float, 4, 1>(0,0,0,0);
        }
    }
  
    return  Eigen::Matrix<float, 4, 1>(norm.x(), norm.y(), norm.z(), negative_OA_dot_norm);
}

template<typename T>
inline Eigen::Matrix<T, 4, 1> TranslePlane(const Eigen::Matrix<T, 4, 1>& plane, const Eigen::Matrix<T, 4, 4>& translation)
{
    Eigen::Matrix<T, 3, 1> norm = translation.template block<3,3>(0,0) * plane.template block<3,1>(0,0) ;
    Eigen::Matrix<T, 3, 1> point(T(1), T(1), T(1));
    if(plane(0) != T(0))
        point(0) = -(plane(1) * point(1) + plane(2) * point(2) + plane(3)) / plane(0);
    else if (plane(1) != T(0))
        point(1) = -(plane(0) * point(0) + plane(2) * point(2) + plane(3)) / plane(1);
    else if(plane(2) != T(0))
        point(2) = -(plane(0) * point(0) + plane(1) * point(1) + plane(3)) / plane(2); 
    point = translation.template block<3,3>(0,0) * point + translation.template block<3,1>(0,3);
    return Eigen::Matrix<T, 4, 1>(norm(0), norm(1), norm(2), -norm.dot(point));  
}

inline float VectorAngle2D(const cv::Point2f& v1_start, const cv::Point2f& v1_end, 
                        const cv::Point2f& v2_start, const cv::Point2f& v2_end)
{
    cv::Vec2f v1(v1_start.x - v1_end.x, v1_start.y - v1_end.y);
    cv::Vec2f v2(v2_start.x - v2_end.x, v2_start.y - v2_end.y);
    return acos( v1.dot(v2) / cv::norm(v1) / cv::norm(v2) );
}

// 三维空间下两个向量的夹角，0-pi
// T-double float
template<typename T>
inline T VectorAngle3D(const cv::Point3_<T>& v1, const cv::Point3_<T>& v2, const bool normalized=false)
{
    T cos_angle = v1.dot(v2);
    if(!normalized)
    {
        T norm1 = sqrt(Square(v1.x) + Square(v1.y) + Square(v1.z));
        T norm2 = sqrt(Square(v2.x) + Square(v2.y) + Square(v2.z));
        cos_angle /= (norm1 * norm2);
    }
    if(cos_angle >= T(1.0))
        return T(0);
    else if(cos_angle <= T(-1.0))
        return T(M_PI);
    else 
        return acos(cos_angle);
}

// T - double float
template<typename T>
inline T VectorAngle3D(const T* v1, const T* v2, const bool normalized=false)
{
    T cos_angle = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    if(!normalized)
    {
        T norm1 = sqrt(Square(v1[0]) + Square(v1[1]) + Square(v1[2]));
        T norm2 = sqrt(Square(v2[0]) + Square(v2[1]) + Square(v2[2]));
        cos_angle /= (norm1 * norm2);
    }
    if(cos_angle >= T(1.0))
        return T(0);
    else if(cos_angle <= T(-1.0))
        return T(M_PI);
    else 
        return acos(cos_angle);
}

// 两个平面之间的夹角，由两个平面的法向量计算
// 两个平面之间的夹角永远是锐角
// T - float double
template<typename T>
inline T PlaneAngle(const T* plane1, const T* plane2, const bool normalized=false)
{
    T cos_angle = abs(plane1[0] * plane2[0] + plane1[1] * plane2[1] + plane1[2] * plane2[2]);
    if(!normalized)
    {
        T norm1 = sqrt(Square(plane1[0]) + Square(plane1[1]) + Square(plane1[2]) );
        T norm2 = sqrt(Square(plane2[0]) + Square(plane2[1]) + Square(plane2[2]) );
        cos_angle /= (norm1 * norm2);
    }
    if(cos_angle >= T(1.0))
        return T(0);
    else 
        return acos(cos_angle);
}

/**
 * @description: 计算两个平面相交的得到的直线
 * @param plane1 第一个平面的参数 ax + by + cz + d = 0
 * @param plane2 第二个平面参数
 * @return 形成的直线的参数 a b c d e f , 其中a b c是直线经过的点, d e f是直线的法向量(单位向量)
 */
template<typename T>
inline Eigen::Matrix<T,6,1> PlaneIntersect(const cv::Vec<T,4>& plane1, const cv::Vec<T,4>& plane2)
{
    Eigen::Matrix<T,2,2> A;
    A(0,0) = plane1[0]; A(0,1) = plane1[1];
    A(1,0) = plane2[0]; A(1,1) = plane2[1];
    Eigen::Matrix<T,2,1> b(-plane1[3] - plane1[2], -plane2[3] - plane2[2]);
    Eigen::Matrix<T,2,1> x;
    x = A.inverse() * b; 

    Eigen::Matrix<T,3,1> plane1_norm(plane1[0], plane1[1], plane1[2]);
    Eigen::Matrix<T,3,1> plane2_norm(plane2[0], plane2[1], plane2[2]);
    Eigen::Matrix<T,3,1> norm = plane1_norm.cross(plane2_norm);
    norm.normalize();
    Eigen::Matrix<T,6,1> line;
    line.template block<3,1>(3,0) = norm;
    line(0,0) = x(0,0);
    line(1,0) = x(1,0);
    line(2,0) = T(1.0);
    return line;
}

/**
 * @description: 计算平面和直线的交点
 * @param plane 平面参数，ax+by+cz+d=0中的 a b c d, 要求 a b c必须是单位向量
 * @param line 直线参数 a b c d e f , 其中a b c是直线经过的点, d e f是直线的法向量(单位向量)
 * @return 交点坐标
 */
template<typename T, typename D>
inline D PlaneLineIntersect(const T* plane, const T* line)
{
    T norm_dot = plane[0] * line[3] + plane[1] * line[4] + plane[2] * line[5];  // 两个法向量的内积
    // 内积接近于0，说明直线与平面平行，那么就没有交点
    if(abs(norm_dot) <= T(1e-6))
        return D(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
    
    T tmp = plane[0] * line[0] + plane[1] * line[1] + plane[2] * line[2] + plane[3];  // ax0 + by0 + cz0 + d
    tmp /= norm_dot;                                        // (ax0 + by0 + cz0 + d) / (anx + bny + cnz)
    T x = line[0] - tmp * line[3];
    T y = line[1] - tmp * line[4];
    T z = line[2] - tmp * line[5];
    return D(x,y,z);
}

/**
 * @description: 对三维点进行旋转平移
 * @param T 可以是 float double
 * @param point 三维点
 * @param transformation 变换矩阵
 * @return 变换后的三维点
 */
template<typename T1, typename T2>
inline cv::Point3_<T1> TranslatePoint(const cv::Point3_<T1>& point, const Eigen::Matrix<T2,4,4>& transformation)
{
    return cv::Point3_<T1>(
            point.x * transformation(0,0) + point.y * transformation(0,1) + point.z * transformation(0,2) + transformation(0,3),
            point.x * transformation(1,0) + point.y * transformation(1,1) + point.z * transformation(1,2) + transformation(1,3),
            point.x * transformation(2,0) + point.y * transformation(2,1) + point.z * transformation(2,2) + transformation(2,3));
}

template<typename T, typename D>
inline void TranslatePoint(T& point, const Eigen::Matrix<D, 4, 4>& transformation)
{
    D x , y, z;
    x = point.x * transformation(0,0) + point.y * transformation(0,1) + point.z * transformation(0,2) + transformation(0,3);
    y = point.x * transformation(1,0) + point.y * transformation(1,1) + point.z * transformation(1,2) + transformation(1,3);
    z = point.x * transformation(2,0) + point.y * transformation(2,1) + point.z * transformation(2,2) + transformation(2,3);
    point.x = x;
    point.y = y;
    point.z = z;
}

/**
 * @description: 对位姿进行线性插值
 * @param pose_w1 从局部坐标系1到世界的变换
 * @param pose_w2 从局部坐标系2到世界的变换
 * @param ratio 插值比例，越接近0则靠近pose_w1 越接近1则靠近pose_w2
 * @return 插值后的位姿，是从局部坐标系到世界的变换
 */
template<typename T>
Eigen::Matrix<T, 4, 4> SlerpPose(const Eigen::Matrix<T, 4, 4>& pose_w1, const Eigen::Matrix<T, 4, 4>& pose_w2, T ratio)
{
    Eigen::Matrix<T, 4, 4> T_21 = pose_w2.inverse() * pose_w1;
    Eigen::Quaternion<T> q_21(T_21.template block<3,3>(0,0));
    Eigen::Quaternion<T> q_s1 = Eigen::Quaternion<T>::Identity().slerp(ratio, q_21);
    Eigen::Matrix<T, 3, 1> t_s1 = (T_21.template block<3,1>(0,3)) * ratio;
    Eigen::Matrix<T, 4, 4> T_s1 = Eigen::Matrix<T, 4, 4>::Identity();
    T_s1.template block<3,3>(0,0) = Eigen::Matrix<T, 3, 3>(q_s1);
    T_s1.template block<3,1>(0,3) = t_s1;
    return pose_w1 * T_s1.inverse();
}

/**
 * @description: 从多个点中找到距离最远的两个点
 * @param points_all 所有的点
 * @param idx1 距离最远的两个点中的第一个点的索引
 * @param idx2 距离最远的两个点中的第二个点的索引
 * @param distance 两点间距离
 * @return {*}
 */
template<typename T>
bool FurthestPoints(const eigen_vector<Eigen::Matrix<T, 3, 1>>& points_all, int& idx1, int& idx2, T& distance)
{
    if(points_all.size() <= 1)
        return false;
    distance = -1;
    idx1 = idx2 = -1;
    for(int i = 0; i < points_all.size() - 1; i++)
    {
        for(int j = i + 1; j < points_all.size(); j++)
        {
            T curr_distance = (points_all[i] - points_all[j]).norm();
            // 记录距离最远的两个点
            if(curr_distance > distance)
            {
                idx1 = i;
                idx2 = j;
                distance = curr_distance;
            }
        }
    }
    if(idx1 < 0 || idx1 >= points_all.size() || idx2 < 0 || idx2 >= points_all.size())
        return false;
    return true;
}

template<typename T>
bool FurthestPoints(const pcl::PointCloud<T>& points_all, int& idx1, int& idx2, double& distance)
{
    if(points_all.size() <= 1)
        return false;
    distance = -1;
    idx1 = idx2 = -1;
    for(int i = 0; i < points_all.size() - 1; i++)
    {
        for(int j = i + 1; j < points_all.size(); j++)
        {
            double curr_distance = PointDistanceSquare(points_all.points[i], points_all.points[j]);
            // 记录距离最远的两个点
            if(curr_distance > distance)
            {
                idx1 = i;
                idx2 = j;
                distance = curr_distance;
            }
        }
    }
    distance = sqrt(distance);
    if(idx1 < 0 || idx1 >= points_all.size() || idx2 < 0 || idx2 >= points_all.size())
        return false;
    
    return true;
}


#endif