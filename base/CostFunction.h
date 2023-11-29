/*
 * @Author: Diantao Tu
 * @Date: 2021-10-22 15:08:03
 */

#ifndef _COST_FUNCTION_H_
#define _COST_FUNCTION_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include "Geometry.hpp"

// 计算一对图像之间的绝对旋转和相对旋转的差异，并且优化绝对旋转，使误差最小
struct PairWiseRotationResidual
{
    Eigen::Vector3d relative_rotation_21;   // 图像对之间的相对旋转
    double weight;
    
    PairWiseRotationResidual(const Eigen::Vector3d& _rot, const double _weight=1.0):
        relative_rotation_21(_rot),weight(_weight) {}
    
    template <typename T>
    bool operator()(const T* const angleAxis_1w, const T* const angleAxis_2w, T* residuals) const
    {
        const T angleAxis_21[3] = {T(relative_rotation_21.x()), T(relative_rotation_21.y()), T(relative_rotation_21.z())};
        Eigen::Matrix<T, 3, 3> R_21, R_1w, R_2w;
        ceres::AngleAxisToRotationMatrix(angleAxis_21, R_21.data());
        ceres::AngleAxisToRotationMatrix(angleAxis_1w, R_1w.data());
        ceres::AngleAxisToRotationMatrix(angleAxis_2w, R_2w.data());
        Eigen::Matrix<T, 3, 3> R_cycle = (R_2w * R_1w.transpose()) * R_21.transpose();
        T angleAxis_cycle[3];
        ceres::RotationMatrixToAngleAxis(R_cycle.data(), angleAxis_cycle);
        residuals[0] = T(weight) * angleAxis_cycle[0];
        residuals[1] = T(weight) * angleAxis_cycle[1];
        residuals[2] = T(weight) * angleAxis_cycle[2];

        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d _rot, const double _weight=1.0)
    {
        return (new ceres::AutoDiffCostFunction<PairWiseRotationResidual, 3, 3, 3>(
            new PairWiseRotationResidual(_rot, _weight)));
    }
};

// 计算一个图像对之间的绝对平移，使绝对平移和相对平移之间的差异最小
// residual = t_2w - R_21 * t_1w - scale * t_21
struct PairWiseTranslationResidual
{
    Eigen::Vector3d relative_rotation_21;
    Eigen::Vector3d relative_translation_21;
    double weight;
    PairWiseTranslationResidual(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, const double _weight=1.0):
        relative_translation_21(t_21), weight(_weight)
    {
        ceres::RotationMatrixToAngleAxis(R_21.data(), relative_rotation_21.data());
        relative_translation_21.normalize();
    }

    template <typename T>
    bool operator()(const T* const t_1w, const T* const t_2w, const T* const scale, T* residuals) const
    {
        T rotated_t_1w[3], R_21[3];
        R_21[0] = T(relative_rotation_21.x());
        R_21[1] = T(relative_rotation_21.y());
        R_21[2] = T(relative_rotation_21.z());
        ceres::AngleAxisRotatePoint(R_21, t_1w, rotated_t_1w);
        residuals[0] = T(t_2w[0] - rotated_t_1w[0] - (*scale) * T(relative_translation_21.x()));
        residuals[1] = T(t_2w[1] - rotated_t_1w[1] - (*scale) * T(relative_translation_21.y()));
        residuals[2] = T(t_2w[2] - rotated_t_1w[2] - (*scale) * T(relative_translation_21.z()));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix3d& R_21, const Eigen::Vector3d& t_21, const double _weight=1.0)
    {
        return (new ceres::AutoDiffCostFunction<PairWiseTranslationResidual, 3, 3, 3, 1>(
            new PairWiseTranslationResidual(R_21, t_21, _weight)
        ));
    }
};

// Robust Camera Location Estimation By Convex programming - CVPR 2015
// 这里的残差是 residual = t_w1 - t_w2 - scale * R_w2 * t_21
// t_w1 - t_w2 是以 C2为起点C1为终点的线段
// t_21是相机1的光心在相机2坐标系下的位置，也是从2到1的射线，只不过还要乘以 R_w2 变换到世界坐标系下
struct LUDResidual
{
    Eigen::Vector3d direction;
   
    double weight;
    LUDResidual(const Eigen::Matrix3d& R_w2, const Eigen::Vector3d& t_21, const double _weight=1.0):
         weight(_weight)
    {
        direction = (R_w2 * t_21).normalized();
    }

    template <typename T>
    bool operator()(const T* const t_w1, const T* const t_w2, const T* const scale, T* residuals) const
    {
        T x = t_w1[0] - t_w2[0] - (*scale) * T(direction.x());
        T y = t_w1[1] - t_w2[1] - (*scale) * T(direction.y());
        T z = t_w1[2] - t_w2[2] - (*scale) * T(direction.z());
        residuals[0] = T(weight) * ceres::sqrt(ceres::sqrt(x * x + y * y + z * z));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Matrix3d& R_w2, const Eigen::Vector3d& t_21, const double _weight=1.0)
    {
        return (new ceres::AutoDiffCostFunction<LUDResidual, 1, 3, 3, 1>(
            new LUDResidual(R_w2, t_21, _weight)
        ));
    }
};

// 关于相对位移t_ji 的尺度的优化，如果尺度超过上下界，那么就要受到一定的惩罚
struct ScaleFactor
{
    double weight;
    double upper_bound;
    double lower_bound;
    ScaleFactor(const double upper, const double lower, const double _weight):
                weight(_weight),upper_bound(upper),lower_bound(lower)
                {}
    template <typename T>
    bool operator()(const T* const scale, T* residuals) const
    {
        if(*scale > T(upper_bound))
            residuals[0] = T(weight) * (*scale - T(upper_bound));
        else if(*scale < T(lower_bound))
            residuals[0] = T(weight) * (T(lower_bound) - *scale);
        else 
            residuals[0] = T(0.0);
        return true;
    }
    static ceres::CostFunction *Create(const double upper, const double lower, const double _weight=1.0)
    {
        return (new ceres::AutoDiffCostFunction<ScaleFactor, 1, 1>(
            new ScaleFactor(upper, lower,_weight)
        ));
    }
};

// 计算两个相机的光心C1和C2的连线以及t_21之间的Chrodal 距离，并最小该距离的平方
// 这个公式来源于 Robust Global Translations with 1DSfM - ECCV 2014
// chrodal distance = ||u-v||_2   就是两个向量之差的二范数
struct ChrodalResidual
{
    Eigen::Vector3d direction;
    double weight;
    ChrodalResidual(const Eigen::Vector3d& t_21, const double w):direction(t_21),weight(w){}

    template<typename T>
    bool operator()(const T* const t_w1, const T* const t_w2, T* residual) const
    {
        T norm = sqrt((t_w1[0] - t_w2[0]) * (t_w1[0] - t_w2[0]) +
                (t_w1[1] - t_w2[1]) * (t_w1[1] - t_w2[1]) +
                (t_w1[2] - t_w2[2]) * (t_w1[2] - t_w2[2]) 
            );
        residual[0] = T(weight) * ((t_w1[0] - t_w2[0]) / norm - T(direction.x()));
        residual[1] = T(weight) * ((t_w1[1] - t_w2[1]) / norm - T(direction.y()));
        residual[2] = T(weight) * ((t_w1[2] - t_w2[2]) / norm - T(direction.z()));
        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d& t_21, const double w=1.0)
    {
        return (new ceres::AutoDiffCostFunction<ChrodalResidual, 3, 3, 3>(
            new ChrodalResidual(t_21, w)
        ));
    }
};

// 全景图像的投影误差，把一个空间点投影到单位球面上，得到球坐标系下的两个角度，作为投影点
// 计算投影点和对应的特征点之间的角度差异，并最小化这个差异
// 这里的残差是用两个角度来定义的，也就是 longitude 和 latitude
struct PanoramaReprojResidual_2Angle
{
    Eigen::Vector2d point_2d;   // 这个点是保存的角度 longitude latitude ,不是二维图像坐标x y
    double weight;
    PanoramaReprojResidual_2Angle(const Eigen::Vector2d& pt, double _weight):point_2d(pt), weight(_weight)
    {
        if(point_2d.x() < 0)
            point_2d.x() += 2 * M_PI;
    }
    template <typename T>
    bool operator()(const T* const angleAxis_cw, const T* const t_cw, const T* const point_3d, T* residuals) const
    {
        T point_c[3];
        ceres::AngleAxisRotatePoint(angleAxis_cw, point_3d, point_c);
        point_c[0] += t_cw[0];
        point_c[1] += t_cw[1];
        point_c[2] += t_cw[2];
        T norm = sqrt(point_c[0] * point_c[0] + point_c[1] * point_c[1] + point_c[2] * point_c[2]);
        T longitude = ceres::atan2(point_c[0], point_c[2]);
        T latitude = -ceres::asin(point_c[1] / norm);
        // 经线是从-pi 到 pi的，那么在接近-pi或pi的位置，例如 -179度和179度，实际只相差了2度，但是如果直接相减，
        // 会相差358度，造成计算错误，为了避免这种情况，统一把角度变换到0-2pi
        if(longitude < T(0.0))
            longitude += T(2 * M_PI);
        residuals[0] = T(weight) * (longitude - T(point_2d.x()));
        residuals[1] = T(weight) * (latitude - T(point_2d.y()));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d& pt, double _weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<PanoramaReprojResidual_2Angle, 2, 3, 3, 3>(
            new PanoramaReprojResidual_2Angle(pt, _weight)
        ));
    }
};

// 全景图像的投影误差，把一个空间点点投影到单位球面上，作为投影点
// 计算从球心到投影点的射线以及球心到特征点的射线之间的夹角，最小化这个夹角
// 这里的残差是1个角度，上面的方法用残差是两个角度
struct PanoramaReprojResidual_1Angle
{
    Eigen::Vector3d point_sphere;   // 这个点是特征点在单位圆上的坐标 x y z
    double weight;

    PanoramaReprojResidual_1Angle(const Eigen::Vector3d& pt, double _weight):point_sphere(pt),weight(_weight)
    {
        point_sphere.normalize();
    }
    template <typename T>
    bool operator()(const T* const angleAxis_cw, const T* const t_cw, const T* const point_3d, T* residuals) const
    {
        T point_c[3];
        ceres::AngleAxisRotatePoint(angleAxis_cw, point_3d, point_c);
        point_c[0] += t_cw[0];
        point_c[1] += t_cw[1];
        point_c[2] += t_cw[2];
        T norm = sqrt(point_c[0] * point_c[0] + point_c[1] * point_c[1] + point_c[2] * point_c[2]);
        T dot_product = point_c[0] * T(point_sphere(0)) + point_c[1] * T(point_sphere(1)) + point_c[2] * T(point_sphere(2));
        residuals[0] = T(weight) * ceres::acos(dot_product / norm);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& pt, double _weight = 1.0)
    {
        return (new ceres::AutoDiffCostFunction<PanoramaReprojResidual_1Angle, 1, 3, 3, 3>(
            new PanoramaReprojResidual_1Angle(pt, _weight)
        ));
    }
};

struct PanoramaReprojResidual_Pixel
{
    int rows, cols;
    Eigen::Vector2d point_2d;   // 这个点是保存的是二维图像坐标x y
    double weight;
    PanoramaReprojResidual_Pixel(const Eigen::Vector2d& pt, const int& _rows, const int& _cols, const double& _weight):
            point_2d(pt),rows(_rows),cols(_cols),weight(_weight)
    {}
    template <typename T>
    bool operator()(const T* const angleAxis_cw, const T* const t_cw, const T* const point_3d, T* residuals) const
    {
        T point_c[3];
        ceres::AngleAxisRotatePoint(angleAxis_cw, point_3d, point_c);
        point_c[0] += t_cw[0];
        point_c[1] += t_cw[1];
        point_c[2] += t_cw[2];
        T norm = sqrt(point_c[0] * point_c[0] + point_c[1] * point_c[1] + point_c[2] * point_c[2]);
        T longitude = ceres::atan2(point_c[0], point_c[2]);
        T latitude = -ceres::asin(point_c[1] / norm);
        T project_x = T(cols) * (T(0.5) + longitude / T(2.0 * M_PI));
        T project_y = T(rows) * (T(0.5) - latitude / T(M_PI));
        residuals[0] = T(weight) * (project_x - T(point_2d.x()));
        residuals[1] = T(weight) * (project_y - T(point_2d.y()));
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d& pt, const int& _rows, const int& _cols, double _weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<PanoramaReprojResidual_Pixel, 2, 3, 3, 3>(
            new PanoramaReprojResidual_Pixel(pt, _rows, _cols, _weight)
        ));
    }

    static ceres::CostFunction *Create(const cv::Point2f& pt, const int& _rows, const int& _cols, double _weight = 1)
    {
        return (new ceres::AutoDiffCostFunction<PanoramaReprojResidual_Pixel, 2, 3, 3, 3>(
            new PanoramaReprojResidual_Pixel(Eigen::Vector2d(pt.x, pt.y), _rows, _cols, _weight)
        ));
    }
};

// 两条直线分别在reference和neighbor坐标系下，把直线变换到reference坐标系下，两条直线都和坐标系的原点
// 形成一个平面，要优化这两个平面之间的夹角为0度
// 这里优化的是相对位姿，一般用在camera-LiDAR calibration上，所以相对位姿也就是 T_cl。reference坐标系是相机坐标系
// neighbor坐标系是LiDAR坐标系
struct Plane2Plane_Relative
{
    Eigen::Vector3d plane_ref;      // 在reference坐标系下的平面的法向量
    Eigen::Vector3d point_a;        // neighbor坐标系下直线的起点
    Eigen::Vector3d point_b;        // neighbor坐标系下直线的终点
    double weight;

    Plane2Plane_Relative(const Eigen::Vector3d& _plane_ref, const Eigen::Vector3d& _point_a, 
                        const Eigen::Vector3d& _point_b, const double _w = 1.0):
                plane_ref(_plane_ref), point_a(_point_a), point_b(_point_b), weight(_w)
    {
        plane_ref.normalize();
    }

    template <typename T>
    bool operator()(const T* const angleAxis_cl, const T* const t_cl, T* const residual) const
    {
        // 对point_a 进行坐标变换
        T pt_a[3] = {T(point_a.x()), T(point_a.y()), T(point_a.z())};
        T point_a_translated[3];
        ceres::AngleAxisRotatePoint(angleAxis_cl, pt_a, point_a_translated);
        point_a_translated[0] += t_cl[0];
		point_a_translated[1] += t_cl[1];
		point_a_translated[2] += t_cl[2];

        // 对point_b 进行坐标变换
        T pt_b[3] = {T(point_b.x()), T(point_b.y()), T(point_b.z())};
        T point_b_translated[3];
        ceres::AngleAxisRotatePoint(angleAxis_cl, pt_b, point_b_translated);
        point_b_translated[0] += t_cl[0];
		point_b_translated[1] += t_cl[1];
		point_b_translated[2] += t_cl[2];

        // 计算变换后的 a b 两点和球心形成的平面的法向量
        T a =(point_a_translated[1])*(point_b_translated[2])-(point_a_translated[2])*(point_b_translated[1]);
        T b =(point_a_translated[2])*(point_b_translated[0])-(point_a_translated[0])*(point_b_translated[2]);
        T c =(point_a_translated[0])*(point_b_translated[1])-(point_a_translated[1])*(point_b_translated[0]);

        T lidar_line_plane[3] = {a,b,c};
        T img_plane_norm[3] = {T(plane_ref.x()), T(plane_ref.y()), T(plane_ref.z())};
        residual[0] = T(weight) * PlaneAngle<T>(img_plane_norm, lidar_line_plane) * T(180.0) / T(M_PI);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d& _img_line_plane, 
                            const Eigen::Vector3d& _point_a, const Eigen::Vector3d& _point_b,
                            const double _w = 1.0)
    {
        return(new ceres::AutoDiffCostFunction<Plane2Plane_Relative, 1, 3, 3>(
            new Plane2Plane_Relative(_img_line_plane, _point_a, _point_b, _w)
        ));
    }
};

// 和上面的一样，计算两个直线形成的平面之间的夹角，优化夹角为0
// 不同之处在于这里的优化变量是绝对位姿，不是相对位姿
struct Plane2Plane_Global
{
    Eigen::Vector3d plane_ref;      // 在reference坐标系下的平面的法向量
    Eigen::Vector3d point_a;        // neighbor坐标系下直线的起点
    Eigen::Vector3d point_b;        // neighbor坐标系下直线的终点
    double weight;

    Plane2Plane_Global(const Eigen::Vector3d& _plane_ref, const Eigen::Vector3d& _point_a, 
                        const Eigen::Vector3d& _point_b, const double _w = 1.0):
                plane_ref(_plane_ref), point_a(_point_a), point_b(_point_b), weight(_w)
    {
        plane_ref.normalize();
    }

    template <typename T>
    bool operator()(const T* const angleAxis_rw, const T* const t_rw, 
                    const T* const angleAxis_nw, const T* const t_nw, T* const residual) const
    {
        // R_wn = R_nw.transpose()
        T angleAxis_wn[3] = {T(-1.0) * angleAxis_nw[0], T(-1.0) * angleAxis_nw[1], T(-1.0) * angleAxis_nw[2]};
        // t_wn = -R_wn * t_nw
        T t_wn[3];
        ceres::AngleAxisRotatePoint(angleAxis_wn, t_nw, t_wn);
        t_wn[0] *= T(-1.0);
        t_wn[1] *= T(-1.0);
        t_wn[2] *= T(-1.0);
        // 对point_a 进行坐标变换, 先变换到世界坐标系
        T pt_a[3] = {T(point_a.x()), T(point_a.y()), T(point_a.z())};
        T point_a_world[3];
        
        ceres::AngleAxisRotatePoint(angleAxis_wn, pt_a, point_a_world);
        point_a_world[0] += t_wn[0];
		point_a_world[1] += t_wn[1];
		point_a_world[2] += t_wn[2];
        // 然后把point_a变换到ref坐标系
        T point_a_ref[3];
        ceres::AngleAxisRotatePoint(angleAxis_rw, point_a_world, point_a_ref);
        point_a_ref[0] += t_rw[0];
		point_a_ref[1] += t_rw[1];
		point_a_ref[2] += t_rw[2];
        // 对point_b 进行坐标变换, 先变换到世界坐标系
        T pt_b[3] = {T(point_b.x()), T(point_b.y()), T(point_b.z())};
        T point_b_world[3];
        
        ceres::AngleAxisRotatePoint(angleAxis_wn, pt_b, point_b_world);
        point_b_world[0] += t_wn[0];
		point_b_world[1] += t_wn[1];
		point_b_world[2] += t_wn[2];
        // 然后把point_b变换到相机坐标系
        T point_b_ref[3];
        ceres::AngleAxisRotatePoint(angleAxis_rw, point_b_world, point_b_ref);
        point_b_ref[0] += t_rw[0];
		point_b_ref[1] += t_rw[1];
		point_b_ref[2] += t_rw[2];

        // 计算变换后的 a b 两点和球心形成的平面的法向量
        T a =(point_a_ref[1])*(point_b_ref[2])-(point_a_ref[2])*(point_b_ref[1]);
        T b =(point_a_ref[2])*(point_b_ref[0])-(point_a_ref[0])*(point_b_ref[2]);
        T c =(point_a_ref[0])*(point_b_ref[1])-(point_a_ref[1])*(point_b_ref[0]);

        T plane1[3] = {a,b,c};
        T plane2[3] = {T(plane_ref.x()), T(plane_ref.y()), T(plane_ref.z())};
        residual[0] = T(weight) * PlaneAngle<T>(plane2, plane1);
        return true;
    }


    static ceres::CostFunction *Create(const Eigen::Vector3d& _plane_ref, 
                            const Eigen::Vector3d& _point_a, const Eigen::Vector3d& _point_b,
                            const double _w = 1.0)
    {
        return(new ceres::AutoDiffCostFunction<Plane2Plane_Global, 1, 3, 3, 3, 3>(
            new Plane2Plane_Global(_plane_ref, _point_a, _point_b, _w)
        ));
    }
};

// 两条直线即使处在在同一个平面上，也可能会出现两条直线是完全错开的情况，也就是两条直线在空间上完全没有交集，或者从投影角度理解就是
// 两条直线在图像上的投影完全没有任何重合。为了避免这种情况出现，需要一个类似于两条直线的IOU的残差项。
// 这一项主要就是把neighbor直线的中点投影到reference直线形成的平面上，得到投影点A。reference直线的中点记为B，要求 OA和OB之间的
// 夹角要小于angle, 这里angle是根据需要修改的，一般设置为reference直线对应的圆心角的一半
// 在LiDAR-LiDAR和camera-LiDAR里都使用了这一项残差
// camera-LiDAR中，camera对应于reference坐标系
struct PlaneIOUResidual
{
    Eigen::Vector4d ref_plane;              // reference直线形成的平面的参数 a b c d
    Eigen::Vector3d middle_neighbor;        // neighbor直线的中点，neighbor坐标系下
    Eigen::Vector3d middle_ref;             // reference直线的中点，reference坐标系下
    double angle;                           // 两条直线的中点夹角的阈值，超过这个阈值就要收到惩罚
    double weight;

    // 这个初始化函数适用于 camera-LiDAR
    PlaneIOUResidual(const Eigen::Vector4d& _image_plane, const Eigen::Vector3d& _middle_lidar,
                    const Eigen::Vector3d& image_line_start, const Eigen::Vector3d& image_line_end, 
                    const double _weight = 1.0):
                    middle_neighbor(_middle_lidar),weight(_weight)
    {
        // 把平面法向量归一化
        ref_plane = _image_plane / _image_plane.block<3,1>(0,0).norm();
        angle = VectorAngle3D(image_line_start.data(), image_line_end.data()) / 2.0;
        middle_ref = (image_line_start + image_line_end) / 2.0;
    }
    // 这个初始化函数适用于LiDAR-LiDAR
    PlaneIOUResidual(const Eigen::Vector4d& _ref_plane, const Eigen::Vector3d& _middle_neighbor,
                    const Eigen::Vector3d& _middle_ref, const double _angle, const double _weight = 1.0)
                    :middle_neighbor(_middle_neighbor), middle_ref(_middle_ref),
                    angle(_angle), weight(_weight)
    {
        // 把平面法向量归一化
        ref_plane = _ref_plane / _ref_plane.block<3,1>(0,0).norm();
    }
    template <typename T>
    bool operator()(const T* const angleAxis_rw, const T* const t_rw, 
                    const T* const angleAxis_nw, const T* const t_nw, T* const residual) const
    {
        // R_wn = R_nw.transpose()
        T angleAxis_wn[3] = {T(-1.0) * angleAxis_nw[0], T(-1.0) * angleAxis_nw[1], T(-1.0) * angleAxis_nw[2]};
        // t_wn = -R_wn * t_nw
        T t_wn[3];
        ceres::AngleAxisRotatePoint(angleAxis_wn, t_nw, t_wn);
        t_wn[0] *= T(-1.0);
        t_wn[1] *= T(-1.0);
        t_wn[2] *= T(-1.0);
        // 对middle_neighbor 进行坐标变换, 先变换到世界坐标系
        T pt_a[3] = {T(middle_neighbor.x()), T(middle_neighbor.y()), T(middle_neighbor.z())};
        T middle_neighbor_world[3];
        
        ceres::AngleAxisRotatePoint(angleAxis_wn, pt_a, middle_neighbor_world);
        middle_neighbor_world[0] += t_wn[0];
		middle_neighbor_world[1] += t_wn[1];
		middle_neighbor_world[2] += t_wn[2];
        // 然后把middle_neighbor变换到reference坐标系
        T middle_neighbor_ref[3];
        ceres::AngleAxisRotatePoint(angleAxis_rw, middle_neighbor_world, middle_neighbor_ref);
        middle_neighbor_ref[0] += t_rw[0];
		middle_neighbor_ref[1] += t_rw[1];
		middle_neighbor_ref[2] += t_rw[2];

        T ref_plane_coeff[4] = {T(ref_plane[0]), T(ref_plane[1]), T(ref_plane[2]), T(ref_plane[3])};
        T image_point[3] = {T(middle_ref.x()), T(middle_ref.y()), T(middle_ref.z())};
        T neighbor_projected[3];
        ProjectPointToPlane(middle_neighbor_ref, ref_plane_coeff, neighbor_projected, true);
        T curr_angle = VectorAngle3D(neighbor_projected, image_point);
        if(curr_angle < T(angle))
            residual[0] = T(0.0);
        else 
            residual[0] = T(weight) * (curr_angle - T(angle));
        return true;
    }
    
    static ceres::CostFunction *Create(const Eigen::Vector4d& _ref_plane, const Eigen::Vector3d& _middle_neighbor,
                    const Eigen::Vector3d& _middle_ref, const double _angle, const double _weight = 1.0)
    {
        return(new ceres::AutoDiffCostFunction<PlaneIOUResidual, 1, 3, 3, 3, 3>(
            new PlaneIOUResidual(_ref_plane, _middle_neighbor, _middle_ref, _angle, _weight)
        ));
    }
};

struct PlaneRelativeIOUResidual
{
    Eigen::Vector4d ref_plane;              // reference直线形成的平面的参数 a b c d
    Eigen::Vector3d middle_neighbor;        // neighbor直线的中点，neighbor坐标系下
    Eigen::Vector3d middle_ref;             // reference直线的中点，reference坐标系下
    double angle;                           // reference直线所对应的圆心角的一半
    double weight;

    // 这个初始化函数适用于 camera-LiDAR
    PlaneRelativeIOUResidual(const Eigen::Vector4d& _image_plane, const Eigen::Vector3d& _middle_lidar,
                    const cv::Point3f& image_line_start, const cv::Point3f& image_line_end, 
                    const double _weight = 1.0):
                    middle_neighbor(_middle_lidar),weight(_weight)
    {
        // 把平面法向量归一化
        ref_plane = _image_plane / _image_plane.block<3,1>(0,0).norm();
        angle = VectorAngle3D(image_line_start, image_line_end) / 2.f;
        middle_ref.x() = (image_line_start.x + image_line_end.x) / 2.f;
        middle_ref.y() = (image_line_start.y + image_line_end.y) / 2.f;
        middle_ref.z() = (image_line_start.z + image_line_end.z) / 2.f;
    }

    template <typename T>
    bool operator()(const T* const angleAxis_cl, const T* const t_cl, T* const residual) const
    {
        // 对middle_neighbor 进行坐标变换
        T pt_neighbor[3] = {T(middle_neighbor.x()), T(middle_neighbor.y()), T(middle_neighbor.z())};
        T middle_neighbor_translated[3];
        ceres::AngleAxisRotatePoint(angleAxis_cl, pt_neighbor, middle_neighbor_translated);
        middle_neighbor_translated[0] += t_cl[0];
		middle_neighbor_translated[1] += t_cl[1];
		middle_neighbor_translated[2] += t_cl[2];

        T ref_plane_coeff[4] = {T(ref_plane[0]), T(ref_plane[1]), T(ref_plane[2]), T(ref_plane[3])};
        T ref_point[3] = {T(middle_ref.x()), T(middle_ref.y()), T(middle_ref.z())};
        T neighbor_projected[3];
        ProjectPointToPlane(middle_neighbor_translated, ref_plane_coeff, neighbor_projected, true);
        T curr_angle = VectorAngle3D(neighbor_projected, ref_point);
        if(curr_angle < T(angle))
            residual[0] = T(0.0);
        else 
            residual[0] = T(weight) * (curr_angle - T(angle));
        return true;

    }

    static ceres::CostFunction *Create(const Eigen::Vector4d& _ref_plane, const Eigen::Vector3d& _middle_neighbor,
                            const cv::Point3f& image_line_start, const cv::Point3f& image_line_end, 
                            const double _weight = 1.0)
    {
        return(new ceres::AutoDiffCostFunction<PlaneRelativeIOUResidual, 1, 3, 3>(
            new PlaneRelativeIOUResidual(_ref_plane, _middle_neighbor, image_line_start, image_line_end, _weight)
        ));
    }
};

// 把一个在neighbor坐标系下的点变换到reference坐标系下，要求变换后的点要尽可能接近reference坐标系下该点对应的平面，
// 以此来优化neighbor和reference的位姿，不优化点的坐标
struct Point2Plane_Meter
{
    Eigen::Vector4d plane;              // reference 坐标系下的平面参数，要求法向量是归一化的
	Eigen::Vector3d curr_point;         // neighbor 坐标系下的三维点坐标
	double weight;                      // 权重
	// 把三维点变换到雷达坐标系下，让这个三维点尽可能的接近它所在的雷达坐标系的那个平面
	// 假设三维点到雷达原点组成的向量在法向量方向上的投影为-m，平面到原点距离为d
	// 残差就是d-m，越接近0说明三维点越靠近平面
    // 起始这个d-m就是点到平面距离
    Point2Plane_Meter(  Eigen::Vector3d curr_point_, Eigen::Vector4d _plane, double w_):
						curr_point(curr_point_), plane(_plane), weight(w_) {}
	template <typename T>
    // r-reference   n-neighbor   w-world
	bool operator()(const T* const angleaxis_rw, const T* const t_rw,
				const T* const angleaxis_nw, const T* const t_nw, T *residual ) const
	{
        // 把三维点从neighbor坐标系变换到reference坐标系
		// P_r = R_rw * R_wn * P_n - R_rw * R_wn * t_nw + t_rw
		T point_ref[3];
		T angleaxis_rn[3];
		T angleaxis_wn[3];
		T vec_tmp[3];
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		angleaxis_wn[0] = (T)(-1.0) * angleaxis_nw[0];
		angleaxis_wn[1] = (T)(-1.0) * angleaxis_nw[1];
		angleaxis_wn[2] = (T)(-1.0) * angleaxis_nw[2];
		Eigen::Matrix<T, 3, 3> R_rw;
		Eigen::Matrix<T, 3, 3> R_wn;
		ceres::AngleAxisToRotationMatrix(angleaxis_rw, R_rw.data());
		ceres::AngleAxisToRotationMatrix(angleaxis_wn, R_wn.data());
		Eigen::Matrix<T, 3, 3> R_rn = R_rw * R_wn;
		ceres::RotationMatrixToAngleAxis(R_rn.data(), angleaxis_rn);

		ceres::AngleAxisRotatePoint(angleaxis_rn, point, point_ref);	// R_rw * R_wn * P_n
		ceres::AngleAxisRotatePoint(angleaxis_rn, t_nw, vec_tmp);	// R_rw * R_wn * t_nw
		point_ref[0] = point_ref[0] - vec_tmp[0] + t_rw[0];
		point_ref[1] = point_ref[1] - vec_tmp[1] + t_rw[1];
		point_ref[2] = point_ref[2] - vec_tmp[2] + t_rw[2];

        T plane_coeff[4] = {T(plane(0)), T(plane(1)), T(plane(2)), T(plane(3))};
        residual[0] = T(weight) * PointToPlaneDistance(plane_coeff, point_ref, true);

		return true;
	}

    static ceres::CostFunction *Create( const Eigen::Vector3d& curr_point_, const Eigen::Vector4d& _plane,
									    const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				Point2Plane_Meter, 1,3,3,3,3>(
			new Point2Plane_Meter(curr_point_, _plane, weight_)));
	}
};

// 使用角度来衡量点到平面的距离，并最小化该角度，使点到平面距离最小
// 假设一个P点在neighbor坐标系下，找到了与之匹配的平面，该平面用 a b c d 来表示，在reference坐标系下
// 那么把P变换到reference坐标系，然后把它投影到平面上得到P',这样就可以计算OP以及OP'的夹角，以这个角度作为距离的度量
// O是reference坐标系的原点
// 注意：这里有个问题，因为空间中两个射线的夹角是通过 arccos计算的，也就是说 
// \theta =  arccos[ (OP).dot(OP') / OP.norm() / OP'.norm()] , 但是arccos的导数是 -1/sqrt(1-x^2),这就导致了当x=1或-1时
// 导数为无穷，但其实arccos的最小值正是取在x=1处，因此使用ceres优化时会报错，报错内容是雅克比为inf或nan之类的
// 为了避免这种情况，只能人为设置了一个条件，当PP'距离很小的时候，直接让残差为0
// 很奇怪为什么会报这个错误，因为其他的优化函数都是用的arccos，却没有一个出现问题，只有这个有问题
struct Point2Plane_Angle
{
	Eigen::Vector4d plane;                  // reference 坐标系下的平面
	Eigen::Vector3d curr_point;             // neighbor 坐标系下的雷达点
	double weight;                          // 权重
    bool normalize_distance;
	// 把三维点变换到雷达坐标系下，让这个三维点尽可能的接近它所在的雷达坐标系的那个平面
	Point2Plane_Angle(  const Eigen::Vector3d& curr_point_, const Eigen::Vector4d& _plane,
						bool normalize ,double w_=1.0):
						curr_point(curr_point_), plane(_plane), weight(w_), normalize_distance(normalize) 
    {}
  
	template <typename T>
    // r-reference   n-neighbor   w-world
	bool operator()(const T* const angleaxis_rw, const T* const t_rw,
				const T* const angleaxis_nw, const T* const t_nw, T *residual ) const
	{
        // 把三维点从neighbor坐标系变换到reference坐标系
		// P_r = R_rw * R_wn * P_n - R_rw * R_wn * t_nw + t_rw
		T point_ref[3];
		T angleaxis_rn[3];
		T angleaxis_wn[3];
		T vec_tmp[3];
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
		angleaxis_wn[0] = (T)(-1.0) * angleaxis_nw[0];
		angleaxis_wn[1] = (T)(-1.0) * angleaxis_nw[1];
		angleaxis_wn[2] = (T)(-1.0) * angleaxis_nw[2];
		Eigen::Matrix<T, 3, 3> R_rw;
		Eigen::Matrix<T, 3, 3> R_wn;
		ceres::AngleAxisToRotationMatrix(angleaxis_rw, R_rw.data());
		ceres::AngleAxisToRotationMatrix(angleaxis_wn, R_wn.data());
		Eigen::Matrix<T, 3, 3> R_rn = R_rw * R_wn;
		ceres::RotationMatrixToAngleAxis(R_rn.data(), angleaxis_rn);

		ceres::AngleAxisRotatePoint(angleaxis_rn, point, point_ref);	// R_rw * R_wn * P_n
		ceres::AngleAxisRotatePoint(angleaxis_rn, t_nw, vec_tmp);	// R_rw * R_wn * t_nw
		point_ref[0] = point_ref[0] - vec_tmp[0] + t_rw[0];
		point_ref[1] = point_ref[1] - vec_tmp[1] + t_rw[1];
		point_ref[2] = point_ref[2] - vec_tmp[2] + t_rw[2];

        T plane_coeff[4] = {T(plane(0)), T(plane(1)), T(plane(2)), T(plane(3))};
        T point_projected[3];
        // 这是原本的简单写法
        // ProjectPointToPlane(point_ref, plane_coeff, point_projected, true);
        // residual[0] = VectorAngle3D(point_ref, point_projected);
        // if(residual[0] < T(1e-3))
        //     residual[0] += T(0.001);
		
        
        T dis = PointToPlaneDistance(plane_coeff, point_ref, true);
        if(dis < T(1e-3))
        {
            residual[0] = T(0.0);
            return true;
        }
        point_projected[0] = point_ref[0] - dis * plane_coeff[0];
        point_projected[1] = point_ref[1] - dis * plane_coeff[1];
        point_projected[2] = point_ref[2] - dis * plane_coeff[2];
        if(abs(plane_coeff[0] * point_projected[0] + plane_coeff[1] * point_projected[1] + plane_coeff[2] * point_projected[2] + plane_coeff[3]) > 1e-4)
        {
            point_projected[0] = point_ref[0] + dis * plane_coeff[0];
            point_projected[1] = point_ref[1] + dis * plane_coeff[1];
            point_projected[2] = point_ref[2] + dis * plane_coeff[2];
        }
        // 对距离进行归一化
        if(normalize_distance)
        {
            // residual[0] = ceres::acos(dis * dis / T(2.0) - T(1.0));
            // return true;
            T norm = sqrt(point_projected[0] * point_projected[0] + 
                        point_projected[1] * point_projected[1] + 
                        point_projected[2] * point_projected[2]);
            T ratio = (norm - T(1.0)) / norm;
            // 把球心放到距离投影点1m的位置上
            T center_normalized[3];
            center_normalized[0] = ratio * point_projected[0];
            center_normalized[1] = ratio * point_projected[1];
            center_normalized[2] = ratio * point_projected[2];
            T vec1[3] = {point_projected[0] - center_normalized[0], 
                        point_projected[1] - center_normalized[1], 
                        point_projected[2] - center_normalized[2]};
            T vec2[3] = {point_ref[0] - center_normalized[0], 
                        point_ref[1] - center_normalized[1], 
                        point_ref[2] - center_normalized[2]};
            residual[0] = VectorAngle3D(vec1, vec2);
        }
        else 
            residual[0] = VectorAngle3D(point_ref, point_projected);
        return true;
	}

	static ceres::CostFunction *Create( const Eigen::Vector3d& curr_point_, const Eigen::Vector4d& _plane,
									    const bool normalize, const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				Point2Plane_Angle, 1,3,3,3,3>(
			new Point2Plane_Angle(curr_point_, _plane, normalize, weight_)));
	}

};

// 一对匹配的雷达之间的点到平面距离
struct PairWisePoint2Plane_Meter
{
    Eigen::Vector4d plane;                  // LiDAR2坐标系下的平面参数，法向量要归一化
	Eigen::Vector3d curr_point;             // LiDAR1坐标系下的点
	double weight;
	// 把三维点变换到雷达坐标系下，让这个三维点尽可能的接近它所在的雷达坐标系的那个平面
	// 假设三维点到雷达原点组成的向量在法向量方向上的投影为-m，平面到原点距离为d
	// 残差就是d-m，越接近0说明三维点越靠近平面
    PairWisePoint2Plane_Meter(  Eigen::Vector3d curr_point_, Eigen::Vector4d _plane, double w_):
						curr_point(curr_point_), plane(_plane), weight(w_) {}
						
	template <typename T>
	bool operator()(const T* const angleaxis_21, const T* const t_21, T *residual ) const			 
	{
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        T point_2[3];
        ceres::AngleAxisRotatePoint(angleaxis_21, point, point_2);
        point_2[0] += t_21[0];
        point_2[1] += t_21[1];
        point_2[2] += t_21[2];

        // 残差就是点到平面距离
        T plane_coeff[4] = {T(plane(0)), T(plane(1)), T(plane(2)), T(plane(3))};
        residual[0] = T(weight) * PointToPlaneDistance(plane_coeff, point_2, true);
		return true;
	}

    static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_, const Eigen::Vector4d& _plane, 
                                        const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				PairWisePoint2Plane_Meter, 1,3,3>(
			new PairWisePoint2Plane_Meter(curr_point_, _plane, weight_)));
	}
};

// 用的是点到直线距离作为残差
struct Point2Line_Meter
{
    // Eigen::Vector3d last_point_a, last_point_b;     // 这两个点是在reference坐标系下
    Eigen::Vector3d line_point;             // 直线经过的点，在reference坐标系下
    Eigen::Vector3d line_direction;         // 直线的方向，在reference坐标系下
	Eigen::Vector3d curr_point;             // 这个点是在neighbor坐标系下
	double weight;                          // 权重
	// 把三维空间点变换到雷达坐标系下，要求三维点尽可能靠近他所属的那个直线
	// 这里的curr_point是属于neighbor坐标系下的点, 也就是让neighbor坐标系靠近reference坐标系
	Point2Line_Meter( Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
					Eigen::Vector3d last_point_b_, double w_)
		: line_point(last_point_a_),curr_point(curr_point_), weight(w_) 
    {
        line_direction = (last_point_a_ - last_point_b_).normalized();
    }

	template <typename T>
	// r-reference   n-neighbor   w-world
	bool operator()(const T* const angleaxis_rw, const T* const t_rw,
				const T* const angleaxis_nw, const T* const t_nw, T *residual) const
	{
		// P_r = R_rw * R_wn * P_n - R_rw * R_wn * t_nw + t_rw
		T point_ref[3];
		T angleaxis_rn[3];
		T angleaxis_wn[3];
		T vec_tmp[3];
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};

		angleaxis_wn[0] = (T)(-1.0) * angleaxis_nw[0];
		angleaxis_wn[1] = (T)(-1.0) * angleaxis_nw[1];
		angleaxis_wn[2] = (T)(-1.0) * angleaxis_nw[2];
		Eigen::Matrix<T, 3, 3> R_rw;
		Eigen::Matrix<T, 3, 3> R_wn;
		ceres::AngleAxisToRotationMatrix(angleaxis_rw, R_rw.data());
		ceres::AngleAxisToRotationMatrix(angleaxis_wn, R_wn.data());
		Eigen::Matrix<T, 3, 3> R_rn = R_rw * R_wn;
		ceres::RotationMatrixToAngleAxis(R_rn.data(),angleaxis_rn);

		ceres::AngleAxisRotatePoint(angleaxis_rn, point, point_ref);	// R_rw * R_wn * P_n
		ceres::AngleAxisRotatePoint(angleaxis_rn, t_nw, vec_tmp);	// R_rw * R_wn * t_nw
		point_ref[0] = point_ref[0] - vec_tmp[0] + t_rw[0];
		point_ref[1] = point_ref[1] - vec_tmp[1] + t_rw[1];
		point_ref[2] = point_ref[2] - vec_tmp[2] + t_rw[2];

        T line[6] = {T(line_point.x()), T(line_point.y()), T(line_point.z()),
                    T(line_direction.x()), T(line_direction.y()), T(line_direction.z())};
		residual[0] = T(weight) * PointToLineDistance3D(point_ref, line);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, 
										const Eigen::Vector3d last_point_a_,
									   const Eigen::Vector3d last_point_b_,
									   const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				Point2Line_Meter, 1,3,3,3,3>(
			new Point2Line_Meter(curr_point_ ,last_point_a_, last_point_b_, weight_)));
	}
};

// 一个点在neighbor坐标系下，经过匹配后得到它的匹配直线在reference坐标系下。把这个点先从neighbor变换到reference下，得到点P。
// 然后把点P投影到直线上得到P'，计算OP和OP'的夹角，就可以用来度量点P到直线的距离。
// 但这里也是有奇异性的，在于如果点P，点O，直线L 三者共面，那么无论P与L的距离是多少，度量得到的角度都是0.
// 然而，由于对于任意的两个LiDAR数据来说，它们之间的点到面的约束不会仅仅集中在一个点到一条直线上，而是在空间中各个位置都有的，
// 那么在这种情况下就可以在一定程度上减少这种奇异性的问题
struct Point2Line_Angle
{
    Eigen::Vector3d line_point;             // 直线经过的点，在reference坐标系下
    Eigen::Vector3d line_direction;         // 直线的方向，在reference坐标系下
	Eigen::Vector3d curr_point;             // 这个点是在neighbor坐标系下
	double weight;
    bool normalize_distance;
	// 把三维空间点变换到雷达坐标系下，要求三维点尽可能靠近他所属的那个直线
	// 这里的curr_point是属于neighbor坐标系下的点, 也就是让neighbor坐标系靠近reference坐标系
	Point2Line_Angle(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
					const Eigen::Vector3d& last_point_b_, const bool& normalize, double w_)
		: line_point(last_point_a_),curr_point(curr_point_), weight(w_), normalize_distance(normalize) 
    {
        line_direction = (last_point_a_ - last_point_b_).normalized();
    }

	template <typename T>
	// r-reference   n-neighbor   w-world
	bool operator()(const T* const angleaxis_rw, const T* const t_rw,
				const T* const angleaxis_nw, const T* const t_nw, T *residual) const
	{
		// P_r = R_rw * R_wn * P_n - R_rw * R_wn * t_nw + t_rw
		T point_ref[3];
		T angleaxis_rn[3];
		T angleaxis_wn[3];
		T vec_tmp[3];
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};

		angleaxis_wn[0] = (T)(-1.0) * angleaxis_nw[0];
		angleaxis_wn[1] = (T)(-1.0) * angleaxis_nw[1];
		angleaxis_wn[2] = (T)(-1.0) * angleaxis_nw[2];
		Eigen::Matrix<T, 3, 3> R_rw;
		Eigen::Matrix<T, 3, 3> R_wn;
		ceres::AngleAxisToRotationMatrix(angleaxis_rw, R_rw.data());
		ceres::AngleAxisToRotationMatrix(angleaxis_wn, R_wn.data());
		Eigen::Matrix<T, 3, 3> R_rn = R_rw * R_wn;
		ceres::RotationMatrixToAngleAxis(R_rn.data(),angleaxis_rn);

		ceres::AngleAxisRotatePoint(angleaxis_rn, point, point_ref);	// R_rw * R_wn * P_n
		ceres::AngleAxisRotatePoint(angleaxis_rn, t_nw, vec_tmp);	// R_rw * R_wn * t_nw
		point_ref[0] = point_ref[0] - vec_tmp[0] + t_rw[0];
		point_ref[1] = point_ref[1] - vec_tmp[1] + t_rw[1];
		point_ref[2] = point_ref[2] - vec_tmp[2] + t_rw[2];

        // 点到直线的投影
		T x0 = T(line_point.x());
        T y0 = T(line_point.y());
        T z0 = T(line_point.z());
        T nx = T(line_direction.x());
        T ny = T(line_direction.y());
        T nz = T(line_direction.z());
        T k = nx * (point_ref[0] - x0) + ny * (point_ref[1] - y0) + nz * (point_ref[2] - z0);
        T point_projected[3] = {k * nx + x0, k * ny + y0, k * nz + z0};
        // 点到投影点之间的距离
        T dis = sqrt((point_ref[0] - point_projected[0]) * (point_ref[0] - point_projected[0]) + 
                    (point_ref[1] - point_projected[1]) * (point_ref[1] - point_projected[1]) +
                    (point_ref[2] - point_projected[2]) * (point_ref[2] - point_projected[2]));
        if(dis < T(1e-3))
        {
            residual[0] = T(0.0);
            return true;
        }
         // 对距离进行归一化
        if(normalize_distance)
        {
            T norm = sqrt(point_projected[0] * point_projected[0] + 
                        point_projected[1] * point_projected[1] + 
                        point_projected[2] * point_projected[2]);
            T ratio = (norm - T(1.0)) / norm;
            // 把球心放到距离投影点1m的位置上
            T center_normalized[3];
            center_normalized[0] = ratio * point_projected[0];
            center_normalized[1] = ratio * point_projected[1];
            center_normalized[2] = ratio * point_projected[2];
            T vec1[3] = {point_projected[0] - center_normalized[0], 
                        point_projected[1] - center_normalized[1], 
                        point_projected[2] - center_normalized[2]};
            T vec2[3] = {point_ref[0] - center_normalized[0], 
                        point_ref[1] - center_normalized[1], 
                        point_ref[2] - center_normalized[2]};
            residual[0] = VectorAngle3D(vec1, vec2);
        }
        // 计算投影点和原始点之间的夹角
        else
            residual[0] = VectorAngle3D(point_ref, point_projected);
		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d& curr_point_, 
										const Eigen::Vector3d& last_point_a_,
									   const Eigen::Vector3d& last_point_b_,
                                       const bool normalize,
									   const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				Point2Line_Angle, 1,3,3,3,3>(
			new Point2Line_Angle(curr_point_ ,last_point_a_, last_point_b_, normalize, weight_)));
	}
};

// 一对匹配的雷达之间的点到直线距离
// 其实这个就是上面的 Point2Line_Meter 的特殊形式，因为仅仅是一对之间的，相当于其中一个位姿被固定在原点处
// 因此可以使用上面的残差，但是那样的话就会增加搜索空间（因为有两个位姿），可能没法收敛到最优，所以还是单独写了一个残差项
struct PairWisePoint2Line_Meter
{
    Eigen::Vector3d line_point;             // 直线经过的点，在LiDAR2坐标系下
    Eigen::Vector3d line_direction;         // 直线的方向，在LiDAR2坐标系下
	Eigen::Vector3d curr_point;             // 这个点是在LiDAR1坐标系下
	double weight;                          // 权重
	// 把三维空间点变换到雷达坐标系下，要求三维点尽可能靠近他所属的那个直线
	// 这里的curr_point是属于LiDAR1坐标系下的点, 也就是让LiDAR1坐标系靠近LiDAR2坐标系
	PairWisePoint2Line_Meter(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& line_point_a_,
					Eigen::Vector3d line_point_b_, double w_)
		: line_point(line_point_a_),curr_point(curr_point_), weight(w_) 
    {
        line_direction = (line_point_a_ - line_point_b_).normalized();
    }

	template <typename T>
	bool operator()(const T* const angleaxis_21, const T* const t_21, T *residual) const
	{
		T point[3] = {T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        T point_2[3];
        ceres::AngleAxisRotatePoint(angleaxis_21, point, point_2);
        point_2[0] += t_21[0];
        point_2[1] += t_21[1];
        point_2[2] += t_21[2];
		
        T line[6] = {T(line_point.x()), T(line_point.y()), T(line_point.z()),
                    T(line_direction.x()), T(line_direction.y()), T(line_direction.z())};
		residual[0] = T(weight) * PointToLineDistance3D(point_2, line);

		return true;
	}

	static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, 
										const Eigen::Vector3d line_point_a_,
									   const Eigen::Vector3d line_point_b_,
									   const double weight_ = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				PairWisePoint2Line_Meter, 1,3,3>(
			new PairWisePoint2Line_Meter(curr_point_ ,line_point_a_, line_point_b_, weight_)));
	}
};

// 一条直线在reference坐标系下，一条直线在neighbor坐标系下，两条直线相互匹配，那么这两条直线应该是平行的。
// 这一项就是优化两条直线之间的夹角，令夹角为0. 因为直线之间的夹角只涉及旋转，因此这一个残差项也只有旋转，没有平移
struct Line2Line_Angle
{
    Eigen::Vector3d line_direction_ref;         // reference 坐标系下的直线方向 
    Eigen::Vector3d line_direction_nei;         // neighbor 坐标系下的直线方向
    double weight;
    Line2Line_Angle(const Eigen::Vector3d& _ref_direction, const Eigen::Vector3d& _neighbor_direction, double _weight = 1):
        weight(_weight)
    {
        line_direction_ref = _ref_direction.normalized();
        line_direction_nei = _neighbor_direction.normalized();
    }
    template <typename T>
    bool operator()(const T* const angleaxis_rw, const T* const angleaxis_nw, T *residual) const
    {
        T angleaxis_wn[3] = {-angleaxis_nw[0], -angleaxis_nw[1], -angleaxis_nw[2]};
        T direction_nei[3] = {T(line_direction_nei.x()), T(line_direction_nei.y()), T(line_direction_nei.z())};
        // 先旋转到世界坐标系
        T direction_nei_world[3];
        ceres::AngleAxisRotatePoint(angleaxis_wn, direction_nei, direction_nei_world);
        // 再旋转到reference坐标系
        T direction_nei_ref[3];
        ceres::AngleAxisRotatePoint(angleaxis_rw, direction_nei_world, direction_nei_ref);
        T direction_ref[3] = {T(line_direction_ref.x()), T(line_direction_ref.y()), T(line_direction_ref.z())};
        // 这里用平面夹角的计算来代替两个向量的夹角，这是因为平面夹角是0-90度的，而向量夹角是0-180度的
        // 对于现在这种情况，两条直线之间的夹角只能是0-90度，180度的时候两条直线也算作平行
        residual[0] = PlaneAngle<T>(direction_nei_ref, direction_ref, true);
        if(residual[0] < T(1e-3))
            residual[0] = T(0.0);
        return true;
    }
    static ceres::CostFunction *Create(const Eigen::Vector3d& _ref_direction, 
                                    const Eigen::Vector3d& _neighbor_direction, 
                                    double _weight = 1)
	{
		return (new ceres::AutoDiffCostFunction<
				Line2Line_Angle, 1,3,3>(
            new Line2Line_Angle(_ref_direction ,_neighbor_direction, _weight)));
	}
};

#endif