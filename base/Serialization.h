/*
 * @Author: Diantao Tu
 * @Date: 2021-12-13 12:25:03
 */

#ifndef _SERIALIZATION_H_
#define _SERIALIZATION_H_

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <Eigen/Core>

namespace boost {
namespace serialization {


template<class Archive>
void serialize(Archive &ar, cv::Point_<float>& pt2, const unsigned int)
{
    ar & pt2.x;
    ar & pt2.y;
}

template<class Archive>
void serialize(Archive &ar, cv::Vec<float,6>& vec6, const unsigned int)
{
    ar & vec6[0];
    ar & vec6[1];
    ar & vec6[2];
    ar & vec6[3];
    ar & vec6[4];
    ar & vec6[5];
}

template<class Archive>
void serialize(Archive &ar, cv::Vec<float,4>& vec4, const unsigned int)
{
    ar & vec4[0];
    ar & vec4[1];
    ar & vec4[2];
    ar & vec4[3];
}

template<class Archive>
void serialize(Archive &ar, cv::KeyPoint& kpt, const unsigned int version)
{
    ar & kpt.pt;
    ar & kpt.angle;
    ar & kpt.class_id;
    ar & kpt.octave;
    ar & kpt.size;
    ar & kpt.response;
}

template<class Archive>
void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
{
    int cols, rows, type;
    bool continuous;

    if (Archive::is_saving::value) {
        cols = mat.cols; rows = mat.rows; type = mat.type();
        continuous = mat.isContinuous();
    }

    ar & cols & rows & type & continuous;

    if (Archive::is_loading::value)
        mat.create(rows, cols, type);

    if (continuous) {
        const unsigned int data_size = rows * cols * mat.elemSize();
        ar & boost::serialization::make_array(mat.ptr(), data_size);
    } else {
        const unsigned int row_size = cols*mat.elemSize();
        for (int i = 0; i < rows; i++) {
            ar & boost::serialization::make_array(mat.ptr(i), row_size);
        }
    }
}

template<class Archive>
void serialize(Archive &ar, Eigen::Matrix3d& mat, const unsigned int)
{
    ar & mat(0,0);
    ar & mat(0,1);
    ar & mat(0,2);
    ar & mat(1,0);
    ar & mat(1,1);
    ar & mat(1,2);
    ar & mat(2,0);
    ar & mat(2,1);
    ar & mat(2,2);
}

template<class Archive>
void serialize(Archive &ar, Eigen::Vector3d& vec, const unsigned int)
{
    ar & vec(0);
    ar & vec(1);
    ar & vec(2);
}

template<class Archive>
void serialize(Archive &ar, Eigen::Vector3i& vec, const unsigned int)
{
    ar & vec(0);
    ar & vec(1);
    ar & vec(2);
}

template<class Archive>
void serialize(Archive &ar, cv::DMatch& match, const unsigned int)
{
    ar & match.queryIdx;
    ar & match.trainIdx;
    ar & match.distance;
}

template<class Archive>
void serialize(Archive &ar, cv::line_descriptor::KeyLine& kl, const unsigned int)
{
    ar & kl.angle;
    ar & kl.class_id;
    ar & kl.startPointX;
    ar & kl.startPointY;
    ar & kl.endPointX;
    ar & kl.endPointY;
    ar & kl.sPointInOctaveX;
    ar & kl.sPointInOctaveY;
    ar & kl.ePointInOctaveX;
    ar & kl.ePointInOctaveY;
    ar & kl.lineLength;
    ar & kl.numOfPixels;
    ar & kl.octave;
    ar & kl.pt;
    ar & kl.response;
    ar & kl.size;
}

}
}


#endif