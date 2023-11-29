/*
 * @Author: Diantao Tu
 * @Date: 2022-04-30 16:17:01
 */
#ifndef _BATA_H_
#define _BATA_H_

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <glog/logging.h>
#include "../util/Visualization.h"
#include "../base/ProcessBar.h"
#include "../base/common.h"

struct BATAConfig
{
    double delta = 1e-6;
    int init_iteration = 10;
    int inner_iteration = 10;
    int outer_iteration = 10;
    double robust_threshold = 0.1;
};

eigen_vector<Eigen::Vector3d> BATA(const vector<pair<int,int>>& pairs, const eigen_vector<Eigen::Vector3d>& relative_pose, const BATAConfig& config, const string& output_path);




#endif