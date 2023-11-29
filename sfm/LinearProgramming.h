/*
 * @Author: Diantao Tu
 * @Date: 2021-12-23 19:30:38
 */

#ifndef _LINEAR_PROGRAMMING_H_
#define _LINEAR_PROGRAMMING_H_

#include <OsiClpSolverInterface.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <glog/logging.h>
#include <memory>

enum{
    LP_LESS_OR_EQUAL    = 1,  // (<=)
    LP_GREATER_OR_EQUAL = 2,  // (>=)
    LP_EQUAL            = 3   // (=)
};

struct LPConstrain
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    bool minimize;
    std::vector<int> sign;
    std::vector<double> C;
    std::vector<double> costs;
    std::vector<std::pair<double,double>> bounds;
    size_t num_vars;

    
};

class LPSolver
{
private:
    // std::shared_ptr<OsiClpSolverInterface> si;
    OsiClpSolverInterface si;
public:
    LPSolver(/* args */);
    bool Setup(const LPConstrain& constrains);
    bool Solve();
    bool GetSolution(std::vector<double>& solution);
    ~LPSolver();
};





#endif