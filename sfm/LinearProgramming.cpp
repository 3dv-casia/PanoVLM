/*
 * @Author: Diantao Tu
 * @Date: 2021-12-23 19:38:06
 */
#include "LinearProgramming.h"

using namespace std;


LPSolver::LPSolver(/* args */)
{
    // si.reset(new OsiClpSolverInterface);
    // si->setLogLevel(0);
    si.reset();
    si.setLogLevel(0);
}

bool LPSolver::Setup(const LPConstrain& constrains)
{
    const size_t num_vars = constrains.A.cols();
    vector<double> col_lower_bound(num_vars), col_upper_bound(num_vars);
    // 设置为最小化问题
    si.setObjSense((constrains.minimize) ? 1 : -1);
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A = constrains.A;
    // 设置等式约束，由于API限制，等式约束必须用两个约束来实现，也就是同时满足 >= 和 <=
    const size_t lines = A.rows() + count(constrains.sign.begin(), constrains.sign.end(), LP_EQUAL);
    // 设置约束,默认是无穷（无约束）
    vector<double> row_lower_bound(lines, -si.getInfinity()), row_upper_bound(lines, si.getInfinity());
    unique_ptr<CoinPackedMatrix> matrix(new CoinPackedMatrix(false, 0, 0));
    matrix->setDimensions(0, num_vars);

    LOG(INFO) << "Matrix A is " << (A.IsRowMajor ? "row major" : "col major") << " ,rows: " << A.rows() << " cols: " << A.cols();


    // 开始设置约束
    size_t row_idx = 0;
    for(size_t i = 0; i < A.rows(); i++)
    {
        vector<int> col_idx;
        vector<double> value;
        for(Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it)
        {
            col_idx.push_back(it.col());
            value.push_back(it.value());
        }
        if(constrains.sign[i] == LP_EQUAL || constrains.sign[i] == LP_LESS_OR_EQUAL)
        {
            row_upper_bound[row_idx] = constrains.C[i];
            matrix->appendRow(col_idx.size(), &col_idx[0], &value[0]);
            row_idx++;
        }
        if(constrains.sign[i] == LP_EQUAL || constrains.sign[i] == LP_GREATER_OR_EQUAL)
        {
            for(double& val:value)
                val *= -1;
            row_upper_bound[row_idx] = - constrains.C[i];
            matrix->appendRow(col_idx.size(), &col_idx[0], &value[0]);
            row_idx++;
        }
    }

    for(size_t i = 0; i < num_vars; i++)
    {
        col_lower_bound[i] = constrains.bounds[i].first;
        col_upper_bound[i] = constrains.bounds[i].second;
    }
    
    si.loadProblem(*matrix, &col_lower_bound[0], &col_upper_bound[0], constrains.costs.empty() ? nullptr : &constrains.costs[0],
                    &row_lower_bound[0], &row_upper_bound[0]);
    return true;
}

bool LPSolver::Solve()
{
    si.getModelPtr()->setPerturbation(50);
    si.initialSolve();
    return si.isProvenOptimal();
}

bool LPSolver::GetSolution(std::vector<double> & solutions)
{
    const int n = si.getNumCols();
    solutions.resize(n);
    memcpy(&solutions[0], si.getColSolution(), n * sizeof(double));
    return true;
}

LPSolver::~LPSolver()
{
}