/*
 * @Author: Diantao Tu
 * @Date: 2021-11-21 20:26:40
 */

#include "SIFT.h"

using namespace std;

void RootSIFT(cv::Mat& descriptor)
{
    // For each row
    for (int i = 0; i < descriptor.rows; ++i) 
    {
        // Perform L1 normalization
        cv::normalize(descriptor.row(i), descriptor.row(i), 1.0, 0.0, cv::NORM_L1);
    }
    // Perform sqrt on the whole descriptor matrix
    cv::sqrt(descriptor, descriptor);
    descriptor *= 512.f;
}

bool ExtractSIFT(const cv::Mat& img_gray, vector<cv::KeyPoint>& keypoints, 
                const int num_sift, const cv::Mat& mask)
{
    if(num_sift <= 0)
        return false;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(num_sift);
    detector->detect(img_gray, keypoints, mask);
    return keypoints.size() > 0;
}

bool DistributeSIFTQuadtree(const cv::Mat& img_gray, vector<cv::KeyPoint>& keypoints, 
                const int num_sift, const cv::Mat& mask)
{
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(num_sift * 1.2);
    detector->detect(img_gray, keypoints, mask);
    
    vector<Cell> cells;
    cells.reserve(num_sift / 2);
    // 先放入第一个cell，也就是整张图像
    cells.push_back(Cell(0, 0, img_gray.cols - 1, img_gray.rows - 1));
    vector<int> keypoint_ids(keypoints.size());
    std::iota(keypoint_ids.begin(), keypoint_ids.end(), 0);
    cells[0].keypoint_ids = keypoint_ids;
    int valid_cell_count = 1;
    size_t cell_start_idx = 0, cell_end_idx = 0;
    // 这里退出循环有两个条件，达到任一条件即退出，第一个条件是分割的四叉树已经达到目标要求，
    // 第二个条件是所有的cell都无法再分了，在这种情况下，start idx会每次增加一，而end idx一直不变，那么
    // start idx == end idx 的时候就代表了所有cell已经无法再分割了
    for(; valid_cell_count <= num_sift && cell_start_idx <= cell_end_idx; cell_start_idx++)
    {
        vector<Cell> sub_cells = cells[cell_start_idx].SplitIntoFour(keypoints);
        if(sub_cells.empty())    
            continue;
        valid_cell_count--;
        cells.insert(cells.end(), sub_cells.begin(), sub_cells.end());
        for(const Cell& c : sub_cells)
            valid_cell_count += (c.valid);
        cell_end_idx += 4;
    }
    // 取出每个cell里响应最大的特征点
    vector<cv::KeyPoint> keypoints_cell;
    for(const Cell& c : cells)
    {
        if(c.valid == false)
            continue;
        cv::KeyPoint best_keypoint;
        best_keypoint.response = 0;
        for(const int& id : c.keypoint_ids)
        {
            if(keypoints[id].response > best_keypoint.response)
            {
                best_keypoint = keypoints[id];
            }
        }
        keypoints_cell.push_back(best_keypoint);
    }
    keypoints_cell.swap(keypoints);

    return true;
}

bool ExtractSIFTQuadtree(const cv::Mat& img_gray, vector<cv::KeyPoint>& keypoints, const int tree_depth,
                const int num_sift, const cv::Mat& mask)
{
    assert(tree_depth >= 0);
    vector<vector<Cell>> cell_each_depth(tree_depth + 1);     // 每一层四叉树的cell
    cell_each_depth[0].push_back(Cell(0, 0, img_gray.cols - 1, img_gray.rows - 1));     // 第一层的cell就是整张图像
    for(int i = 1; i <= tree_depth; i++)
    {
        for(Cell& c : cell_each_depth[i - 1])
        {
            vector<Cell> sub_cell = c.SplitIntoFour(vector<cv::KeyPoint>());
            cell_each_depth[i].insert(cell_each_depth[i].end(), sub_cell.begin(), sub_cell.end());
        }
        for(Cell& c : cell_each_depth[i])
            c.valid = true;
    }
    // 计算mask里非零元素的个数，这样可以确定在每个cell里应该提取多少个特征点
    int mask_count = cv::countNonZero(mask);
    for(const Cell& c : cell_each_depth[tree_depth])
    {
        int row_start = c.left_top.y, row_end = c.right_bottom.y, col_start = c.left_top.x, col_end = c.right_bottom.x;
        cv::Mat sub_image = img_gray.rowRange(row_start, row_end + 1).colRange(col_start, col_end + 1);
        cv::Mat sub_mask = mask.rowRange(row_start, row_end + 1).colRange(col_start, col_end + 1);
        int sub_mask_count = cv::countNonZero(sub_mask);
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create(num_sift * (float)sub_mask_count / mask_count);
        vector<cv::KeyPoint> sub_key_points;
        detector->detect(sub_image, sub_key_points, sub_mask);
        for(cv::KeyPoint& kpt : sub_key_points)
        {
            kpt.pt.x += col_start;
            kpt.pt.y += row_start;
        }
        keypoints.insert(keypoints.end() ,sub_key_points.begin(), sub_key_points.end());
    }
    return true;
}

bool ComputeSIFTDescriptor(const cv::Mat& img_gray, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor,const bool root_sift)
{
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(keypoints.size());
    detector->compute(img_gray, keypoints, descriptor);
    if(root_sift)
        RootSIFT(descriptor);
    return descriptor.rows > 0;
}

vector<cv::DMatch> MatchSIFT(const cv::Mat& descriptor1, const cv::Mat& descriptor2, const float dist_ratio_threshold)
{
    if(descriptor1.rows == 0 || descriptor2.rows == 0)
        return vector<cv::DMatch>();
    vector<vector<cv::DMatch>> raw_matches;
    cv::Ptr<cv::FlannBasedMatcher> flann_mather = cv::FlannBasedMatcher::create();
    flann_mather->knnMatch(descriptor1, descriptor2, raw_matches, 2);
    vector<cv::DMatch> matches;
    for(size_t i = 0; i < raw_matches.size(); i++)
    {
        if(raw_matches[i][0].distance < dist_ratio_threshold * raw_matches[i][1].distance)
            matches.push_back(raw_matches[i][0]);
    }
    return matches;
}

#ifdef USE_CUDA
vector<cv::DMatch> MatchSIFT(const cv::cuda::GpuMat& descriptor1, const cv::cuda::GpuMat& descriptor2, const float dist_ratio_threshold)
{
    if(descriptor1.rows == 0 || descriptor2.rows == 0)
        return vector<cv::DMatch>();
    vector<vector<cv::DMatch>> raw_matches;
    cv::Ptr<cv::cuda::DescriptorMatcher> bf_matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    bf_matcher->knnMatch(descriptor1, descriptor2, raw_matches, 2);
    vector<cv::DMatch> matches;
    // 在相同的距离阈值设置下，Bruce force得到的匹配点会比 FLANN少一些，但是准确率会更高一些
    for(size_t i = 0; i < raw_matches.size(); i++)
    {
        if(raw_matches[i][0].distance < dist_ratio_threshold * raw_matches[i][1].distance)
            matches.push_back(raw_matches[i][0]);
    }
    return matches;
}
#endif