/*
 * @Author: Diantao Tu
 * @Date: 2022-04-30 09:34:47
 */
#include "VLAD.h"

using namespace std;

VLADMatcher::VLADMatcher(const std::vector<Frame>& _frames, const Config& _config,const int _normalization_type):
            frames(_frames),normalization_type(_normalization_type),config(_config)
{}


bool VLADMatcher::GenerateCodeBook(float ratio, const int book_size, const int max_iteration)
{
    if(ratio > 1 || ratio < 0)
    {
        LOG(ERROR) << "learning ratio must between 0 and 1";
        return false;
    }
    LOG(INFO) << "learning ratio is " << ratio;
    // 所有图像的描述子竖直拼接，组成一个大的矩阵，矩阵的每一行是一个描述子
    cv::Mat descriptor_array(0, 128, CV_32F);
    vector<size_t> frame_for_codebook(frames.size());
    iota(frame_for_codebook.begin(), frame_for_codebook.end(), 0);
    if(ratio < 1)
        frame_for_codebook = CreateRandomArray(static_cast<size_t>(ratio * frames.size()), frame_for_codebook);
    LOG(INFO) << "concat frame descriptors";
    // 为了加速描述子的拼接，首先分配内存，然后再把数据都拷贝过去
    size_t descriptor_array_length = 0;
    for(const size_t& frame_id : frame_for_codebook)
        descriptor_array_length += frames[frame_id].GetDescriptor().rows;
    descriptor_array.resize(descriptor_array_length);
    descriptor_array_length = 0;
    for(const size_t& frame_id : frame_for_codebook)
    {
        size_t length = frames[frame_id].GetDescriptor().rows;
        frames[frame_id].GetDescriptor().copyTo(descriptor_array.rowRange(descriptor_array_length, descriptor_array_length + length));
        descriptor_array_length += length;
    }
    vector<size_t> data_to_center;
    codebook = Kmeans(descriptor_array, book_size, max_iteration, data_to_center);
    return true;
}

cv::Mat VLADMatcher::Kmeans(const cv::Mat& data, const size_t num_center, const int max_iteration, std::vector<size_t>& data_to_center)
{
    LOG(INFO) << "Use K-means to generate codebook";
    // 随机初始化出聚类中心
    cv::Mat centers(0, 128, CV_32F);
    for(const size_t& id : CreateRandomArray(num_center, size_t(0), size_t(data.rows)))
        cv::vconcat(centers, data.row(id), centers);
    
    omp_set_num_threads(config.num_threads);
    data_to_center.resize(data.rows, 0);
    bool changed = true;
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
    for(int iter = 0; iter < max_iteration && changed; iter++)
    {
        changed = false;
        // 计算每个点到每个聚类中心的距离，并且找出距离最近的那个中心。这里用来衡量距离的是L2范数，
        // 所以可以直接用opencv的BFMatcher来得到最近的距离，代码写起来简单而且运行速度比自己实现的要
        // 快得多, 大概快了5倍左右吧
        #pragma omp parallel for shared(changed)
        for(size_t data_id = 0; data_id < data.rows; data_id ++)
        {
            vector<cv::DMatch> matches;
            matcher->match(data.row(data_id), centers, matches);
            size_t nearest_center = matches[0].trainIdx;
            // 更新距离当前点最近的聚类中心
            if(nearest_center != data_to_center[data_id])
            {
                data_to_center[data_id] = nearest_center;
                changed = true;
            }
        }
        // 计算新的聚类中心
        vector<vector<size_t>> center_to_data(num_center);
        for(size_t data_id = 0; data_id < data.rows; data_id ++)
            center_to_data[data_to_center[data_id]].push_back(data_id);
        #pragma omp parallel for
        for(size_t center_id = 0; center_id < num_center; center_id++)
        {
            cv::Mat center = cv::Mat::zeros(1, 128, CV_32F);
            for(const size_t& data_id : center_to_data[center_id])
                center += data.row(data_id);
            center /= center_to_data[center_id].size();
            #pragma omp critical
            {
                center.copyTo(centers.row(center_id));
            }
        }
    }
    return centers;
}

bool VLADMatcher::ComputeVLADEmbedding()
{
    LOG(INFO) << "Compute VLAD for each frame";
    if(frames.empty() || codebook.empty())
        return false;
    // 每张图像有一个vlad描述子，这个描述子的维度是 n * d维，其中n代表特征点描述子的维度，比如SIFT就是n=128，d是codebook的
    // 大小，也就是codebook有多少个单词
    vlad_descriptors = cv::Mat::zeros(frames.size(), 128 * codebook.rows, CV_32F);
    #pragma omp parallel for schedule(dynamic)
    for(size_t frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        cv::Mat vlad_desc = cv::Mat::zeros(1, 128 * codebook.rows, CV_32F);
        vector<cv::DMatch> matches; 
        const cv::Mat& descriptor = frames[frame_id].GetDescriptor();
        // 使用暴力匹配，图像上的每个特征点都要找到一个单词与之相对应
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
        matcher->match(descriptor, codebook, matches);
        // 计算图像上特征点和对应的单词之间的距离，并且进行正则化
        for(const cv::DMatch& m : matches)
        {
            cv::Mat residual = descriptor.row(m.queryIdx) - codebook.row(m.trainIdx);
            if(normalization_type == RESIDUAL_NORMALIZATION_PWR_LAW)
                residual = residual / cv::norm(residual);
            vlad_desc.colRange(128 * m.trainIdx, 128 * (m.trainIdx + 1)) += residual;
        }
        // 将每个单词对应的描述子独立进行正则化
        for(size_t center_id = 0; center_id < codebook.rows; center_id++)
        {
            size_t center_start = center_id * 128;
            size_t center_end = (center_id + 1) * 128;
            if(normalization_type == INTRA_NORMALIZATION)
                vlad_desc.colRange(center_start, center_end) /= cv::norm(vlad_desc.colRange(center_start, center_end));
            else if(normalization_type == SIGNED_SQUARE_ROOTING)
            {
                for(size_t i = center_start; i < center_end ; i ++)
                {
                    float val = vlad_desc.at<float>(0, i);
                    vlad_desc.at<float>(0,i) = val > 0 ? sqrt(val) : -sqrt(-val);
                }
            }
            else if(normalization_type == RESIDUAL_NORMALIZATION_PWR_LAW)
            {
                for(size_t i = center_start; i < center_end ; i ++)
                {
                    float val = vlad_desc.at<float>(0, i);
                    vlad_desc.at<float>(0,i) = val > 0 ? pow(val, 0.2) : -pow(-val, 0.2);
                }
            }
        }
        // 整体进行正则化，并保存下来
        vlad_desc /= cv::norm(vlad_desc);
        #pragma omp critical
        {
            vlad_desc.copyTo(vlad_descriptors.row(frame_id));
        }
    }
    return true;
}

std::vector<std::vector<size_t>> VLADMatcher::FindNeighbors(int neighbor_size)
{
    std::vector<std::vector<size_t>> neighbors_all(frames.size());
    #pragma omp parallel for schedule(dynamic)
    for(size_t frame_id = 0; frame_id < frames.size(); frame_id++)
    {
        // 当前frame到其他frame的距离，pair.first = 距离，pair.second = 其他frame的idx
        vector<pair<double, size_t>> distance_to_frames;
        const cv::Mat& curr_descriptor = vlad_descriptors.row(frame_id);
        // 暴力匹配，计算描述子之间的差异，这里用的是向量内积，由于内积是余弦(cos)，夹角越小对应的余弦越大，描述子相似度越高。
        // 因此使用内积结果作为相似度的衡量标准很合适，相似度越高内积也越大。对于每张图像，只需要找相似度最高的n个图像即可。
        // 这就需要排序，但排序默认是从小到大排序的，
        // 所以加上负号，这样就符合现实了，两个向量越是接近，距离就越小
        for(size_t row = 0; row < vlad_descriptors.rows; row++)
        {
            double dist = -curr_descriptor.dot(vlad_descriptors.row(row));
            distance_to_frames.push_back({dist, row});
        }
        sort(distance_to_frames.begin(), distance_to_frames.end());
        vector<size_t> neighbors;
        for(size_t i = 0; i < neighbor_size && i < frames.size(); i++)
        {
            neighbors.push_back(distance_to_frames[i].second);
        }
        neighbors_all[frame_id] = neighbors;
    }
    return neighbors_all;
}