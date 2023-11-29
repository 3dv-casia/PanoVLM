#include "MVS.h"

using namespace std;

mutex monitor;
int monitor_x, monitor_y;
vector<int> monitor_neighbor_ids;
bool update = false;
int half_window = 5;
int ncc_window_size = 2 * half_window + 1;
float half_window_sphere =half_window * 0.1 * M_PI / 180.f; 

void onMouse2(int event, int x, int y, int flags, void* data)
{
    if(event == cv::EVENT_LBUTTONDBLCLK)
    {
        // cv::Mat& best_neighbor = *(cv::Mat*)data;
        Frame& frame = *(Frame*)data;
        float conf = frame.conf_map.at<float>(y,x);
        const vector<pair<float,int>>& neighbor_score = frame.score_each_pixel.find(y * frame.GetImageCols() + x)->second;
        if(conf == -1 || neighbor_score.empty())
            return;
        vector<int> neighbor_ids;
        for(const auto& n : neighbor_score)
            neighbor_ids.push_back(n.second);
        LOG(INFO) << "x : " << x << " y: " << y << " neighbor : " << Join(neighbor_ids) << " conf : " << conf;
         
        lock_guard<mutex> guard(monitor);
        monitor_x = x;
        monitor_y = y;
        monitor_neighbor_ids = neighbor_ids;
        update = true;
    }
}

void ShowProject(const vector<Frame>& frames , const std::vector<std::vector<NeighborInfo>>& neighbors, int ref_id, const cv::Mat& ref_color)
{
    if(!update)
        return;
    {
        lock_guard<mutex> guard(monitor);
        update = false;
    }
    const Frame& ref_frame = frames[ref_id];
    Equirectangular eq(ref_frame.GetImageRows(), ref_frame.GetImageCols());
    
    cv::Point3f X0 = eq.ImageToCam(cv::Point2i(monitor_x, monitor_y)) * ref_frame.depth_map.at<float>(monitor_y, monitor_x);
    const cv::Vec3f& normal = ref_frame.normal_map.at<cv::Vec3f>(monitor_y, monitor_x);
    const float d = X0.dot(normal);
    const cv::Vec4f plane(normal(0), normal(1), normal(2), -d);
    for(size_t idx = 0; idx < monitor_neighbor_ids.size() && idx < 2; idx++)
    {
        int neighbor_idx = 0;
        while(neighbors[ref_id][neighbor_idx].id != monitor_neighbor_ids[idx])
            neighbor_idx++;
        const NeighborInfo& neighbor = neighbors[ref_id][neighbor_idx];
        const cv::Matx33f H = neighbor.R_nr + (1.f / d) * neighbor.t_nr * normal.t();
        cv::Mat img_project = frames[monitor_neighbor_ids[idx]].GetImageColor();
        for (int i = 0; i < ncc_window_size; i += 1)
        {
            for (int j = 0; j < ncc_window_size; j += 1)
            {
                // 根据H计算参考图像上当前点的近邻点在邻域图像上的投影
                // 这就是ComputePatchInfo 会导致变慢的地方
                // cv::Point3f X1 = H * eq.ImageToCam(cv::Point2i(monitor_x - half_window + j, monitor_y - half_window + i));
                cv::Point3f view_ray = eq.ImageToCam(cv::Point2i(monitor_x - half_window + j, monitor_y - half_window + i));
                const cv::Vec6f line(0,0,0, view_ray.x, view_ray.y, view_ray.z);
                cv::Point3f point_inersect = PlaneLineIntersect<float, cv::Point3f>(plane.val, line.val);
                cv::Point3f X1 = neighbor.R_nr * point_inersect + cv::Point3f(neighbor.t_nr);
                // if(isinf(X2.x) || isnan(X2.x))
                //     goto next_image;
                // if(!IsEqual((X1.x / X2.x), (X1.y / X2.y)) || !IsEqual((X1.x / X2.x), (X1.z / X2.z)))
                // {
                //     cout << "in ref " << ref_id << ", i = " << i << " j = " << j << ", unequal" << endl;
                // }
                cv::Point2f x1 = eq.CamToImage(X1);
                if(!frames[neighbor.id].IsInside(x1, 1, 1))
                    continue;
                cv::circle(img_project, x1, 1, cv::Scalar(0,0,255), 1);
            }
        }
        cv::Mat tmp = ref_color.clone();
        cv::rectangle(tmp, cv::Point2i(monitor_x - half_window, monitor_y - half_window), 
                        cv::Point2i(monitor_x + half_window, monitor_y + half_window), cv::Scalar(0,0,255), 2);
        cv::Mat img_patch = MVS::ExtractImagePatch(tmp, monitor_y, monitor_x, 50, 50);
        img_patch.copyTo(img_project.rowRange(0, img_patch.rows).colRange(0, img_patch.cols));
        cv::imshow("project_" + num2str(idx), img_project);
    }
}


void ShowProjectSphere(const vector<Frame>& frames , const std::vector<std::vector<NeighborInfo>>& neighbors, int ref_id, const cv::Mat& ref_color)
{
    if(!update)
        return;
    {
        lock_guard<mutex> guard(monitor);
        update = false;
    }
    const Frame& ref_frame = frames[ref_id];
    Equirectangular eq(ref_frame.GetImageRows(), ref_frame.GetImageCols());
    cv::Point2f pt_sphere = eq.ImageToSphere(cv::Point2f(monitor_x, monitor_y));
    cv::Point3f X0 = eq.ImageToCam(cv::Point2i(monitor_x, monitor_y)) * ref_frame.depth_map.at<float>(monitor_y, monitor_x);
    const cv::Vec3f& normal = ref_frame.normal_map.at<cv::Vec3f>(monitor_y, monitor_x);
    const float d = X0.dot(normal);
    const cv::Vec4f plane(normal(0), normal(1), normal(2), -d);
    for(size_t idx = 0; idx < monitor_neighbor_ids.size() && idx < 2; idx++)
    {
        int neighbor_idx = 0;
        while(neighbors[ref_id][neighbor_idx].id != monitor_neighbor_ids[idx])
            neighbor_idx++;
        const NeighborInfo& neighbor = neighbors[ref_id][neighbor_idx];
        const cv::Matx33f H = neighbor.R_nr + (1.f / d) * neighbor.t_nr * normal.t();
        cv::Mat img_project = frames[monitor_neighbor_ids[idx]].GetImageColor();
        for (int i = 0; i < ncc_window_size; i += 1)
        {
            for (int j = 0; j < ncc_window_size; j += 1)
            {
                // 根据H计算参考图像上当前点的近邻点在邻域图像上的投影
                // 这就是ComputePatchInfo 会导致变慢的地方
                // cv::Point3f X1 = H * eq.ImageToCam(cv::Point2i(monitor_x - half_window + j, monitor_y - half_window + i));
                // cv::Point3f view_ray = eq.ImageToCam(cv::Point2i(monitor_x - half_window + j, monitor_y - half_window + i));
                cv::Point3f view_ray =  eq.SphereToCam(
                            cv::Point2f(pt_sphere.x - half_window_sphere + j * 0.1 * M_PI / 180.f, 
                                    pt_sphere.y - half_window_sphere + i * 0.1 * M_PI / 180.f), 1.f);
                const cv::Vec6f line(0,0,0, view_ray.x, view_ray.y, view_ray.z);
                cv::Point3f point_inersect = PlaneLineIntersect<float, cv::Point3f>(plane.val, line.val);
                cv::Point3f X1 = neighbor.R_nr * point_inersect + cv::Point3f(neighbor.t_nr);
                // if(isinf(X2.x) || isnan(X2.x))
                //     goto next_image;
                // if(!IsEqual((X1.x / X2.x), (X1.y / X2.y)) || !IsEqual((X1.x / X2.x), (X1.z / X2.z)))
                // {
                //     cout << "in ref " << ref_id << ", i = " << i << " j = " << j << ", unequal" << endl;
                // }
                cv::Point2f x1 = eq.CamToImage(X1);
                if(!frames[neighbor.id].IsInside(x1, 1, 1))
                    continue;
                cv::circle(img_project, x1, 1, cv::Scalar(0,0,255), 1);
            }
        }
        cv::Mat tmp = ref_color.clone();
        cv::rectangle(tmp, cv::Point2i(monitor_x - half_window, monitor_y - half_window), 
                        cv::Point2i(monitor_x + half_window, monitor_y + half_window), cv::Scalar(0,0,255), 2);
        cv::Mat img_patch = MVS::ExtractImagePatch(tmp, monitor_y, monitor_x, 50, 50);
        img_patch.copyTo(img_project.rowRange(0, img_patch.rows).colRange(0, img_patch.cols));
        cv::imshow("project_" + num2str(idx), img_project);
    }
}

void TimeReport2( map<string, double> time_spent)
{
    double total_time = 0;
    map<string, double>::iterator it = time_spent.begin();
    for(; it != time_spent.end(); it++)
    {
        if(it->second > 0.001)
        {
            LOG(INFO) << it->first << " : " << it->second << " s" << endl;
            total_time += it->second;
        }
        else 
        {
            LOG(INFO) << it->first << " : 0 s" << endl;
        }
    }
    // LOG(INFO) << "total time: " << total_time << " s" << endl;
    LOG(INFO) << "======================================";
}


void MVS::test3(const std::set<int>& ids, const cv::Mat& mask)
{
    int ref_id = 1;
    Frame& ref_frame = frames[ref_id];
    if(false)
    {
        Initialize(ref_id, mask, config.propagate_strategy);
        EstimateDepthMapSingle(ref_id, config.propagate_strategy, 3, -0.7, false);
        pcl::io::savePCDFileBinary("depth_normal_" + num2str(ref_id) + "_before_filter.pcd", DepthNormalToCloud(ref_id, false));
        
        ExportFrameDepthAll(config.mvs_depth_path, frames, false);
        ExportFrameConfAll(config.mvs_conf_path, frames);
        ExportFrameNormalAll(config.mvs_normal_path, frames);
    }
    
    /* 读取参考图像和其他近邻图像的灰度图 */
    vector<int> frame_ids = {ref_id};
    {
        for(const NeighborInfo& n : neighbors[ref_id])
            frame_ids.push_back(n.id);
        #pragma omp parallel for
        for(const int& id : frame_ids)
        {
            if(frames[id].GetImageGrayRef().empty())
                frames[id].LoadImageGray(frames[id].name);
        }
    }

    
    ReadFrameDepthAll(config.mvs_depth_path, frames, "_pho.bin");
    ReadFrameConfAll(config.mvs_conf_path, frames, "_pho.bin");
    ReadFrameNormalAll(config.mvs_normal_path, frames, "_pho.bin");
    // ref_frame.depth_map *= 2.f;

    // 把参考图像根据深度投影到近邻图像上
    for(const NeighborInfo& n : neighbors[ref_id])
    {
        cv::Mat ref_color_projected = ProjectRGBToNeighbor(ref_id, n, false);
        cv::imwrite("project_" + num2str(ref_id) + "_to_" + num2str(n.id) + ".jpg", ref_color_projected);
        ref_color_projected.convertTo(ref_color_projected, CV_32FC3);
        cv::Mat nei_color;
        frames[n.id].GetImageColor().convertTo(nei_color, CV_32FC3);
        cv::Mat diff_color = ref_color_projected - nei_color;
        cv::Mat tmp = cv::Mat::zeros(ref_color_projected.size(), CV_8UC3);
        for(int row = 0; row < diff_color.rows; row++)
        {
            for(int col = 0; col < diff_color.cols; col++)
            {
                cv::Vec3f diff = diff_color.at<cv::Vec3f>(row, col);
                tmp.at<cv::Vec3b>(row, col) = Gray2Color(min(255, int(abs(diff(0)) + abs(diff(1)) + abs(diff(2)))));
            }
        }
        cv::imwrite("project_" + num2str(ref_id) + "_to_" + num2str(n.id) + "_diff.jpg", tmp);
    }    

    string project_folder = "./high_ncc_project/";
    if(!boost::filesystem::exists(project_folder))
        boost::filesystem::create_directories(project_folder);
    int count = 0, max_count = 100;
    cv::Mat ref_color = ref_frame.GetImageColor();
    for(int row = 0; row < ref_frame.GetImageRows(); row++)
    {
        for(int col = 0; col < ref_frame.GetImageCols(); col++)
        {
            const cv::Point2i pt(col, row);
            float& conf = ref_frame.conf_map.at<float>(row, col);
            vector<pair<float, int>> neighbor_score;
            const float& depth = ref_frame.depth_map.at<float>(row, col);
            if(depth <= 0)
                continue;
            const cv::Vec3f& normal = ref_frame.normal_map.at<cv::Vec3f>(row, col);
            cv::Point3f X0 = eq.ImageToCam(pt) * depth;
            cv::Vec4f plane(normal(0), normal(1), normal(2), -normal.dot(X0));
            conf = ScorePixelSphere(ref_id, cv::Point2i(col, row), normal, depth, neighbor_score, plane);
                   
            if(conf <= -1)
                continue;
            ref_frame.score_each_pixel[row * ref_frame.GetImageCols() + col] = neighbor_score;
        }
    }
    cv::imwrite("conf_" + num2str(ref_id) + ".jpg", DepthImageRGB(ref_frame.conf_map, 1));
    cv::namedWindow("conf_" + num2str(ref_id));
    cv::imshow("conf_" + num2str(ref_id), DepthImageRGB(ref_frame.conf_map, 1));
    cv::setMouseCallback("conf_" + num2str(ref_id), onMouse2, (void*)&ref_frame);
    while(true)
    {
        cv::waitKey(10);
        ShowProjectSphere(frames, neighbors, ref_id, ref_color);
    }
    
    return ;
}

void MVS::test2(const cv::Mat& mask)
{
    ResetFrameCount();
    #pragma omp parallel for schedule(dynamic)
    for(int ref_id = 0; ref_id < frames.size(); ref_id++)
    {
        if(!frames[ref_id].IsPoseValid())
            continue;
        InitDepthNormal(ref_id, mask, true);
        frames[ref_id].normal_map.release();
        frames[ref_id].depth_map.release();

    }
}

void MVS::test(const std::set<int>& ids, const cv::Mat& mask)
{
    omp_set_num_threads(config.num_threads);
    // #pragma omp parallel for
    for(const int& ref_id : vector<int>(ids.begin(), ids.end()))
    {
        Initialize(ref_id, mask, false, true, false);
        EstimateDepthMapSingle(ref_id, config.propagate_strategy, 2, -0.7, false);
        cv::imwrite(config.mvs_result_path + "conf_" + num2str(ref_id) + "_final.jpg", DepthImageRGB(frames[ref_id].conf_map));
        cv::imwrite(config.mvs_result_path + "normal_" + num2str(ref_id) + "_final.jpg", DrawNormalImage(frames[ref_id].normal_map, true));
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_final.jpg", DepthImageRGB(frames[ref_id].depth_map, config.max_depth_visual, config.min_depth)); 
    }
}

cv::Mat MVS::ExtractGroundPixel(const cv::Mat& ground_depth)
{
    cv::Mat ground_pixel;
    cv::threshold(ground_depth, ground_pixel, 0.1, 1, CV_THRESH_BINARY);
    ground_pixel.convertTo(ground_pixel, CV_8U);
    ground_pixel *= 255;
    
    cv::Mat kernel_7 = cv::Mat::ones(7,7, CV_8U);
    cv::morphologyEx(ground_pixel, ground_pixel, cv::MORPH_CLOSE, kernel_7, cv::Point2i(-1, -1), 2);

    return ground_pixel;
}

void MVS::test_ground(const std::set<int>& ids, const cv::Mat& mask)
{
    for(const int& ref_id : ids)
    {
        Frame& frame = frames[ref_id];
        Velodyne lidar(lidars[ref_id]);
        if(!frame.IsPoseValid() || !lidar.IsPoseValid())
            continue;
        Eigen::Matrix4d T_cl = frame.GetPose().inverse() * lidar.GetPose();
        lidar.LoadLidar();
        lidar.ReOrderVLP();
        pcl::PointCloud<PointType> ground_cloud, other_cloud;
        Eigen::Vector4f plane_coeff;
        if(!lidar.ExtractGroundPointCloud(ground_cloud, other_cloud, plane_coeff))
            continue;
        cv::imwrite(config.mvs_result_path + num2str(ref_id) + "_ground.jpg", ProjectLidar2PanoramaRGB(ground_cloud, frame.GetImageColor(), T_cl, config.min_depth, config.max_depth_visual));
        cv::imwrite(config.mvs_result_path + num2str(ref_id) + "_cloud.jpg", ProjectLidar2PanoramaRGB(lidar.cloud, frame.GetImageColor(), T_cl, config.min_depth, config.max_depth_visual));
        frame.depth_map = DepthCompletion(
                    ProjectLidar2PanoramaDepth(ground_cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);
        cv::imwrite(config.mvs_result_path + num2str(ref_id) + "_ground_completion.jpg", CombineDepthWithRGB(frame.depth_map, frame.GetImageColor(), config.max_depth_visual, config.min_depth));
        Eigen::Vector4f plane_coeff_camera = TranslePlane(plane_coeff, Eigen::Matrix4f(T_cl.cast<float>()));
        cv::Mat ground_pixel = ExtractGroundPixel(ProjectLidar2PanoramaDepth(ground_cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3));
        cv::Mat depth_ground = cv::Mat::zeros(frame.GetImageRows(), frame.GetImageCols(), CV_32F);
        for(int row = 0; row < frame.GetImageRows(); row ++)
        {
            for(int col = 0; col < frame.GetImageCols(); col++)
            {
                if(ground_pixel.at<uchar>(row, col) == 0)
                    continue;
                cv::Point3f X0 = eq.ImageToCam(cv::Point2i(col, row));
                Vector6f view_ray(0, 0, 0, X0.x, X0.y, X0.z);
                Eigen::Vector3f p = PlaneLineIntersect<float, Eigen::Vector3f>(plane_coeff_camera.data(), view_ray.data());
                if(isinf(p.x()) )
                    continue;
                depth_ground.at<float>(row, col) = p.norm();
            }
        }
        cv::imwrite(config.mvs_result_path + num2str(ref_id) + "_ground_depth.jpg", CombineDepthWithRGB(depth_ground, frame.GetImageColor(), config.max_depth_visual, config.min_depth));
        
        
        frame.depth_map = depth_ground;
        pcl::io::savePCDFileBinary(config.mvs_result_path + num2str(ref_id) + "_depth_ground.pcd", DepthImageToCloud(ref_id, false));

        frame.depth_map = DepthCompletion(
            ProjectLidar2PanoramaDepth(lidar.cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);

        pcl::io::savePCDFileBinary(config.mvs_result_path + num2str(ref_id) + "_depth_comp.pcd", DepthImageToCloud(ref_id, false));
        frame.depth_map = DepthCompletion(
            ProjectLidar2PanoramaDepth(other_cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);
        for(int row = 0; row < frame.GetImageRows(); row ++)
        {
            for(int col = 0; col < frame.GetImageCols(); col++)
            {
                if(ground_pixel.at<uchar>(row, col) == 0)
                    continue;
                frame.depth_map.at<float>(row, col) = depth_ground.at<float>(row, col);
            }
        }
        pcl::io::savePCDFileBinary(config.mvs_result_path + num2str(ref_id) + "_depth_combined.pcd", DepthImageToCloud(ref_id, false));

        pcl::PointCloud<PointType> tmp;
        pcl::transformPointCloud(lidar.cloud, tmp, T_cl.cast<float>());
        pcl::io::savePCDFileBinary(config.mvs_result_path + num2str(ref_id) + "_cloud.pcd", tmp);
        pcl::transformPointCloud(ground_cloud, tmp, T_cl.cast<float>());
        pcl::io::savePCDFileBinary(config.mvs_result_path + num2str(ref_id) + "_ground.pcd", tmp);
    }
}

void MVS::FuseLidarDepth(const std::set<int>& ids, const cv::Mat& mask)
{
    bool enable_parallel = (config.propagate_strategy == Propagate::SEQUENTIAL);
    set<int> frame_with_depth(ids), frame_depth_filter(ids);
    for(const auto& id : frame_depth_filter)
        for(const auto& neighbor : neighbors[id])
            frame_with_depth.insert(neighbor.id);
    #pragma omp parallel for
    for(int ref_id : vector<int>(frame_with_depth.begin(), frame_with_depth.end()))
    {
        Frame& frame = frames[ref_id];
        Velodyne lidar(lidars[ref_id]);
        lidar.LoadLidar(lidar.name);
        if(!frame.IsPoseValid() || !lidar.IsPoseValid())
            continue;
        /* 初始化深度图 */
        Eigen::Matrix4d T_cl = frame.GetPose().inverse() * lidar.GetPose();
        #if 1
        frame.depth_map = DepthCompletion(
                    ProjectLidar2PanoramaDepth(lidar.cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3), config.max_depth);
        #else 
        frame.depth_map = ProjectLidar2PanoramaDepth(lidar.cloud, frame.GetImageRows(), frame.GetImageCols(), T_cl, 3);
        frame.depth_map.convertTo(frame.depth_map, CV_32F);
        frame.depth_map /= 256.f;
        #endif 
        frame.depth_map = frame.depth_map.mul(mask);
        frame.conf_map = cv::Mat::zeros(frame.depth_map.size(), CV_32F);
        frame.normal_map = cv::Mat::zeros(frame.depth_map.size(), CV_32FC3);

        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_init.jpg", 
                DepthImageRGB(frames[ref_id].depth_map, config.max_depth_visual, config.min_depth)); 
        RemoveSmallSegments(ref_id);
        GapInterpolation(ref_id);
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_final.jpg", 
                DepthImageRGB(frames[ref_id].depth_map, config.max_depth_visual, config.min_depth));
    }
    #pragma omp parallel for
    for(int ref_id : vector<int>(frame_depth_filter.begin(), frame_depth_filter.end()))
    {
        if(!frames[ref_id].IsPoseValid())
            continue;
        FilterDepthImage(ref_id);
        cv::imwrite(config.mvs_result_path + "depth_" + num2str(ref_id) + "_filter.jpg", 
                DepthImageRGB(frames[ref_id].depth_filter, config.max_depth_visual, config.min_depth)); 

    }
    pcl::io::savePCDFileBinary(config.mvs_result_path + "MVS-merge.pcd", MergeDepthImages(2));

}

cv::Mat MVS::ExtractImagePatch(const cv::Mat& src_image, const int row, const int col, int half_row, int half_col)
{
    cv::Mat image_patch = cv::Mat::zeros(2 * half_row + 1, 2 * half_col + 1, src_image.type());
    int src_row_start = max(0, row - half_row);
    int src_row_end = min(src_image.rows, row + half_row);      // _end 是不包含在索引范围内的
    int src_col_start = max(0, col - half_col);
    int src_col_end = min(src_image.cols, col + half_col);
    int patch_row_start = (row - half_row < 0 ? half_row - row : 0);
    int patch_col_start = (col - half_col < 0 ? half_col - col : 0);
    src_image.rowRange(src_row_start, src_row_end).colRange(src_col_start, src_col_end).copyTo(
                image_patch.rowRange(patch_row_start, patch_row_start + src_row_end - src_row_start).colRange(patch_col_start, patch_col_start + src_col_end - src_col_start));
    return image_patch;
}


cv::Mat MVS::ProjectRGBToNeighbor(int ref_id, const NeighborInfo& info, bool use_filtered_depth)
{
    const Frame& frame_ref = frames[ref_id];
    const Frame& frame_nei = frames[info.id];
    const cv::Mat& img_depth = (use_filtered_depth ? frames[ref_id].depth_filter : frames[ref_id].depth_map);
    cv::Mat img_color = frames[ref_id].GetImageColor();
    cv::Mat img_project = cv::Mat::zeros(frames[ref_id].GetImageRows(), frames[ref_id].GetImageCols(), CV_8UC3);
    vector<cv::Point2i> pts(4);
    for(int row = 0; row < frames[ref_id].GetImageRows(); row++)
    {
        for(int col = 0; col < frames[ref_id].GetImageCols(); col++)
        {
            const float depth = img_depth.at<float>(row, col);
            if(depth <= 0 || depth >= config.max_depth)
                continue;
            cv::Point3f point_ref = eq.ImageToCam(cv::Point2i(col, row)) * depth;
            cv::Point3f point_neighbor = info.R_nr * point_ref + cv::Point3f(info.t_nr);
            cv::Point2f point_pixel = eq.CamToImage(point_neighbor);
            const cv::Vec3b color = img_color.at<cv::Vec3b>(row, col);
            pts[0] = cv::Point2i(ceil(point_pixel.x), ceil(point_pixel.y));
            pts[1] = cv::Point2i(ceil(point_pixel.x), floor(point_pixel.y));
            pts[2] = cv::Point2i(floor(point_pixel.x), floor(point_pixel.y));
            pts[3] = cv::Point2i(floor(point_pixel.x), ceil(point_pixel.y));
            for(cv::Point2i& pt : pts)
            {
                if(frame_nei.IsInside(pt))
                    img_project.at<cv::Vec3b>(pt) = color;
            }
        }
    }
    return img_project;
}

void MVS::PrintPatch(const PixelPatch& patch)
{
    int rows = sqrt(num_texels);
    LOG(INFO) << "texels0 :";
    for(int i = 0; i < rows; i++)
    {
        vector<float> values;
        for(int j = 0; j < rows; j++)
            values.push_back(patch.texels0[i * rows + j]);
        LOG(INFO) << Join(values);
    }
    LOG(INFO) << "weight :";
    for(int i = 0; i < rows; i++)
    {
        vector<float> values;
        for(int j = 0; j < rows; j++)
            values.push_back(patch.weight[i * rows + j]);
        LOG(INFO) << Join(values);
    }
    LOG(INFO) << "sq0 : " << patch.sq0;
}