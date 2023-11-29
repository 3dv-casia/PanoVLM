/*
 * @Author: Diantao Tu
 * @Date: 2022-08-11 19:35:53
 */
#include "SfM.h"

bool SfM::ReComputePairs(size_t idx1, size_t idx2)
{
    idx2 = min(idx2, frames.size() - 1);
    assert(idx1 <= idx2);
    vector<MatchPair> pairs_preserve;
    for(const MatchPair& p : image_pairs)
    {
        if(IsInside(p.image_pair.first, idx1, idx2) && IsInside(p.image_pair.second, idx1, idx2))
            continue;
        pairs_preserve.push_back(p);
    }

    image_pairs.clear();
    for(size_t a = idx1; a <= idx2; a++)
        for(size_t b = a + 1; b <= idx2; b++)
            image_pairs.push_back(MatchPair(a, b));

    for(size_t i = idx1; i <= idx2; i++)
    {
        frames[i].LoadImageGray(frames[i].name);
        frames[i].ComputeDescriptor(config.root_sift);
    }

    MatchImagePairs(config.sift_match_num_threshold);
    FilterImagePairs(24, 3, config.keep_pairs_no_scale);
    for(size_t i = idx1; i <= idx2; i++)
    {
        frames[i].ReleaseImageGray();
        frames[i].ReleaseDescriptor();
    }

    pairs_preserve.insert(pairs_preserve.end(), image_pairs.begin(), image_pairs.end());
    image_pairs.swap(pairs_preserve);
    return true;
}

bool SfM::FilterByStraightMotion(int motion_duration)
{
    vector<MatchPair> good_pairs;
    for(const MatchPair& pair : image_pairs)
    {
        // if(pair.image_pair.second - pair.image_pair.first > 2 * 5 && 
        //     pair.image_pair.second - pair.image_pair.first < 300)
        //     continue;
        if(pair.image_pair.second - pair.image_pair.first < motion_duration)
        {
            Eigen::Vector3d t_12 = -pair.R_21.transpose() * pair.t_21;
            if(t_12.z() > 0)
                continue;
        }
        
        good_pairs.push_back(pair);
    }
    LOG(INFO) << "filter " << image_pairs.size() - good_pairs.size() << " pairs by straight motion";
    image_pairs = good_pairs;
    return true;
}

bool SfM::FilterByIndexDifference(int min_diff, int max_diff, int start_idx, int end_idx)
{
    vector<int> invalid_ids;
    vector<MatchPair> good_pairs;
    for(const MatchPair& pair : image_pairs)
    {
        int diff = pair.image_pair.second - pair.image_pair.first;
        // 不在过滤范围内的图像对，直接保留
        if(!IsInside(int(pair.image_pair.first), start_idx, end_idx) || !IsInside(int(pair.image_pair.second), start_idx, end_idx))
            good_pairs.push_back(pair);
        else if(!IsInside(diff, min_diff, max_diff))
            good_pairs.push_back(pair); 
    }
    LOG(INFO) << "filter " << image_pairs.size() - good_pairs.size() << " pairs by index difference";
    image_pairs = good_pairs;
    return true;
}

bool SfM::SetToStraightMotion(size_t start_idx, size_t end_idx, size_t length)
{
    end_idx = min(end_idx, frames.size() - 1);
    assert(start_idx + length <= end_idx);
    vector<MatchPair> pairs_preserve;
    for(const MatchPair& p : image_pairs)
    {
        if(IsInside(p.image_pair.first, start_idx, end_idx) && IsInside(p.image_pair.second, start_idx, end_idx))
            continue;
        pairs_preserve.push_back(p);
    }

    image_pairs.clear();

    LoadGPS(config.gps_path);

    for(size_t i = start_idx; i <= end_idx; i++)
    {
        frames[i].LoadImageGray(frames[i].name);
        frames[i].ComputeDescriptor(config.root_sift);
    }

    for(size_t idx1 = start_idx; idx1 < end_idx; idx1++)
        for(size_t idx2 = idx1 + 1; idx2 <= idx1 + length && idx2 <= end_idx; idx2++)
        {
            const Frame& frame1 = frames[idx1];
            const Frame& frame2 = frames[idx2];
            double scale_gps = (frame1.GetGPS() - frame2.GetGPS()).norm();
            Eigen::Vector3d t_12(0,0,scale_gps);
            Eigen::Matrix3d R_21 = Eigen::Matrix3d::Identity();
            Eigen::Vector3d t_21 = - R_21 * t_12;
            vector<cv::DMatch> matches = MatchSIFT(frame1.GetDescriptor(), frame2.GetDescriptor(), config.sift_match_dist_threshold);
            MatchPair pair(idx1, idx2, matches);
            pair.R_21 = R_21;
            pair.t_21 = t_21;
            pair.upper_scale = pair.lower_scale = 0;
            // 三维向量到反对称矩阵
            Eigen::Matrix3d t_21_hat = Eigen::Matrix3d::Zero();
            t_21_hat <<    0,          -t_21.z(),      t_21.y(),
                        t_21.z(),           0,        -t_21.x(),
                        -t_21.y(),      t_21.x(),           0;
            pair.E_21 = t_21_hat * R_21;

            Equirectangular eq(frame1.GetImageRows(), frame1.GetImageCols());
            vector<cv::Point3f> keypoints_sphere1, keypoints_sphere2;
            for(auto& kp : frame1.GetKeyPoints())
                keypoints_sphere1.push_back(eq.ImageToCam(kp.pt));
            for(auto& kp : frame2.GetKeyPoints())
                keypoints_sphere2.push_back(eq.ImageToCam(kp.pt));

            
            vector<bool> is_inlier(matches.size(), true);
            CheckRT(R_21, t_21, is_inlier, matches, keypoints_sphere1, keypoints_sphere2, *(new double), pair.triangulated, pair.inlier_idx);
            // 画出匹配关系
            // vector<cv::DMatch> good_matches;
            // for(const size_t& idx : pair.inlier_idx)
            //     good_matches.push_back(pair.matches[idx]);
            // cv::Mat matches_vertical = DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(),
            //                         frame2.GetImageColor(), frame2.GetKeyPoints(), good_matches);
            // cv::imwrite(config.sfm_result_path + "/straight_" + num2str(idx1) + "_" + num2str(idx2) + ".jpg", matches_vertical);
            image_pairs.push_back(pair);

        }
    image_pairs.insert(image_pairs.end(), pairs_preserve.begin(), pairs_preserve.end());
    for(size_t i = start_idx; i <= end_idx; i++)
    {
        frames[i].ReleaseDescriptor();
        frames[i].ReleaseImageGray();
    }
    return true;
}

bool SfM::AddPair(size_t idx1, size_t idx2, bool straight_motion)
{
    map<pair<size_t, size_t>, size_t> pair_to_idx;
        for(size_t i = 0; i < image_pairs.size(); i++)
            pair_to_idx[image_pairs[i].image_pair] = i;
    Frame& frame1 = frames[idx1];
    Frame& frame2 = frames[idx2];
    frame1.LoadImageGray(frame1.name);
    frame2.LoadImageGray(frame2.name);
    frame1.ComputeDescriptor(config.root_sift);
    frame2.ComputeDescriptor(config.root_sift);
    vector<cv::DMatch> matches = MatchSIFT(frame1.GetDescriptor(), frame2.GetDescriptor(), config.sift_match_dist_threshold);
    MatchPair pair(idx1, idx2, matches);
    Eigen::Matrix3d R_21 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_21(0,0,0);
    if(straight_motion)
    {
        Eigen::Vector3d t_12(0,0,1);
        t_21 = - R_21 * t_12;
    }
    else
    {
        int neighbor_pair_idx1 = -1, neighbor_pair_idx2 = -1, distance1 = -1, distance2 = -1;
        for(int i = 0; i < 5; i++)
        {
            if(neighbor_pair_idx1 < 0 && pair_to_idx.count(make_pair(idx1 + i, idx2 + i)))
            {
                neighbor_pair_idx1 = pair_to_idx[make_pair(idx1 + i, idx2 + i)];
                distance1 = i;
            }
            if(neighbor_pair_idx2 < 0 && pair_to_idx.count(make_pair(idx1 - i, idx2 - i)))
            {
                neighbor_pair_idx2 = pair_to_idx[make_pair(idx1 - i, idx2 - i)];
                distance2 = i;
            }
        }
        
        
        
    }
    pair.R_21 = R_21;
    pair.t_21 = t_21;
    pair.upper_scale = pair.lower_scale = 0;
    // 三维向量到反对称矩阵
    Eigen::Matrix3d t_21_hat = Eigen::Matrix3d::Zero();
    t_21_hat <<    0,          -t_21.z(),      t_21.y(),
                t_21.z(),           0,        -t_21.x(),
                -t_21.y(),      t_21.x(),           0;
    pair.E_21 = t_21_hat * R_21;

    Equirectangular eq(frame1.GetImageRows(), frame1.GetImageCols());
    vector<cv::Point3f> keypoints_sphere1, keypoints_sphere2;
    for(auto& kp : frame1.GetKeyPoints())
        keypoints_sphere1.push_back(eq.ImageToCam(kp.pt));
    for(auto& kp : frame2.GetKeyPoints())
        keypoints_sphere2.push_back(eq.ImageToCam(kp.pt));

    vector<bool> is_inlier(matches.size(), true);
    CheckRT(R_21, t_21, is_inlier, matches, keypoints_sphere1, keypoints_sphere2, *(new double), pair.triangulated, pair.inlier_idx);
    // 画出匹配关系
    // vector<cv::DMatch> good_matches;
    // for(const size_t& idx : pair.inlier_idx)
    //     good_matches.push_back(pair.matches[idx]);
    // cv::Mat matches_vertical = DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(),
    //                         frame2.GetImageColor(), frame2.GetKeyPoints(), good_matches);
    // cv::imwrite(config.sfm_result_path + "/straight_" + num2str(idx1) + "_" + num2str(idx2) + ".jpg", matches_vertical);
    if(LoadGPS(config.gps_path))
    {
        double scale_gps = (frame1.GetGPS() - frame2.GetGPS()).norm();
        pair.upper_scale = scale_gps * 1.5;
        pair.lower_scale = scale_gps * 0.5;
        pair.t_21 *= scale_gps;
        for(Eigen::Vector3d& p : pair.triangulated)
            p *= scale_gps;
    }
    else 
    {
        vector<string> depth_image_names = IterateFiles(config.depth_path, ".bin");
        if(frame1.depth_map.empty())
            ReadOpenCVMat(depth_image_names[idx1], frame1.depth_map);
        if(frame2.depth_map.empty())
            ReadOpenCVMat(depth_image_names[idx2], frame2.depth_map);
        SetTranslationScaleDepthMap(eq, pair);
        frame1.depth_map.release();
        frame2.depth_map.release();
    }
    image_pairs.push_back(pair);

    frame1.ReleaseDescriptor();
    frame2.ReleaseDescriptor();
    frame1.ReleaseImageGray();
    frame2.ReleaseImageGray();
    image_pairs.push_back(pair);
    return true;
}

void SfM::PrintRelativePose(const string& file_path)
{
    ofstream f(file_path);
    if(!f.is_open())
    {
        LOG(ERROR) << "fail to open file " << file_path;
        return;
    }
    for(const MatchPair& pair : image_pairs)
    {
        bool scale_is_good = (pair.upper_scale > 0 && pair.lower_scale > 0);
        Eigen::Vector3d t_12 = -pair.R_21.transpose() * pair.t_21;
        f << "pair : " << pair.image_pair.first << " " << pair.image_pair.second << (scale_is_good ? " good" : " bad") << endl;
        f << t_12.x() << " " << t_12.y() << " " << t_12.z() << endl;
        Eigen::AngleAxisd angleAxis(pair.R_21.transpose());
        f << angleAxis.axis().x() << " " << angleAxis.axis().y() << " " << angleAxis.axis().z() << " " << angleAxis.angle() * 180.0 / M_PI << endl;
    }
    f.close();
}

void SfM::PrintGlobalPose(const string& file_path)
{
    ofstream f(file_path);
    if(!f.is_open())
    {
        LOG(ERROR) << "fail to open file " << file_path;
        return;
    }
    // 这里的下标c 代表center，是指的新的世界坐标系
    const Eigen::Matrix3d R_wc = frames[0].GetPose().block<3,3>(0,0);
    for(size_t i = 0; i < frames.size(); i++)
    {
        Eigen::Matrix3d R_iw = frames[i].GetPose().block<3,3>(0,0).transpose();
        Eigen::Matrix3d R_ic = R_iw * R_wc;     // 新的世界坐标系到相机坐标系的变换
        Eigen::Matrix3d R_ci = R_ic.transpose();
        Eigen::AngleAxisd angleAxis(R_ci);
        f << "frame : " << i << ", rotation : " 
            << angleAxis.axis().x() << " " << angleAxis.axis().y() << " " << angleAxis.axis().z() << " " << angleAxis.angle() * 180.0 / M_PI << endl;

    }
    f.close();
}

bool SfM::test()
{
    LoadFrameBinary(config.image_path, config.sfm_result_path + "frames_after_RA", true);
    ReadPointTracks(config.sfm_result_path + "points.bin", structure);
    ColorizeStructure();
    ExportStructurePCD(config.sfm_result_path + "structured_colored.pcd");
    return true;
}

bool SfM::test2()
{
    size_t id = 2104;
    frames[id].LoadImageColor(frames[id].name);
    for(const MatchPair& p : image_pairs)
    {
        if(p.image_pair.first == id)
        {
            cv::Mat matches_vertical = DrawMatchesVertical(frames[id].GetImageColorRef(), frames[id].GetKeyPoints(),
                                    frames[p.image_pair.second].GetImageColor(), frames[p.image_pair.second].GetKeyPoints(), p.matches, p.inlier_idx);
            cv::imwrite(config.sfm_result_path + "/relative_pose_inlier_" + num2str(id) + "_" + num2str(p.image_pair.second) + ".jpg", matches_vertical);
            matches_vertical = DrawMatchesVertical(frames[id].GetImageColorRef(), frames[id].GetKeyPoints(),
                                    frames[p.image_pair.second].GetImageColor(), frames[p.image_pair.second].GetKeyPoints(), p.matches);
            cv::imwrite(config.sfm_result_path + "/sift_" + num2str(id) + "_" + num2str(p.image_pair.second) + ".jpg", matches_vertical);
            LOG(INFO) << "pair : " << p.image_pair.first << " " << p.image_pair.second << 
                    ", sift matches : " << p.matches.size() << ", inlier : " << p.inlier_idx.size();
        }
    }
    return true;
}

bool SfM::test_sift()
{
    vector<string> image_names = IterateFiles(config.image_path, ".jpg");
    cv::Mat mask = cv::imread(config.mask_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::threshold(mask, mask, 0.1, 1, CV_THRESH_BINARY);
    set<int> ids = {3206, 3207, 3208, 3209, 3210, 3211, 3212};
    // 目前假设所有图像的分辨率都是相同的
    cv::Mat img = cv::imread(image_names[0]);
    
    for(int id = 0; id < image_names.size(); id++)
    {
        Frame frame(img.rows, img.cols, id, image_names[id]);
        if(ids.count(id) > 0)
        {
            frame.LoadImageGray(frame.name);
            frame.ExtractKeyPoints(config.num_sift, mask);
            frame.ComputeDescriptor(config.root_sift);
            LOG(INFO) << "Extract " << frame.GetKeyPoints().size() << " key points";
            cv::imwrite(config.sfm_result_path + "keypoints_" + num2str(id) + ".jpg", DrawKeyPoints(frame.GetImageColor(), frame.GetKeyPoints()));
            frame.ReleaseImageGray();
        }
        frames.push_back(frame);
    }
    for(const int& id1 : vector<int>(ids.begin(), ids.end()))
    {
        for(const int& id2: vector<int>(ids.begin(), ids.end()))
        {
            if(id1 >= id2)
                continue;
            image_pairs.push_back(MatchPair(id1, id2));
        }
    }
    MatchImagePairs(config.sift_match_num_threshold);
    for(const MatchPair& p : image_pairs)
    {
        const size_t& idx1 = p.image_pair.first;
        const size_t& idx2 = p.image_pair.second;
        const Frame& frame1 = frames[idx1];
        const Frame& frame2 = frames[idx2];
        cv::imwrite(config.sfm_result_path + "sift_" + num2str(p.image_pair.first) + "_" + num2str(p.image_pair.second) + ".jpg", 
            DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(), frame2.GetImageColor(), frame2.GetKeyPoints(), p.matches));
        LOG(INFO) << "sift matches for image pair " << idx1 << " - " << idx2 << " : " << p.matches.size();
    }
    FilterImagePairs();
    for(const MatchPair& p : image_pairs)
    {
        const size_t& idx1 = p.image_pair.first;
        const size_t& idx2 = p.image_pair.second;
        const Frame& frame1 = frames[idx1];
        const Frame& frame2 = frames[idx2];
        // 画出匹配关系
        cv::Mat matches_vertical = DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(),
                                frame2.GetImageColor(), frame2.GetKeyPoints(), p.matches, p.inlier_idx);
        cv::imwrite(config.sfm_result_path + "/pose_" + num2str(idx1) + "_" + num2str(idx2) + ".jpg", matches_vertical);
        LOG(INFO) << "rel pose matches for image pair " << idx1 << " - " << idx2 << " : " << p.inlier_idx.size();
    }
    
    return true;
}

// 1.52004 degree

bool SfM::CASIA()
{
    #if 1
    LoadFrameBinary(config.image_path, config.sfm_result_path + "frames_after_RA");
    LoadMatchPairBinary(config.sfm_result_path + "pairs_after_relative_pose");
    PrintGlobalPose(config.sfm_result_path + "frame_pose-after-RA.txt");
    // return false;
    // LoadMatchPairBinary(config.sfm_result_path + "pairs_after_RA");
    
    
    set<int> ids = {1503, 1504, 1505, 1546, 1547};
    #pragma omp parallel for
    for(const MatchPair& pair : image_pairs)
    {
        if(ids.count(pair.image_pair.first) <= 0 && ids.count(pair.image_pair.second) <= 0)
            continue;
        const Frame& frame1 = frames[pair.image_pair.first];
        const Frame& frame2 = frames[pair.image_pair.second];
        vector<cv::DMatch> good_matches;
        for(auto& id : pair.inlier_idx)
            good_matches.push_back(pair.matches[id]);
        cv::imwrite(config.sfm_result_path + "inlier_" + num2str(pair.image_pair.first) + "_" + num2str(pair.image_pair.second) + ".jpg", 
                    DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(), frame2.GetImageColor(), frame2.GetKeyPoints(), good_matches));
        cv::imwrite(config.sfm_result_path + "sift_" + num2str(pair.image_pair.first) + "_" + num2str(pair.image_pair.second) + ".jpg", 
                    DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(), frame2.GetImageColor(), frame2.GetKeyPoints(), pair.matches));

    }
   
    set<size_t> covered_frames;
    vector<size_t> not_covered_frames;
    for(const MatchPair& pair : image_pairs)
    {
        covered_frames.insert(pair.image_pair.first);
        covered_frames.insert(pair.image_pair.second);
    }
    for(size_t i = 0; i < frames.size(); i++)
    {
        if(covered_frames.count(i) <= 0)
            not_covered_frames.push_back(i);
    }
    LOG(INFO) << "covered frames : " << covered_frames.size() << "\n \t not covered frames : " << Join(not_covered_frames);
    PrintRelativePose(config.sfm_result_path + "relative_pose-raw.txt");
    exit(0);
    return false;
    
    LoadFrameBinary(config.image_path, config.sfm_result_path + "/frames_after_RA", true);  
    LoadMatchPairBinary(config.sfm_result_path + "/pairs_after_RA");
    set<uint32_t> frame_with_rotation;
    vector<uint32_t> frame_without_rotation;
    for(const Frame& f : frames)
    {
        if(f.GetPose().block<3,3>(0,0).isZero())
            frame_without_rotation.push_back(f.id);
        else 
            frame_with_rotation.insert(f.id);
    }
    
    LOG(INFO) << "frame with roation : " << frame_with_rotation.size() << " , frame without ration : " << Join(frame_without_rotation);
    #endif
    LoadFrameBinary(config.image_path, config.frame_path);
    LoadMatchPairBinary(config.sfm_result_path + "pairs_after_relative_pose");
    // LoadGPS(config.gps_path);
    // SetToStraightMotion(3323 - 10, 3323 + 10, 1);

    // 重新计算某些图像对
    ReComputePairs(1500, 1507);
    ReComputePairs(1540, 1550);


    // vector<MatchPair> good_pairs;
    // for(const MatchPair& p : image_pairs)
    // {
    //     const size_t& idx1 = p.image_pair.first;
    //     const size_t& idx2 = p.image_pair.second;
    //     if(frame_with_rotation.count(idx1) > 0 && frame_with_rotation.count(idx2) > 0)
    //     {
    //         const Eigen::Matrix3d& R_21 = p.R_21;
    //         Eigen::Matrix3d R_w2 = frames[idx2].GetPose().block<3,3>(0,0);
    //         Eigen::Matrix3d R_w1 = frames[idx1].GetPose().block<3,3>(0,0);
    //         const Eigen::Matrix3d error_mat = R_w2 * R_21 * R_w1.transpose();
    //         Eigen::Vector3d error_angleaxis;
    //         ceres::RotationMatrixToAngleAxis(error_mat.data(), error_angleaxis.data());
    //         double error = error_angleaxis.norm();
    //         if(error * 180.0 / M_PI < 1.6)
    //             good_pairs.push_back(p);
    //     }
    // }
    // image_pairs = good_pairs;
    
  
    sort(image_pairs.begin(), image_pairs.end(), 
        [this](const MatchPair& mp1,const MatchPair& mp2)
        {
            if(mp1.image_pair.first < mp2.image_pair.first)
                return true;
            else 
                return mp1.image_pair.second < mp2.image_pair.second;
        }
        );
    EstimateGlobalTranslation(config.translation_averaging_method);
    return true;
}

bool SfM::test_pipeline()
{
    LoadFrameBinary(config.image_path, config.frame_path);
    // set<int> ids = {1224, 1225, 1226, 1227, 1228};
    // set<int> ids = {124, 125, 126, 127, 128};
    // set<int> ids = {1324, 1325, 1326, 1327, 1328};
    set<int> ids = {2455, 2458};
    
    for(const int& id1 : vector<int>(ids.begin(), ids.end()))
    {
        for(const int& id2: vector<int>(ids.begin(), ids.end()))
        {
            if(id1 >= id2)
                continue;
            image_pairs.push_back(MatchPair(id1, id2));
        }
    }
    MatchImagePairs(config.sift_match_num_threshold);
    auto t1 = chrono::high_resolution_clock::now();
    FilterImagePairs();
    auto t2 = chrono::high_resolution_clock::now();
    LOG(INFO) << "time spend " << chrono::duration_cast<chrono::duration<double> >(t2 - t1).count() * 1000.0 << " ms";
    for(const MatchPair& p : image_pairs)
    {
        const size_t& idx1 = p.image_pair.first;
        const size_t& idx2 = p.image_pair.second;
        const Frame& frame1 = frames[idx1];
        const Frame& frame2 = frames[idx2];
        // 画出匹配关系
        cv::Mat matches_vertical = DrawMatchesVertical(frame1.GetImageColor(), frame1.GetKeyPoints(),
                                frame2.GetImageColor(), frame2.GetKeyPoints(), p.matches, p.inlier_idx);
        cv::imwrite(config.sfm_result_path + "/pose_" + num2str(idx1) + "_" + num2str(idx2) + ".jpg", matches_vertical);
        LOG(INFO) << "rel pose matches for image pair " << idx1 << " - " << idx2 << " : " << p.inlier_idx.size();
    }
    PrintRelativePose(config.sfm_result_path + "relative_pose-raw.txt");
    return true;
}

#include "../util/Util.h"
bool SfM::test_GPS_sync()
{
    
    if(!config.gps_path.empty())
        LoadGPS(config.gps_path);
    // ReadPoseT()
    LoadFramePose(frames, config.sfm_result_path + "camera_pose_final.txt");
    
    vector<int> start_ids = {502, 5957};
    vector<int> end_ids = {723, 6000};
    int length = 0;
    for(size_t i = 0; i < start_ids.size(); i++)
        length += end_ids[i] - start_ids[i] + 1;
    Eigen::Matrix<double, 3, Eigen::Dynamic> source_pose, target_pose;
    source_pose.resize(3, length);
    target_pose.resize(3, length);
    for(size_t i = 0, col = 0; i < start_ids.size(); i++)
    {
        
        for(int j = start_ids[i]; j <= end_ids[i]; j++, col++)
        {
            source_pose.col(col) = frames[j].GetGPS();
            target_pose.col(col) = frames[j].GetPose().block<3,1>(0,3);
        }
    }
    Eigen::Matrix4d T_w1_w2 = Eigen::umeyama(source_pose, target_pose, false);
    Eigen::Matrix3d R_w1_w2 = T_w1_w2.block<3,3>(0,0);
    eigen_vector<Eigen::Vector3d> init_gps_pose(frames.size()), gps_with_rot(frames.size()), gps_with_T(frames.size());

    for(size_t i = 0; i < frames.size(); i++)
    {
        init_gps_pose[i] = frames[i].GetGPS();
        gps_with_rot[i] = R_w1_w2 * frames[i].GetGPS();
        gps_with_T[i] =( T_w1_w2 * frames[i].GetGPS().homogeneous()).hnormalized();
    }
    CameraCenterPCD(config.sfm_result_path + "GPS_before_sync.pcd", init_gps_pose);
    CameraCenterPCD(config.sfm_result_path + "GPS_with_rotation.pcd", gps_with_rot);
    CameraCenterPCD(config.sfm_result_path + "GPS_with_T.pcd", gps_with_T);
    LOG(INFO) << "Transformation is : \n" << T_w1_w2;
    return true;
}

void SfM::AddScaleToPose(double scale)
{
    for(PointTrack& track : structure)
    {
        track.point_3d *= scale;
    }
    for(Frame& frame : frames)
    {
        Eigen::Vector3d t_wc = frame.GetPose().block<3,1>(0,3);
        t_wc *= scale;
        frame.SetTranslation(t_wc);
    }
}

void SfM::ClearPairScale()
{
    LOG(INFO) << "clear all pair scale";
    for(MatchPair& pair : image_pairs)
    {
        double scale = pair.t_21.norm();
        pair.t_21 /= scale;
        for(Eigen::Vector3d& point : pair.triangulated)
            point /= scale;
        pair.upper_scale = -1;
        pair.lower_scale = -1;
    }
}
