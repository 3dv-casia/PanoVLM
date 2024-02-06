/*
 * @Author: Diantao Tu
 * @Date: 2022-04-20 11:56:37
 */
#include "PanoramaLineMatch.h"

bool PanoramaLineMatcher::VisualizeTrack(const pair<uint32_t, set<pair<uint32_t, uint32_t>>>& track, const vector<PanoramaLine>& image_lines_all, const string path)
{
    uint32_t track_id = track.first;
    // 对于直线形成的track而言，可能并不是一对一的关系，也就是说有可能存在一张图像上的两条直线都属于同一条track，所以需要
    // 进行一下统计，找到当前track里每张图像上的直线id，画直线的时候要画在一起
    // key = image id   value = {line id，line id， line id}
    map<uint32_t, set<uint32_t>> image_lines;
    for(const auto& pair : track.second)
        image_lines[pair.first].insert(pair.second);
    for(map<uint32_t, set<uint32_t>>::const_iterator it = image_lines.begin(); it != image_lines.end(); it++)
    {
        const uint32_t& image_id = it->first;
        cv::Mat img_gray = image_lines_all[image_id].GetImageGray();
        cv::cvtColor(img_gray, img_gray, cv::COLOR_GRAY2BGR);
        for(const uint32_t& line_id : it->second)
            DrawLine(img_gray, image_lines_all[image_id].GetLines()[line_id],  cv::Scalar(0,0,255), 6, true);
        cv::imwrite(path + "/track" + num2str(track_id) + "_" + num2str(image_id) + ".jpg", img_gray);
    }
    return true;
}

PanoramaLineMatcher::PanoramaLineMatcher(const std::vector<PanoramaLine>& _image_lines_all, const eigen_vector<Eigen::Matrix3d>& _R_wc_list,
                        const eigen_vector<Eigen::Vector3d>& _t_wc_list):
                        image_lines_all(_image_lines_all), R_wc_list(_R_wc_list), t_wc_list(_t_wc_list), neighbor_size(2),min_track_length(3)
{}

std::vector<cv::DMatch> PanoramaLineMatcher::MatchInitLine(const PanoramaLine& lines1, const PanoramaLine& lines2, const int method)
{
    vector<cv::DMatch> matches;
    cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> lbd = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    if(method == KNN)
    {
        vector<vector<cv::DMatch>> matches_all;
        lbd->knnMatch(lines1.GetLineDescriptor(), lines2.GetLineDescriptor(), matches_all, 2);
        for(const vector<cv::DMatch>& m : matches_all)
        {
            if(m[0].distance < 0.9 * m[1].distance)
                matches.push_back(m[0]);
        }
    }
    else if(method == BASIC)
    {
        lbd->match(lines1.GetLineDescriptor(), lines2.GetLineDescriptor(), matches);
    }
    else 
    {
        LOG(ERROR) << "panorama line match method not support";
        return vector<cv::DMatch>();
    }
    vector<cv::DMatch> matches_after_filter = FilterLineMatchOpticalFlow(lines1.GetInitLines(), lines2.GetInitLines(), lines1.GetImageGray(), lines2.GetImageGray(), matches);
    return matches_after_filter;
}

std::vector<cv::DMatch> PanoramaLineMatcher::MatchPanoramaLine(const PanoramaLine& lines1, const PanoramaLine& lines2, 
                                                                const int method, bool cross_check)
{
    vector<cv::DMatch> matches = MatchInitLine(lines1, lines2, method);
    if(cross_check)
    {
        vector<cv::DMatch> matches2 = MatchInitLine(lines2, lines1, method);
        for(cv::DMatch& m : matches2)
        {
            int tmp = m.trainIdx;
            m.trainIdx =  m.queryIdx;
            m.queryIdx = tmp;
        }
        matches = MatchesIntersection(matches, matches2);
    }
    // cv::imwrite("init_line_match" + num2str(lines1.id) + "-" + num2str(lines2.id) + ".jpg", 
    //             DrawMatchesVertical(lines1.GetImageGray(), lines1.GetInitLines(), lines2.GetImageGray(), lines2.GetInitLines(), matches));
   
    const vector<vector<size_t>>& init_to_final1 = lines1.GetInitToFinal();
    const vector<vector<size_t>>& init_to_final2 = lines2.GetInitToFinal();
    const vector<vector<size_t>>& final_to_init1 = lines1.GetFinalToInit();
    const vector<vector<size_t>>& final_to_init2 = lines2.GetFinalToInit();
    // 通过初始直线之间的匹配确定最终直线的匹配关系，主要方法就是计算一个匹配矩阵，这个矩阵共m行n列，
    // m=图像1的直线数，n=图像2的直线数，矩阵的每一个元素代表着当前的两条直线有多少个初始直线是相互匹配的
    cv::Mat match_matrix = cv::Mat::zeros(lines1.GetLines().size(), lines2.GetLines().size(), CV_16U);
    for(const cv::DMatch& m : matches)
    {
        int init_line_id1 = m.queryIdx;
        int init_line_id2 = m.trainIdx;
        for(const size_t& final_line_id1 : init_to_final1[init_line_id1])
            for(const size_t& final_line_id2 : init_to_final2[init_line_id2])
                match_matrix.at<uint16_t>(final_line_id1, final_line_id2) += 1;
    }
    matches.clear();
    for(size_t final_line_id1 = 0; final_line_id1 < match_matrix.rows; final_line_id1++)
    {
        for(size_t final_line_id2 = 0; final_line_id2 < match_matrix.cols; final_line_id2++)
        {
            uint16_t match_size = match_matrix.at<uint16_t>(final_line_id1, final_line_id2);
            // 判断两条直线是否是匹配的，有三个条件，满足任意一个就行
            if( match_size >= 2 ||
                match_size >= 0.5* final_to_init1[final_line_id1].size() || 
                match_size >= 0.5* final_to_init2[final_line_id2].size() )
            {
                cv::DMatch match;
                match.queryIdx = final_line_id1;
                match.trainIdx = final_line_id2;
                matches.push_back(match);
            }
        }
    }
    return matches;
}

std::vector<cv::DMatch> PanoramaLineMatcher::FilterLineMatchOpticalFlow(const std::vector<cv::line_descriptor::KeyLine>& lines1, 
                                                        const std::vector<cv::line_descriptor::KeyLine>& lines2,
                                                        const cv::Mat& img_gray1, const cv::Mat& img_gray2,
                                                        const std::vector<cv::DMatch>& matches)
{
    int num_keypoints = 10;
    vector<cv::Point2f> points1, points2;
    // 在每一条直线上均匀的采样n个，由于这些直线都是用LSD检测出来的，所以肯定是直的，不需要用全景直线的近似方法
    for(const cv::line_descriptor::KeyLine& l : lines1)
    {
        float delta_x = l.endPointX - l.startPointX;
        float delta_y = l.endPointY - l.startPointY;
        for(int i = 0; i < num_keypoints; i++)
            points1.push_back(cv::Point2f(delta_x * i / num_keypoints + l.startPointX, delta_y * i / num_keypoints + l.startPointY));
    }
    // 对每条直线上的点计算光流
    vector<uchar> statue;
    vector<float> errors;
    cv::calcOpticalFlowPyrLK(img_gray1, img_gray2, points1, points2, statue, errors);
    vector<cv::DMatch> good_matches;
    // 得到光流后，计算每个匹配对之间对应点的误差
    for(const cv::DMatch& m : matches)
    {
        vector<cv::Point2f> points_avaliable;
        for(size_t i = m.queryIdx * num_keypoints; i < (m.queryIdx + 1) * num_keypoints; i++)
        {
            if(statue[i])
                points_avaliable.push_back(points2[i]);
        }
        // 至少要有80%的点能在另一张图像上通过光流找到
        if(points_avaliable.size() < 0.8 * num_keypoints)
            continue;
        // 计算所有光流跟踪的点到初始的匹配直线的距离，距离越小代表当前点越接近初始的匹配直线，那就说明初始的匹配是正确的
        // 但是经过实验，发现一个现象，当前直线有10个点，可能其中的8-9个点对应的点到直线距离都很小，但是剩下的几个点的距离就很大
        // 但是实际上这对匹配关系是正确的，光流却出现了计算错误，因此这里使用了所有距离的中值
        // 如果中值也很大，那么大概率这就是个错误匹配点
        cv::Vec4f line(lines2[m.trainIdx].startPointX, lines2[m.trainIdx].startPointY, lines2[m.trainIdx].endPointX, lines2[m.trainIdx].endPointY);
        vector<float> sq_dist = PointToLineDistance2DSquare(line, points_avaliable);
        nth_element(sq_dist.begin(), sq_dist.size()/2 + sq_dist.begin(), sq_dist.end());
        if(sq_dist[sq_dist.size()/2]  > 60 * 60)
            continue;
        good_matches.push_back(m);
    }
    return good_matches;
}

void PanoramaLineMatcher::CompareMatchMethod(const PanoramaLine& lines1, const PanoramaLine& lines2)
{
    vector<cv::DMatch> matches_knn = MatchPanoramaLine(lines1, lines2, KNN);
    cv::Mat img_line_match = DrawMatchesVertical(lines1.GetImageGray(), lines1.GetLines(), 
                lines2.GetImageGray(), lines2.GetLines(), matches_knn, true);
    cv::imwrite("final_line_match_knn" + num2str(lines1.id) + "-" + num2str(lines2.id) + ".jpg", img_line_match);

    vector<cv::DMatch> matches_basic = MatchPanoramaLine(lines1, lines2, BASIC);
    img_line_match = DrawMatchesVertical(lines1.GetImageGray(), lines1.GetLines(), 
                lines2.GetImageGray(), lines2.GetLines(), matches_basic, true);
    cv::imwrite("final_line_match_basic" + num2str(lines1.id) + "-" + num2str(lines2.id) + ".jpg", img_line_match);

    set<pair<int,int>> match_pair_knn;
    for(auto& m : matches_knn)
        match_pair_knn.insert({m.queryIdx, m.trainIdx});
    vector<cv::DMatch> matches_diff;
    for(auto& m : matches_basic)
    {
        if(match_pair_knn.count({m.queryIdx, m.trainIdx}) == 0)
            matches_diff.push_back(m);
    }
    img_line_match = DrawMatchesVertical(lines1.GetImageGray(), lines1.GetLines(), 
                lines2.GetImageGray(), lines2.GetLines(), matches_diff, true);
    cv::imwrite("final_line_match_diff" + num2str(lines1.id) + "-" + num2str(lines2.id) + ".jpg", img_line_match);

}

bool PanoramaLineMatcher::GenerateTracks(const int method)
{
    const float angle_threshold = 4.0;
    vector<pair<size_t, size_t>> pairs;
    vector<vector<cv::DMatch>> pair_matches;
    // 首先对所有图像对之间进行直线匹配，得到初始的直线匹配关系
    for(int i = 0; i < int(image_lines_all.size() - neighbor_size); i++)
    {
        for(int j = i + 1; j < image_lines_all.size() && j <= i + neighbor_size; j++)
        {
            vector<cv::DMatch> matches = MatchPanoramaLine(image_lines_all[i], image_lines_all[j], method);
            cv::Mat img_line_match = DrawMatchesVertical(image_lines_all[i].GetImageGray(), image_lines_all[i].GetLines(), 
                        image_lines_all[j].GetImageGray(), image_lines_all[j].GetLines(), matches, true);
            cv::imwrite("final_line_match" + num2str(image_lines_all[i].id) + "-" + num2str(image_lines_all[j].id) + ".jpg", img_line_match);

            pairs.push_back({i,j});
            pair_matches.push_back(matches);
        }
    }
    // 根据直线匹配关系建立track
    TrackBuilder tracks_builder(true);
    tracks_builder.Build(pairs, pair_matches);
    tracks_builder.Filter(min_track_length);
    tracks_builder.ExportTracks(tracks);
    LOG(INFO) << "generate " << tracks.size() << " image line tracks";
    if(tracks.empty())
        return false;
    // for(const LineTrack& track : tracks)
        // VisualizeTrack({track.id, track.feature_pairs}, image_lines_all, "./");

    // 下面这部分是生成一些要用的信息，包括特征对{image id，line id}到id的映射，以及特征对之间的近邻关系
    // 这些信息都是用于之后的过滤的
    uint32_t idx = 0;
    for(size_t i = 0; i < pairs.size(); i++)
    {
        for(const cv::DMatch& curr_match : pair_matches[i])
        {
            uint32_t idx1, idx2;
            pair<uint32_t, uint32_t> feature(pairs[i].first, curr_match.queryIdx);
            if(feature_to_index.find(feature) == feature_to_index.end())
            {
                idx1 = idx;
                feature_to_index[feature] = idx;
                index_to_feature[idx] = feature;
                idx++;
            }
            else 
                idx1 = feature_to_index.find(feature)->second;
            feature.first = pairs[i].second;
            feature.second = curr_match.trainIdx;
            if(feature_to_index.find(feature) == feature_to_index.end())
            {
                idx2 = idx;
                feature_to_index[feature] = idx;
                index_to_feature[idx] = feature;
                idx++;
            }
            else 
                idx2 = feature_to_index.find(feature)->second;
            neighbors_all[idx1].push_back(idx2);
            neighbors_all[idx2].push_back(idx1);
        }
    }
    
    // 过滤
    FilterTracks(angle_threshold);
    LOG(INFO) << "after filter with angle, " << tracks.size() << " image line tracks";
    // for(const LineTrack& track : tracks)
    //     VisualizeTrack({track.id, track.feature_pairs}, image_lines_all, "./");
    return true;
}

bool PanoramaLineMatcher::FilterTracks(const float angle_threshold)                                          
{
    eigen_vector<Eigen::Matrix4f> T_wc_list;
    for(size_t i = 0; i < R_wc_list.size(); i++)
    {
        Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
        T_wc.block<3,3>(0,0) = R_wc_list[i];
        T_wc.block<3,1>(0,3) = t_wc_list[i];
        T_wc_list.push_back(T_wc.cast<float>());
    }
    Equirectangular eq(image_lines_all[0].GetImageGray().rows, image_lines_all[0].GetImageGray().cols);
    // 计算每条直线和当前相机的光心形成的平面在世界坐标系下的参数
    vector<vector<cv::Vec4f>> planes_world(image_lines_all.size());
    vector<vector<cv::Point3f>> end_points_world(image_lines_all.size());
    for(size_t i = 0; i < image_lines_all.size(); i++)
    {
        const vector<cv::Vec4f>& final_lines = image_lines_all[i].GetLines();
        for(const cv::Vec4f& l : final_lines)
        {
            // 把起始点和终止点都变换成球上的XYZ坐标,并且变换到世界坐标系下
            cv::Point3f p1 = TranslatePoint(eq.ImageToCam(cv::Point2f(l[0], l[1])), T_wc_list[i]);
            cv::Point3f p2 = TranslatePoint(eq.ImageToCam(cv::Point2f(l[2], l[3])), T_wc_list[i]);
            cv::Point3f p3(t_wc_list[i].x(), t_wc_list[i].y(), t_wc_list[i].z());
            // 计算三个点在世界坐标系下形成的平面
            cv::Vec4f plane = FormPlane(p1, p2, p3) ;
            planes_world[i].push_back(plane); 
            end_points_world[i].push_back(p1);
            end_points_world[i].push_back(p2);
        }
    }
    std::set<std::pair<uint32_t, uint32_t>> valid_feature_index_pair;
    for(const LineTrack& track : tracks)
    {
        // 对每条track里的特征进行过滤，不同track之间的特征是没有近邻匹配关系的，因此可以同时的并行过滤
        std::set<std::pair<uint32_t, uint32_t>> valid_pairs = FilterPairsInTrack(track.feature_pairs, planes_world, angle_threshold);
        valid_feature_index_pair.insert(valid_pairs.begin(), valid_pairs.end());
    }

    // 经过过滤后，特征对之间的匹配关系发生了改变，因为有很多错误的匹配被滤除了，所以特征对之间的近邻关系也要随之改变
    neighbors_all.clear();
    // 每对图像上匹配的特征点，key = 图像对的id，value=这对图像上匹配的特征id对的集合
    map<pair<size_t,size_t>, set<pair<uint32_t, uint32_t>>> matched_lines;
    for(const pair<uint32_t, uint32_t>& feature_id_pair : valid_feature_index_pair)
    {
        const pair<uint32_t, uint32_t>& feature1 = index_to_feature.find(feature_id_pair.first)->second;
        const pair<uint32_t, uint32_t>& feature2 = index_to_feature.find(feature_id_pair.second)->second;
        matched_lines[{feature1.first, feature2.first}].insert({feature1.second, feature2.second});
        // 重新建立特征之间的近邻关系
        neighbors_all[feature_id_pair.first].push_back(feature_id_pair.second);
        neighbors_all[feature_id_pair.second].push_back(feature_id_pair.first);
    }
    // 经过过滤后剩余的可靠匹配对重新链接，组成新的track
    vector<pair<size_t,size_t>> image_pairs;
    vector<set<pair<uint32_t, uint32_t>>> feature_pairs;
    for(map<pair<size_t,size_t>, set<pair<uint32_t, uint32_t>>>::const_iterator it = matched_lines.begin(); 
            it != matched_lines.end(); it++)
    {
        image_pairs.push_back(it->first);
        feature_pairs.push_back(it->second);
    }
    TrackBuilder tracks_builder(true);
    tracks_builder.Build(image_pairs, feature_pairs);
    tracks_builder.Filter(min_track_length);
    tracks_builder.ExportTracks(tracks);

    return true;
}

std::set<std::pair<uint32_t, uint32_t>> PanoramaLineMatcher::FilterPairsInTrack(const std::set<std::pair<uint32_t, uint32_t>>& init_features,
                                                                                const vector<vector<cv::Vec4f>>& planes_world,
                                                                                const float angle_threshold)
{
    // 过滤后认为可靠的匹配特征的id
    set<pair<uint32_t, uint32_t>> valid_feature_index_pair;
    for(const pair<uint32_t, uint32_t>& feature : init_features)
    {
        const uint32_t& idx1 = feature_to_index.find(feature)->second;
        const vector<uint32_t>& neighbors = neighbors_all.find(idx1)->second;
        if(neighbors.size() < 2)
            continue;
        for(int i = 0; i < neighbors.size(); i++)
        {
            for(int j = i + 1; j < neighbors.size(); j++)
            {
                const uint32_t& idx2 = neighbors[i];
                const uint32_t& idx3 = neighbors[j];
                const pair<uint32_t, uint32_t>& feature2 = index_to_feature.find(idx2)->second;
                const pair<uint32_t, uint32_t>& feature3 = index_to_feature.find(idx3)->second;

                 // 两个平面相交得到的空间直线,这条直线应该在第三个平面上
                Vector6f line = PlaneIntersect(planes_world[feature.first][feature.second], planes_world[feature2.first][feature2.second]);
                Eigen::Vector3f line_norm(line.block<3,1>(3,0));
                Eigen::Vector3f plane_norm(planes_world[feature3.first][feature3.second][0],
                                    planes_world[feature3.first][feature3.second][1],
                                    planes_world[feature3.first][feature3.second][2]);
                // 计算直线和第三个平面的法向量之间的夹角，如果这三条直线都是同一条空间直线的话，那么这个夹角应该接近90度
                // 也就是说直线在第三个平面上
                float angle1 = acos(plane_norm.dot(line_norm) / plane_norm.norm()) * 180.0 / M_PI;
                if(abs(angle1 - 90.0) > angle_threshold)
                    continue;
                // 计算三条直线在世界坐标系下形成的三个平面之间的夹角，如果是同一条直线，那么这三个平面两两之间的夹角都应该比较小
                // 这里选择了三个夹角的中值作为判断标准，因为经过观察发现就算是同一条直线，平面1和平面3之间的夹角一般也会稍大，
                // 这可能是由于直线检测没那么准确+图像位姿不够准确造成的。所以为了对位姿稍微鲁棒一些，选择了夹角的中值作为标准，
                // 这样可以排除掉由于位姿等噪声导致的误差。如果这三条直线匹配错误了，并不是对应于同一条空间直线，那么就会有两个
                // 夹角都特别大，那么用中值也可以判断出来
                angle1 = PlaneAngle(planes_world[feature.first][feature.second].val, planes_world[feature2.first][feature2.second].val) * 180.0 / M_PI;
                float angle2 = PlaneAngle(planes_world[feature.first][feature.second].val, planes_world[feature3.first][feature3.second].val) * 180.0 / M_PI;
                float angle3 = PlaneAngle(planes_world[feature2.first][feature2.second].val, planes_world[feature3.first][feature3.second].val) * 180.0 / M_PI;
                float middle_angle = (max(angle1, angle2) > angle3 ) ? max(min(angle1, angle2),angle3) : max(angle1, angle2);
                if(middle_angle > angle_threshold + 2.0)
                    continue;
                valid_feature_index_pair.insert({idx1, idx2});
                valid_feature_index_pair.insert({idx1, idx3});
            }
        }
    }
    return valid_feature_index_pair;
}

void PanoramaLineMatcher::RemoveParallelLines()
{
    vector<LineTrack> valid_tracks;
    for(const LineTrack& track : tracks)
    {
        LineTrack t(track.id);
        // 所有在同一张图像上的直线特征
        vector<pair<uint32_t, uint32_t>> features_same_image;
        // 可以用迭代器来遍历同一张图像上的直线，这是因为特征是以{image id，line id}的pair存储的，而set在存储pair的时候默认根据
        // pair.first 从小到大排序，所以直接从set的第一个开始遍历，相同的image id会出现在连续的位置上
        for(std::set<std::pair<uint32_t, uint32_t>>::const_iterator it_feature = track.feature_pairs.begin();
                it_feature != track.feature_pairs.end(); it_feature++)
        {
            // 如果没有同一张图像上的特征，或者当前直线特征依然属于同一张图像，那么就直接保存下来
            if(features_same_image.empty() || features_same_image[features_same_image.size()-1].first == it_feature->first)
            {
                features_same_image.push_back(*it_feature);
                continue;
            }
            // 当前特征和以前的特征已经不属于同一张图像了，那么就判断一下之前那些属于同一张图像的直线特征是否平行，
            // 如果不平行的话，那就把这些特征加入新的track中,如果平行的话就舍弃这些特征
            // 最后把当前特征加入到vector中，作为新的一轮开始
            if(!IsParallel(features_same_image))
                t.feature_pairs.insert(features_same_image.begin(), features_same_image.end());
            features_same_image.clear();
            features_same_image.push_back(*it_feature);
        }
        if(!IsParallel(features_same_image))
            t.feature_pairs.insert(features_same_image.begin(), features_same_image.end());
        // 判断当前track包含了多少张不同的图像，如果太少了就要舍弃
        set<uint32_t> image_ids;
        for(const auto& feature : t.feature_pairs)
            image_ids.insert(feature.first);
        if(image_ids.size() >= min_track_length)
            valid_tracks.emplace_back(t);
    }
    valid_tracks.swap(tracks);
    LOG(INFO) << "after remove parallel lines, " << tracks.size() << " image line tracks";
}

bool PanoramaLineMatcher::IsParallel(const std::vector<std::pair<uint32_t, uint32_t>>& features)
{
    if(features.size() < 2)
        return false;
    const PanoramaLine& image_line = image_lines_all[features[0].first];
    Equirectangular eq(image_line.GetImageGray().rows, image_line.GetImageGray().cols);
    for(size_t i = 0; i < features.size(); i++)
    {
        const cv::Vec4f& line1 = image_line.GetLines()[features[i].second];
        cv::Point3f p1 = eq.ImageToCam(cv::Point2f(line1[0], line1[1]));
        cv::Point3f p2 = eq.ImageToCam(cv::Point2f(line1[2], line1[3]));
        cv::Point3f line1_middle = (p1 + p2) / 2.0;
        cv::Vec4f plane1 = FormPlane(p1, p2, cv::Point3f(0,0,0));
        plane1 /= cv::norm(plane1);
        float half_plane_angle = VectorAngle3D(p1, p2, true) / 2.0;     // 平面所对应的圆心角的一半
        for(size_t j = i + 1; j < features.size(); j++)
        {
            const cv::Vec4f& line2 = image_line.GetLines()[features[j].second];
            p1 = eq.ImageToCam(cv::Point2f(line2[0], line2[1]));
            p2 = eq.ImageToCam(cv::Point2f(line2[2], line2[3]));
            // 判断两条直线是否平行的方法：直线L1和球心形成了一个平面，把直线L2的两个端点和中点投影到平面上
            // 如果这三个点中有两个点都落在L1形成的扇形平面内，那么就认为直线L1和直线L2平行。
            // 判断某个点是否落在扇形平面内，可以计算当前点和扇形平面的中点的夹角，如果夹角小于扇形所对圆心角的一半，那就在平面内
            // 这个方法判断平行是有问题的，因为如果L1和L2是以一个较大的角度交叉，那么也能满足上述条件，
            // 但是之前经过TrackFilter后，同一张图像上属于同一个track的直线基本都是小角度交叉，所以可以判断
            cv::Point3f p1_projected = ProjectPointToPlane(p1, plane1, true);
            cv::Point3f p2_projected = ProjectPointToPlane(p2, plane1, true);
            float angle1 = VectorAngle3D(p1_projected, line1_middle);
            float angle2 = VectorAngle3D(p2_projected, line1_middle);
            float angle3 = VectorAngle3D((p1 + p2) / 2.0, line1_middle);
            float middle_angle = (max(angle1, angle2) > angle3 ) ? max(min(angle1, angle2),angle3) : max(angle1, angle2);
            if(middle_angle <= half_plane_angle)
                return true;
        }
    }
    return false;
}

vector<cv::DMatch> PanoramaLineMatcher::MatchesDiff(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2)
{
    set<pair<int,int>> match_pair2;
    for(auto& m : matches2)
        match_pair2.insert({m.queryIdx, m.trainIdx});
    vector<cv::DMatch> matches_diff;
    for(auto& m : matches1)
    {
        if(match_pair2.count({m.queryIdx, m.trainIdx}) == 0)
            matches_diff.push_back(m);
    }
    return matches_diff;
}

std::vector<cv::DMatch> PanoramaLineMatcher::MatchesIntersection(const std::vector<cv::DMatch>& matches1, const std::vector<cv::DMatch>& matches2)
{
    set<pair<int,int>> match_pair2;
    for(auto& m : matches2)
        match_pair2.insert({m.queryIdx, m.trainIdx});
    vector<cv::DMatch> matches_intersect;
    for(auto& m : matches1)
    {
        if(match_pair2.count({m.queryIdx, m.trainIdx}) > 0)
            matches_intersect.push_back(m);
    }
    return matches_intersect;
}

void PanoramaLineMatcher::SetNeighborSize(const int size)
{
    assert(size > 0);
    neighbor_size = size;
}

void PanoramaLineMatcher::SetMinTrackLength(const int length)
{
    assert(length > 0);
    min_track_length = length;
}


const std::vector<LineTrack>& PanoramaLineMatcher::GetTracks() const
{
    return tracks;
}