/*
 * @Author: Diantao Tu
 * @Date: 2021-11-01 14:47:41
 */

#include "PanoramaLine.h"
#include "../base/common.h"
#include <sstream>
#include <Eigen/Core>

using namespace std;

PanoramaLine::PanoramaLine():id(-1)
{}

PanoramaLine::PanoramaLine(const cv::Mat& _img, int _id):id(_id)
{
    if(_img.channels() == 3)    
        cv::cvtColor(_img, img_gray, cv::COLOR_BGR2GRAY);
    else if(_img.channels() == 4)
        cv::cvtColor(_img, img_gray, cv::COLOR_RGBA2GRAY);
    else 
        img_gray = _img.clone(); 
    rows = img_gray.rows;
    cols = img_gray.cols;
}

void PanoramaLine::SetDepthImage(const cv::Mat& depth)
{
    assert(depth.type() == CV_32F);
    img_depth = depth.clone();
}

void PanoramaLine::Detect(float vertical_s, float vertical_e)
{
    assert(vertical_s > vertical_e);
    int row_s = rows * (0.5 - vertical_s / 180);
    int row_e = rows * (0.5 - vertical_e / 180);
    vector<cv::line_descriptor::KeyLine> keylines;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    lbd->detect(img_gray.rowRange(row_s, row_e), keylines);
    for(cv::line_descriptor::KeyLine& kl : keylines)
    {
        kl.startPointY += row_s;
        kl.endPointY += row_s;
        kl.sPointInOctaveY += row_s;
        kl.ePointInOctaveY += row_s;
        kl.pt.y += row_s;
        lines.push_back(cv::Vec4f(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY));
    }
    lbd->compute(img_gray, keylines, descriptor);
    init_keylines = keylines;
    // 把初始小直线到最终直线的映射进行初始化，此时每条最初的小直线唯一对应于最终的直线，
    // 每一条最终的直线都对应于唯一一条初始小直线，因为还没经过后面的过滤融合，自然这种匹配关系
    // 是一一对应的
    init_to_final.resize(init_keylines.size());
    final_to_init.resize(init_keylines.size());
    for(int i = 0; i < init_keylines.size(); i++)
    {
        init_to_final[i].push_back(i);
        final_to_init[i].push_back(i);
    }
}

void PanoramaLine::Detect(const cv::Mat& mask)
{
    assert(mask.rows == rows && mask.cols == cols);
    assert(mask.type() == CV_8U);
    vector<cv::line_descriptor::KeyLine> keylines, valid_keylines;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    lbd->detect(img_gray, keylines);
    
    // 根据mask过滤掉不在范围内的直线，这里还要重新分配class id，因为在计算描述子的时候要求class id是连续的，且从0开始的
    for(cv::line_descriptor::KeyLine& kl : keylines)
    {
        cv::Point2i start_point(round(kl.startPointX), round(kl.startPointY));
        cv::Point2i end_point(round(kl.endPointX), round(kl.endPointY));
        // 如果直线的起点和终点有任意一点不在图像内，那么就直接跳过
        // 会出现这种原因主要是因为前面使用了round函数，可能会超出图像范围，不过是很小概率发生的
        if(max(start_point.x, end_point.x) > img_gray.cols - 1 || 
            min(start_point.x, end_point.x) < 0 || 
            max(start_point.y, end_point.y) > img_gray.rows -1 || 
            min(start_point.y, end_point.y) < 0)
            continue;
        if(mask.at<uchar>(start_point) != 0 && mask.at<uchar>(end_point) != 0)
        {
            kl.class_id = valid_keylines.size();
            valid_keylines.push_back(kl);
            lines.push_back(cv::Vec4f(kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY));
        }
    }
    lbd->compute(img_gray, valid_keylines, descriptor);
    init_keylines = valid_keylines;
    init_to_final.resize(init_keylines.size());
    final_to_init.resize(init_keylines.size());
    for(int i = 0; i < init_keylines.size(); i++)
    {
        init_to_final[i].push_back(i);
        final_to_init[i].push_back(i);
    }
}

bool PanoramaLine::FilterByLength(float length_threshold)
{
    float sq_length = length_threshold * length_threshold;
    vector<cv::Vec4f> lines_valid;
    for(cv::Vec4f l :lines)
    {
        if(PointDistanceSquare(&(l[0]), &(l[2])) > sq_length)
            lines_valid.push_back(l);
    }
    lines.clear();
    lines.swap(lines_valid);
    return true;
}

bool PanoramaLine::FilterByAngle(float angle_threshold)
{
    assert(angle_threshold >= 0 && angle_threshold <= 180);
    // 角度边弧度，减少后面的计算量
    float radius_threshold = angle_threshold * M_PI / 180.f;
    Equirectangular eq(rows, cols);
    vector<cv::Vec4f> lines_valid;
    for(const cv::Vec4f& l : lines)
    {
        // 把起始点和终止点都变换成单位圆上的XYZ坐标
        cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), 1.f);
        cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), 1.f);
        float angle = VectorAngle3D(p1, p2);
        if(angle >= radius_threshold)
            lines_valid.push_back(l);
    }
    lines_valid.swap(lines);
    return true;
}

bool PanoramaLine::FilterByLengthAngle(float length_threshold, float angle_threshold, cv::Mat mask)
{
    assert(angle_threshold >= 0 && angle_threshold <= 180);
    // 角度变弧度，减少后面的计算量
    float radius_threshold = angle_threshold * M_PI / 180.f;
    float sq_length = length_threshold * length_threshold;
    Equirectangular eq(rows, cols);
    vector<cv::Vec4f> lines_valid;
    for(const cv::Vec4f& l : lines)
    {
        // 把起始点和终止点都变换成单位圆上的XYZ坐标
        cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), 1.f);
        cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), 1.f);
        float angle = VectorAngle3D(p1, p2);
        
        if(angle >= radius_threshold || PointDistanceSquare(&(l[0]), &(l[2])) > sq_length)
        {
            lines_valid.push_back(l);
            continue;
        }
        if(mask.empty())
            continue;
        int point_in_mask = 0;
        int point_all = 0;
        const vector<cv::Point2f> segments = eq.BreakToSegments(l, 30);
        // 得到了一段段的直线后，就要把每一段的直线所占据的空间在final_line_occupy_mat上表示出来
        for(int i = 0; i < segments.size() - 1; i++)
        {
            if(abs(segments[i].x - segments[i+1].x) > 0.8 * cols)       
                continue;
            // 找到一个bounding box恰好能包含当前直线，而且要有一定的余量，余量就是 dis_threshold 的大小
            // 而且这个bounding box不能超出图像的尺寸
            cv::Point2f bounding_box_left_top(round(max(min(segments[i].x, segments[i+1].x), 0.f)), 
                                              round(max(min(segments[i].y, segments[i+1].y), 0.f)));
            cv::Point2f bounding_box_right_bottom(round(min(max(segments[i].x, segments[i+1].x), cols-1.f)), 
                                            round(min(max(segments[i].y, segments[i+1].y), rows-1.f)));
            // 用cv::line 把这条直线画出来，用255表示，其他区域就用0表示。所以在后面占据空间里就可以用画出的直线来表明
            // 当前直线经过了哪些像素。这里有个效率问题，因为实际上直线是很短的（30个像素左右长度），没必要在整张图片上
            // 把它画出来，所以就用了一个bounding box，只在这个小区域内画
            cv::Mat occupied_matrix = cv::Mat::zeros(static_cast<int>(bounding_box_right_bottom.y - bounding_box_left_top.y), 
                                                    static_cast<int>(bounding_box_right_bottom.x - bounding_box_left_top.x),
                                                    CV_8U);
            cv::line(occupied_matrix, segments[i] - bounding_box_left_top, segments[i+1] - bounding_box_left_top, 255);
            for(int row = 0; row < occupied_matrix.rows; row++)
            {
                for(int col = 0; col < occupied_matrix.cols; col++)
                {
                    if(occupied_matrix.at<uchar>(row, col) > 0)
                    {
                        point_all ++;
                        point_in_mask += (mask.at<uchar>(row + int(bounding_box_left_top.y), col + int(bounding_box_left_top.x)) > 0);
                    }
                }
            }
        } 
        if(1.f * point_in_mask / point_all > 0.7)
            lines_valid.push_back(l);  
    }
    lines_valid.swap(lines);
    return true;
}

int PanoramaLine::FilterByNCC(const float ncc_threshold)
{
    size_t lines_before = lines.size();
    Equirectangular eq(rows, cols);
    vector<cv::Vec4f> lines_valid;
    for(const cv::Vec4f& l : lines)
    {
        float segment_length = sqrt(PointDistanceSquare(l)) / 10.f;
        if(segment_length > 60)
            segment_length = 60;
        else if(segment_length < 15)
            segment_length = 15;
        vector<cv::Point2f> point_on_line = eq.BreakToSegments(l, segment_length);
        int start_idx = 0, end_idx = point_on_line.size() - 1;
        
        // 从第一个点开始，计算当前点和之前的每一个点之间的NCC评分，如果在第i个点处出现了NCC和之前相差较大，也就是说
        // 第i个点和之前的i-1个点相似度较低，那么前面的i-1个点就可以形成一个新的直线，而第i个点没准能和后面的那些点形成
        // 新的直线。因此此时的start idx要更新为i, 而且对于第j个点(j>i)来说，已经不需要考虑start idx之前的点了
        for(size_t i = 1; i < point_on_line.size(); i++)
        {
            for(int j = i - 1; j >= start_idx ; j--)
            {
                float ncc = ComputeNCC(img_gray, img_gray, point_on_line[i], point_on_line[j]);
                if(ncc < ncc_threshold)
                {
                    if(i - 1 - start_idx > 0)
                        lines_valid.push_back(cv::Vec4f(point_on_line[i-1].x, point_on_line[i-1].y, 
                                                point_on_line[start_idx].x, point_on_line[start_idx].y));
                    start_idx = i;  // 更新了start_idx后，自然会导致不满足for循环的条件 j>=start_idx 
                }
            }
        }
        // 要把最后一个点和start idx组成的直线放进去
        if(point_on_line.size() - 1 - start_idx > 0)
            lines_valid.push_back(cv::Vec4f(point_on_line[point_on_line.size()-1].x, 
                                            point_on_line[point_on_line.size()-1].y, 
                                            point_on_line[start_idx].x, 
                                            point_on_line[start_idx].y));
        
    }
    lines.swap(lines_valid);
    return lines_before - lines.size();
}

bool PanoramaLine::FileterByInitLine(int count_threshold, float cover_threshold)
{
    if(init_to_final.empty() || final_to_init.empty())
        return false;
    vector<cv::Vec4f> good_lines;
    vector<vector<size_t>> good_final_to_init;
    Equirectangular eq(rows, cols);
    for(size_t i = 0; i < lines.size(); i++)
    {
        if(final_to_init[i].size() >= count_threshold)
        {
            good_lines.push_back(lines[i]);
            good_final_to_init.push_back(final_to_init[i]);
            continue;
        }
        vector<cv::Point2f> segments = eq.BreakToSegments(lines[i], 30);
        float final_line_length = 0, init_line_length = 0;
        for(size_t j = 0; j < segments.size() - 1; j++)
        {
            if(abs(segments[j].x - segments[j+1].x) > 0.8 * cols)       
                continue;
            final_line_length += sqrt(PointDistanceSquare(segments[j], segments[j+1]));
        }
        for(const size_t& init_line_idx : final_to_init[i])
            init_line_length += sqrt(PointDistanceSquare(cv::Vec4f(init_keylines[init_line_idx].startPointX, init_keylines[init_line_idx].startPointY,
                                            init_keylines[init_line_idx].endPointX, init_keylines[init_line_idx].endPointY)));
        if(init_line_length > cover_threshold * final_line_length)
        {
            good_lines.push_back(lines[i]);
            good_final_to_init.push_back(final_to_init[i]);
        }
    }
    good_final_to_init.swap(final_to_init);
    good_lines.swap(lines);
    init_to_final.clear();
    init_to_final.resize(init_keylines.size(), vector<size_t>());
    for(size_t final_line_id = 0; final_line_id < lines.size(); final_line_id++)
    {
        const vector<size_t>& each_line_to_init = final_to_init[final_line_id];
        for(const size_t& init_id : each_line_to_init)
            init_to_final[init_id].push_back(final_line_id);
    }

    return true;
    // 以下为可视化，分别显示最终的直线对应于哪些初始直线
    vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                cv::Scalar(255,0,255)};
    // 计算初始的直线和最终直线对应的平面参数
    vector<cv::Vec4f> init_planes;
    vector<cv::Vec4f> init_lines;
    for(const cv::line_descriptor::KeyLine& key_line : init_keylines)
    {
        init_lines.push_back(cv::Vec4f(key_line.startPointX, key_line.startPointY, key_line.endPointX, key_line.endPointY));
    }
    LinesToPlane(init_lines, init_planes, *(new vector<cv::Point3f>()));
    for(int i = 0; i < lines.size(); i++)
    {
        cv::Mat img_line;
        cv::cvtColor(img_gray, img_line, cv::COLOR_GRAY2BGR);
        DrawLine(img_line, lines[i], cv::Scalar(0,0,255), 5, true);
        cv::imwrite("final_line-" + num2str(i) + ".jpg", img_line);
        vector<cv::Vec4f> curr_init_lines;
        for(const size_t& idx : final_to_init[i])
            curr_init_lines.push_back(init_lines[idx]);
        img_line = DrawLinesOnImage(img_gray, curr_init_lines, colors, 3, true);
        cv::imwrite("final_line-" + num2str(i) + "_1.jpg", img_line);
    }
    return true;
}

bool PanoramaLine::LinesToPlane(const std::vector<cv::Vec4f>& lines, std::vector<cv::Vec4f>& planes, std::vector<cv::Point3f>& points_in_sphere)
{
    if(lines.empty())
        return false;
    planes.clear();
    points_in_sphere.clear();
    Equirectangular eq(rows, cols);
    for(const cv::Vec4f& l : lines)
    {
        // 把起始点和终止点都变换成球上的XYZ坐标
        cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), float(5.0));
        cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), float(5.0));
        cv::Point3f p3(0,0,0);
        float a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
 
        float b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
    
        float c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );
        cv::Vec4f plane = cv::Vec4f(a,b,c,0);
        plane /= cv::norm(plane);   // 变成单位向量
        planes.push_back(plane); 
        points_in_sphere.push_back(p1);
        points_in_sphere.push_back(p2);
    }
    return true;
}


cv::Vec4f PanoramaLine::Fuse(std::vector<cv::Vec4f> line_group, bool ransac)
{
    // 第一种方法比较简单，找到所有端点里相距最远的两个或最近的两个点，然后连起来即可
    float max_length = 0;

    int start = 0, end = 0;
    vector<cv::Vec2f> points;
    for(cv::Vec4f line : line_group)
    {
        points.push_back(cv::Vec2f(line[0], line[1]));
        points.push_back(cv::Vec2f(line[2], line[3]));
    }
    for(size_t i = 0; i < points.size(); i++)
    {
        for(size_t j = i + 1; j < points.size(); j++)
        {
            float distance = PointDistanceSquare(&(points[i][0]), &(points[j][0]));
            if(distance > max_length)
            {
                max_length = distance;
                start = i;
                end = j;
            }
        }
    }
    if(!ransac)
        return cv::Vec4f(points[start][0], points[start][1], points[end][0], points[end][1]);
    
    // 另一种方法是使用RANSAC来拟合直线，拟合直线后的长度为上面计算得到的最远的两个端点之间的距离
    // 线段的起始点和终止点就是那两个端点到拟合直线的垂直投影
    Equirectangular eq(rows, cols);
    vector<cv::Point3f> points_3d;
    for(cv::Vec4f l : line_group)
    {
        // 把起始点和终止点都变换成单位圆上的XYZ坐标
        cv::Point3f p1 = eq.ImageToCam(cv::Point2f(l[0], l[1]), float(5.0));
        cv::Point3f p2 = eq.ImageToCam(cv::Point2f(l[2], l[3]), float(5.0));
        points_3d.push_back(p1);
        points_3d.push_back(p2);

    }
    cv::Vec4f plane = FindPlaneRansac(points_3d);
    cv::Point2f p1 = eq.CamToImage(ProjectPointToPlane(points_3d[start], plane));
    cv::Point2f p2 = eq.CamToImage(ProjectPointToPlane(points_3d[end], plane));
    return cv::Vec4f(p1.x, p1.y, p2.x, p2.y);
}


int PanoramaLine::Fuse(float ncc_threshold, bool visualization)
{
    vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                            cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                            cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                            cv::Scalar(255,0,255)};                         // 紫
    int size_line = lines.size();
    cv::Mat img_line;
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_0.jpg", img_line);
    } 
    // 第一步：融合离得很近的直线
    int num_fuse = FuseNearLines(false);
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_1.jpg", img_line); 
    }
    // 第二步：一直融合这些离得近的直线，直到没办法所有离得近的直线都融合在一起了
    // 此时图像上就从小直线变成了中直线
    while (num_fuse != 0)
    {
        num_fuse = FuseNearLines(true);
        if(visualization)
        {
            img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
            cv::imwrite("img_line_filtered_2.jpg", img_line); 
        }
    }    
    cv::Mat belief_mat = OccupiedMatrix();
    if(visualization)
        cv::imwrite("belief.jpg", belief_mat);
    
    // 第三步：融合离得远的直线，离得远的那些直线有的是属于同一边缘的，有的不是，因此需要对距离和NCC做出限制
    // 防止有错误的融合
    while(FuseFarLines(400, ncc_threshold));
    // DrawEachLine("./", img_gray, lines, cv::Scalar(0,0,255), 4, true);
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_3.jpg", img_line); 
    }

    // 第四步：进行一些过滤，如果到了这一步还有一些小直线，那么基本能肯定这些小直线是错误检测或者是边缘太过微弱了
    // 这一类的直线就直接过滤掉即可，还可以加快后面的计算速度
    FilterByLengthAngle(rows / 20, 20);
    // 第五步：融合由于二维图像的分割而导致断开的直线，也就是说直线一段在图像的最左边，一段在最右边
    // FuseBoundaryLines();
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_4.jpg", img_line); 
    }
    // DrawEachLine("./", img_gray, lines, cv::Scalar(0,0,255), 4, true);
    // 第六步：在第三步的时候，有些距离远的直线并不是同一个空间直线，但是它们恰好处在同一个平面上，就被融合了
    // 所以需要一个更严格的NCC检测，把这些长直线分成一段段的短直线，每个短直线基本能保证是同一个物理边缘的
    // FilterByNCC(0);
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_4_5.jpg", img_line); 
    }
    // DrawEachLine("./", img_gray, lines, cv::Scalar(0,0,255), 4, true);
    // 第七步：经过上一步，就会出现很多短直线，这些就是那些错误链接的直线剩余的部分，因此要过滤掉
    FilterByLengthAngle(rows / 20, 20, belief_mat);
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_5.jpg", img_line);
    }
    // 第八步：上一步过滤短直线之后，剩下的都是长直线，这些直线还是有一部分属于同一个物理边缘的，那么就以一个较宽松的条件
    // 把这些直线连接起来
    FuseFarLines(400, ncc_threshold);

    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_6.jpg", img_line);
    }
    while(FuseOverlapLines());

    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_7.jpg", img_line);
    }
    // 对直线过滤分成两个步骤，第一步过滤掉很小的直线，无论这个直线是否可靠
    // 第二步就过滤掉一些长一点的直线，但是会根据置信度来保留一些长度或角度低于阈值的直线
    FilterByLengthAngle(rows / 30, 10);
    FilterByLengthAngle(rows / 20, 10, belief_mat);
    if(visualization)
    {
        img_line = DrawLinesOnImage(img_gray, lines, colors, 3, true);
        cv::imwrite("img_line_filtered_8.jpg", img_line);
    }
    SetLineMap(6);
    FileterByInitLine(6, 0.7);
  
    // 释放灰度图，节省内存
    img_gray.release();
    return size_line - lines.size();
}

// 融合离得近的线段，也就是同一个线被分成了很多段
int PanoramaLine::FuseNearLines(bool ncc)
{
    Equirectangular eq(rows, cols);
    int size_line = lines.size();
    cv::Mat points(2 * size_line, 2, CV_32F);
    for(int i = 0; i < size_line; i++)
    {
        points.at<float>(2*i, 0) = lines[i][0];
        points.at<float>(2*i, 1) = lines[i][1];
        points.at<float>(2*i+1, 0) = lines[i][2];
        points.at<float>(2*i+1, 1) = lines[i][3];
    }
    vector<cv::Vec4f> planes;   // ax+by+cz+d=0 存储 a b c d
    LinesToPlane(lines, planes, *(new vector<cv::Point3f>()));

    // 建立一棵kd树，衡量距离默认使用L2距离
    cv::flann::Index kdtree(points, cv::flann::KDTreeIndexParams(1)); //此部分建立kd-tree索引同上例，故不做详细叙述

    /**预设knnSearch所需参数及容器**/
    unsigned queryNum = 4;                  // 用于设置返回邻近点的个数
    vector<float> curr_point(2);            // 存放 查询点 的容器（本例都是vector类型）
    vector<int> vecIndex(queryNum);         // 存放返回的点索引
    vector<float> vecDist(queryNum);        // 存放距离
    cv::flann::SearchParams params(32);     // 设置knnSearch搜索参数
    float threshold_dist = (rows / 55.0) * (rows / 55.0);

    vector<vector<int>> neighbor_idx(size_line);
    for(size_t i = 0; i < size_line; i++)
    {
        const cv::Vec4f& curr_line = lines[i];
        const cv::Vec4f& curr_plane = planes[i];
        // 找到距离当前线段起点和终点最近的两个点，k=0是起点，k=2是终点
        for(size_t k = 0; k <= 2; k += 2)
        {
            curr_point = {curr_line[k], curr_line[k+1]};
            kdtree.knnSearch(curr_point, vecIndex, vecDist, queryNum, cv::flann::SearchParams(-1));
            // 跳过第一个最近邻点，从第二个开始，因为第一个最近邻点永远是自己
            for(size_t j = 1; j < vecDist.size(); j++)
            {
                int point_idx = vecIndex[j];
                if(vecDist[j] > threshold_dist)
                    break;
                // 近邻点所属于的直线，如果近邻点所在的直线就是当前直线，那么直接跳过
                int line_idx = point_idx / 2;
                if(line_idx == i)
                    continue;
                const cv::Vec4f& neighbor_plane = planes[line_idx];
                // 计算两个平面的夹角
                float diff_angle = PlaneAngle(neighbor_plane.val, curr_plane.val, true) * 180.0 / M_PI;
                if(abs(diff_angle) > 3)
                    continue;
                // 之后要用到很多和两个直线的起点以及终点相关的方法，所以先把他们的起点终点单独写出来
                // sp = start point     ep = end point
                cv::Point2f l1_sp = cv::Point2f(lines[i][0], lines[i][1]);
                cv::Point2f l1_ep = cv::Point2f(lines[i][2], lines[i][3]);
                cv::Point2f l2_sp = cv::Point2f(lines[line_idx][0], lines[line_idx][1]);
                cv::Point2f l2_ep = cv::Point2f(lines[line_idx][2], lines[line_idx][3]);
                // 要求两条直线要基本在两侧，也就是是说两条直线不能有太多重合的部分
                if(true)
                {
                    cv::Point2f l1_sp_proj = ProjectPoint2Line2D(lines[line_idx], l1_sp);
                    cv::Point2f l1_ep_proj = ProjectPoint2Line2D(lines[line_idx], l1_ep);
                    cv::Point2f l2_sp_proj = ProjectPoint2Line2D(lines[i], l2_sp);
                    cv::Point2f l2_ep_proj = ProjectPoint2Line2D(lines[i], l2_ep);
                    
                    int direction_sp1_sp2 = (l2_sp_proj.y - l1_sp.y) > 0 ? 1 : -1;
                    int direction_sp1_ep2 = (l2_ep_proj.y - l1_sp.y) > 0 ? 1 : -1;
                    int direction_ep1_sp2 = (l2_sp_proj.y - l1_ep.y) > 0 ? 1 : -1;
                    int direction_ep1_ep2 = (l2_ep_proj.y - l1_ep.y) > 0 ? 1 : -1;
                    if(!(direction_sp1_sp2 == direction_sp1_ep2 && direction_sp1_sp2 == direction_ep1_sp2
                        && direction_sp1_sp2 == direction_ep1_ep2))
                        continue;
                    
                    float d1 = PointDistanceSquare(l1_sp, l1_sp_proj);
                    float d2 = PointDistanceSquare(l1_ep, l1_ep_proj);
                    float d3 = PointDistanceSquare(l2_sp, l2_sp_proj);
                    float d4 = PointDistanceSquare(l2_ep, l2_ep_proj);

                    if(!((d1 < 20 * 20 && d2 < 20 * 20) || (d3 < 20 * 20 && d4 < 20 * 20)))
                        continue;

                }
                if(ncc)
                {
                    float d1 = ComputeNCC(img_gray, img_gray, l1_sp, l2_sp);
                    float d2 = ComputeNCC(img_gray, img_gray, l1_sp, l2_ep);
                    float d3 = ComputeNCC(img_gray, img_gray, l1_ep, l2_sp);
                    float d4 = ComputeNCC(img_gray, img_gray, l1_ep, l2_ep);
                    if(min(d1,min(d2,min(d3,d4))) < 0)
                        continue;
                } 
                // 如果有深度图，那么就把想要匹配的直线的起点和终点变换到空间下，然后判断这4个点是否能成为同一条直线
                // 由于深度图范围比较小，所以可能存在只有三个点有深度的情况，那这种情况下就只判断三个点即可，如果只有两个
                // 点有深度,那就跳过即可
                if(!img_depth.empty())
                {
                    cv::Point2i pt1(round(lines[i][0]), round(lines[i][1]));
                    cv::Point2i pt2(round(lines[i][2]), round(lines[i][3]));
                    cv::Point2i pt3(round(lines[line_idx][0]), round(lines[line_idx][1]));
                    cv::Point2i pt4(round(lines[line_idx][2]), round(lines[line_idx][3]));
                    eigen_vector<Eigen::Vector3f> points;
                    float d1 = img_depth.at<float>(pt1);
                    if(d1 > 0)
                    {
                        cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[i][0], lines[i][1]), d1);
                        points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                    }
                    float d2 = img_depth.at<float>(pt2);
                    if(d2 > 0)
                    {
                        cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[i][2], lines[i][3]), d2);
                        points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                    }
                    float d3 = img_depth.at<float>(pt3);
                    if(d3 > 0)
                    {
                        cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[line_idx][0], lines[line_idx][1]), d3);
                        points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                    }
                    float d4 = img_depth.at<float>(pt4);
                    if(d4 > 0)
                    {
                        cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[line_idx][2], lines[line_idx][3]), d4);
                        points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                    }
                    if(points.size() >= 3)
                    {
                        // if(points.size() == 4)
                        //     LOG(INFO) << "points size " << points.size();
                        Eigen::Matrix<float, 6, 1> line_coeff = FormLine(points, 5.f);
                        // 如果形成的直线的方向向量为0，就说明没法形成直线，那么就直接跳过
                        if(line_coeff.isZero())
                        {
                            // LOG(INFO) << "bad line";
                            continue;
                        }
                    }
                }

                neighbor_idx[i].push_back(line_idx);
            }
        }
    }

    vector<bool> fused(size_line, false);
    vector<cv::Vec4f> lines_fused;
    size_t group_count = 0;
    // neighbor_idx 里存储了每个线段的近邻线段，要根据线段之间的关系把要融合的线段放在同一个group里
    for(size_t line_idx = 0; line_idx < size_line; line_idx++)
    {
        if(fused[line_idx])
            continue;
        vector<cv::Vec4f> group_lines;
        group_lines = FindNeighbors(fused, neighbor_idx, line_idx);
        lines_fused.push_back(Fuse(group_lines));
    }
    lines.clear();
    lines.swap(lines_fused);
  
    return size_line - lines.size();

}

// 融合离得远的线段，也就是一条物理边缘被多次检测到
int PanoramaLine::FuseFarLines(float dist_threshold, float ncc_threshold)
{
    float sq_dist = dist_threshold * dist_threshold;
    Equirectangular eq(rows, cols);
    size_t size_line = lines.size();
    vector<cv::Vec4f> planes;   // ax+by+cz+d=0 存储 a b c d
    vector<cv::Point3f> points;
    LinesToPlane(lines, planes, points);
    vector<vector<int>> neighbor_idx(size_line);
    for(size_t i = 0; i < planes.size(); i++)
    {
        const cv::Vec4f& curr_plane = planes[i];
        for(size_t j = i + 1; j < planes.size(); j++)
        {
            const cv::Vec4f& neighbor_plane = planes[j];
            float diff_angle = PlaneAngle(neighbor_plane.val, curr_plane.val, true) * 180.0 / M_PI;
            if(abs(diff_angle) > 2)
                continue;
            if(PointToPlaneDistance(curr_plane, points[2 * j], true) > 0.05)
                continue;
            if(PointToPlaneDistance(curr_plane, points[2 * j + 1], true) > 0.05)
                continue;
            if(PointToPlaneDistance(neighbor_plane, points[2 * i], true) > 0.05)
                continue;
            if(PointToPlaneDistance(neighbor_plane, points[2 * i + 1], true) > 0.05)
                continue;
            float d1 = VectorAngle3D(points[2 * i], points[2 * j]) * 180.0 / M_PI;
            float d2 = VectorAngle3D(points[2 * i], points[2 * j + 1]) * 180.0 / M_PI;
            float d3 = VectorAngle3D(points[2 * i + 1], points[2 * j]) * 180.0 / M_PI;
            float d4 = VectorAngle3D(points[2 * i + 1], points[2 * j + 1]) * 180.0 / M_PI;
            if(min(d1,min(d2,min(d3,d4))) > 12.0)  
                continue;
            d1 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[j][0], lines[j][1]));
            d2 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[j][2], lines[j][3]));
            d3 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][2], lines[i][3]), cv::Point2f(lines[j][0], lines[j][1]));
            d4 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][2], lines[i][3]), cv::Point2f(lines[j][2], lines[j][3]));
            // 要求至少有两个点相似度达到较高水平，不然就不能认为这两个线段属于同一边缘
            if(max(d1,max(d2,max(d3,d4))) < ncc_threshold)
                continue; 
            // 相似度的下限是-0.4，也就是说四个顶点之间差异不能特别大
            if(min(d1,min(d2,min(d3,d4))) < -0.4)
                continue; 
            if(!img_depth.empty())
            {
                cv::Point2i pt1(round(lines[i][0]), round(lines[i][1]));
                cv::Point2i pt2(round(lines[i][2]), round(lines[i][3]));
                cv::Point2i pt3(round(lines[j][0]), round(lines[j][1]));
                cv::Point2i pt4(round(lines[j][2]), round(lines[j][3]));
                eigen_vector<Eigen::Vector3f> points;
                float d1 = img_depth.at<float>(pt1);
                if(d1 > 0)
                {
                    cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[i][0], lines[i][1]), d1);
                    points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                }
                float d2 = img_depth.at<float>(pt2);
                if(d2 > 0)
                {
                    cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[i][2], lines[i][3]), d2);
                    points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                }
                float d3 = img_depth.at<float>(pt3);
                if(d3 > 0)
                {
                    cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[j][0], lines[j][1]), d3);
                    points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                }
                float d4 = img_depth.at<float>(pt4);
                if(d4 > 0)
                {
                    cv::Point3f p = eq.ImageToCam(cv::Point2f(lines[j][2], lines[j][3]), d4);
                    points.push_back(Eigen::Vector3f(p.x, p.y, p.z));
                }
                if(points.size() >= 3)
                {
                    Eigen::Matrix<float, 6, 1> line_coeff = FormLine(points, 5.f);
                    // 如果形成的直线的方向向量为0，就说明没法形成直线，那么就直接跳过
                    if(line_coeff.isZero())
                        continue;
                }
            }
            neighbor_idx[i].push_back(j);
        }
    }

    vector<bool> fused(size_line, false);
    vector<cv::Vec4f> lines_fused;
    size_t group_count = 0;

    for(size_t line_idx = 0; line_idx < size_line; line_idx++)
    {
        if(fused[line_idx])
            continue;
        vector<cv::Vec4f> group_lines;
        group_lines = FindNeighbors(fused, neighbor_idx, line_idx);
        lines_fused.push_back(Fuse(group_lines));
    }
    lines.clear();
    lines.swap(lines_fused);
  
    return size_line - lines.size();
}

int PanoramaLine::FuseOverlapLines()
{
    Equirectangular eq(rows, cols);
    size_t size_line = lines.size();
    vector<cv::Vec4f> planes;   // ax+by+cz+d=0 存储 a b c d
    vector<cv::Point3f> end_points;     // 直线的端点
    LinesToPlane(lines, planes, end_points);
    vector<vector<cv::Point3f>> sample_points;  // 对直线进行采样，变成一个个在单位球上的点
    for(size_t line_idx = 0; line_idx < size_line; line_idx++)
    {
        // 由于一条直线在全景图像上是曲线，首先把这条直线分成一段段的小直线来近似
        vector<cv::Point2f> segments = eq.BreakToSegments(lines[line_idx], 30);
        vector<cv::Point3f> points;
        for(const cv::Point2f& pt : segments)
            points.push_back(eq.ImageToCam(pt));
        sample_points.push_back(points);
    }
    vector<vector<int>> neighbor_idx(size_line);
    for(size_t i = 0; i < planes.size(); i++)
    {
        const cv::Vec4f& curr_plane = planes[i];
        cv::Point3f curr_middle = (end_points[2*i] + end_points[2*i+1]) / 2.0;
        const float line_scope = VectorAngle3D(end_points[2*i], end_points[2*i+1]) / 2.0;
        for(size_t j = 0; j < planes.size(); j++)
        {
            if(j == i)  continue;
            const cv::Vec4f& neighbor_plane = planes[j];
            float diff_angle = PlaneAngle(neighbor_plane.val, curr_plane.val) * 180.0 / M_PI;
            if(abs(diff_angle) > 3)
                continue;
            vector<float> distance;
            for(const cv::Point3f& pt : sample_points[j])
            {
                cv::Point3f pt_projected = ProjectPointToPlane(pt, curr_plane); 
                if(VectorAngle3D(pt_projected, curr_middle) > line_scope)
                    continue;
                distance.push_back(VectorAngle3D(pt, pt_projected) * 180.0 / M_PI);
            }
            if(distance.size() < sample_points[j].size() / 2)
                continue;
            nth_element(distance.begin(), distance.begin() + distance.size()/2, distance.end());
            if(distance[distance.size()/2] > 1)
                continue;
            
            neighbor_idx[i].push_back(j);
        }
    }
    vector<bool> fused(size_line, false);
    vector<cv::Vec4f> lines_fused;
    size_t group_count = 0;

    for(size_t line_idx = 0; line_idx < size_line; line_idx++)
    {
        if(fused[line_idx])
            continue;
        vector<cv::Vec4f> group_lines;
        group_lines = FindNeighbors(fused, neighbor_idx, line_idx);
        lines_fused.push_back(Fuse(group_lines));
    }
    lines.clear();
    lines.swap(lines_fused);
  
    return size_line - lines.size();
}

int PanoramaLine::FuseBoundaryLines()
{
    Equirectangular eq(rows, cols);
    size_t size_line = lines.size();
    vector<cv::Vec4f> planes;   // ax+by+cz+d=0 存储 a b c d
    vector<cv::Point3f> points;
    LinesToPlane(lines, planes, points);
    vector<vector<int>> neighbor_idx(size_line);
    for(size_t i = 0; i < planes.size(); i++)
    {
        cv::Vec4f curr_plane = planes[i];
        cv::Vec4f curr_line = lines[i];
        // 当前点的起始点或终止点只要任一点在图像两侧，就不会continue
        if(curr_line[0] >= 200 && curr_line[0] <= cols - 200 
            && curr_line[2] >= 200 && curr_line[2] <= cols - 200)
            continue;

        for(size_t j = i + 1; j < planes.size(); j++)
        {
            cv::Vec4f neighbor_plane = planes[j];
            cv::Vec4f neighbor_line = lines[j];
            if(neighbor_line[0] >= 200 && neighbor_line[0] <= cols - 200 
                && neighbor_line[2] >= 200 && neighbor_line[2] <= cols - 200)
                continue;
            float diff_angle = PlaneAngle(neighbor_plane.val, curr_plane.val);
            if(abs(diff_angle) > 5)
                continue;
            if(PointToPlaneDistance(curr_plane, points[2 * j]) > 0.2)
                continue;
            if(PointToPlaneDistance(curr_plane, points[2 * j + 1]) > 0.2)
                continue;

            float d1 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[j][0], lines[j][1]));
            float d2 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][0], lines[i][1]), cv::Point2f(lines[j][2], lines[j][3]));
            float d3 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][2], lines[i][3]), cv::Point2f(lines[j][0], lines[j][1]));
            float d4 = ComputeNCC(img_gray, img_gray, cv::Point2f(lines[i][2], lines[i][3]), cv::Point2f(lines[j][2], lines[j][3]));
            if(min(d1,min(d2,min(d3,d4))) < -0.2)
                continue; 
            neighbor_idx[i].push_back(j);

        }
    }

    vector<bool> fused(size_line, false);
    vector<cv::Vec4f> lines_fused;
    size_t group_count = 0;

    for(size_t line_idx = 0; line_idx < size_line; line_idx++)
    {
        if(fused[line_idx])
            continue;
        vector<cv::Vec4f> group_lines;
        group_lines = FindNeighbors(fused, neighbor_idx, line_idx);
        if(group_lines.size() < 2)
        {
            lines_fused.push_back(group_lines[0]);
            continue;
        }
        
        vector<cv::Vec2f> points;
        for(cv::Vec4f line : group_lines)
        {
            points.push_back(cv::Vec2f(line[0], line[1]));
            points.push_back(cv::Vec2f(line[2], line[3]));
        }
        // 融合两条位于图像两侧的直线的时候，要找到距离最近的两个端点，而且这两个端点不能属于同一条直线
        float min_distance = FLT_MAX;
        int start = 0, end = 0;
        for(size_t i = 0; i < points.size(); i++)
        {
            for(size_t j = i + 1; j < points.size(); j++)
            {
                if(i / 2 == j / 2)
                    continue;
                float distance = PointDistanceSquare(&(points[i][0]), &(points[j][0]));
                if(distance < min_distance)
                {
                    min_distance = distance;
                    start = i;
                    end = j;
                }
            }
        }
        lines_fused.push_back(cv::Vec4f(points[start][0], points[start][1], points[end][0], points[end][1]));
    }
    lines.clear();
    lines.swap(lines_fused);
  
    return size_line - lines.size();

}

// neighbor_idx 里存储了每个线段的近邻线段，要根据线段之间的关系把要融合的线段放在同一个group里
// 1 近邻 2 3 ；  2 近邻 3 ； 3 近邻 1 4 ； 4近邻 5； 5 近邻 4
// 那么 1 2 3 4 5 就应该属于同一个group
// 为了有序地遍历这种结构，使用了栈
// fused 用来指示这个线段有没有被其他的group选中
// line_idx 表示当前选的group是以第line_idx为核心的
std::vector<cv::Vec4f> PanoramaLine::FindNeighbors(vector<bool>& fused, std::vector<std::vector<int>> neighbor_idx, size_t line_idx)
{
    set<int> group;
    vector<int> stack;
    vector<int> neighbors = neighbor_idx[line_idx];
    vector<cv::Vec4f> group_lines;
    if(neighbors.empty())
    {
        group_lines.push_back(lines[line_idx]);
        return group_lines;
    }
    stack.push_back(line_idx);
    while (!stack.empty())
    {
        int idx = stack[stack.size() - 1];
        stack.pop_back();
        if(fused[idx])
            continue;
        fused[idx] = true;
        neighbors = neighbor_idx[idx];
        stack.insert(stack.end(), neighbors.begin(), neighbors.end());
        group.insert(idx);
    }

    for(const int& g : group)
        group_lines.push_back(lines[g]);
    return group_lines;
}

cv::Vec4f PanoramaLine::FindPlaneRansac(std::vector<cv::Point3f> points)
{
    cv::RNG rng;
    size_t max_inlier = 0;
    vector<cv::Point3f> inlier;
    vector<cv::Point3f> best_inlier;
    cv::Vec4f bset_plane;
    for(int i = 0; i < 50; i++)
    {
        size_t idx1 = rng.uniform(0, points.size());
        size_t idx2 = rng.uniform(0, points.size());
        while (idx1 == idx2)
        {
            idx2 = rng.uniform(0, points.size());
        }
        cv::Point3f p1 = points[idx1];
        cv::Point3f p2 = points[idx2];
        cv::Point3f p3(0,0,0);
        float a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
 
        float b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
    
        float c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );
        cv::Vec4f plane(a,b,c,0);

        inlier.clear();
        
        for(int j = 0; j < points.size(); j++)
        {
            if(PointToPlaneDistance(plane, points[j]) < 0.05)
                inlier.push_back(points[j]);
        }
        if(inlier.size() > max_inlier)
        {
            max_inlier = inlier.size();
            best_inlier.swap(inlier);
            bset_plane = plane;
        }
    }
    if(max_inlier < points.size() / 2)
        return cv::Vec4f(0,0,0,0);
    size_t idx1 = rng.uniform(0, best_inlier.size());
    size_t idx2 = rng.uniform(0, best_inlier.size());
    while (idx1 == idx2)
    {
        idx2 = rng.uniform(0, best_inlier.size());
    }
    cv::Point3f p1 = best_inlier[idx1];
    cv::Point3f p2 = best_inlier[idx2];
    cv::Point3f p3(0,0,0);
    float a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );

    float b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );

    float c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );
    // 用最小二乘法解一个平面
    cv::Point3f center(0,0,0);
    for(cv::Point3f& p : inlier)
        center += p;
    center.x /= inlier.size();
    center.y /= inlier.size();
    center.z /= inlier.size();
    cv::Matx33f cov_mat = cv::Matx33f::zeros();
    for(cv::Point3f p : inlier)
    {
        p -= center;
        cv::Vec3f vec_p(p);
        cov_mat += vec_p * vec_p.t();
    }
    cv::Mat eigen_values, eigen_vectors;
    cv::eigen(cov_mat, eigen_values, eigen_vectors);
    a = eigen_vectors.at<float>(2,0);
    b = eigen_vectors.at<float>(2,1);
    c = eigen_vectors.at<float>(2,2);

    // return bset_plane;
    return cv::Vec4f(a,b,c,0);
}


cv::Mat PanoramaLine::OccupiedMatrix(int line_width)
{
    cv::Mat occupied_mat = cv::Mat::zeros(img_gray.size(), CV_8U);
    Equirectangular eq(rows, cols);
    for(const cv::Vec4f& l : lines)
    {
        vector<cv::Point2f> segments = eq.BreakToSegments(l, 30);
        for(int i = 0; i < segments.size() - 1; i++)
        {
            if(abs(segments[i].x - segments[i+1].x) > 0.8 * cols)       
                continue;
            cv::line(occupied_mat, segments[i], segments[i+1], 255, line_width);
        } 
    }
    return occupied_mat;
}

bool PanoramaLine::SetLineMap(int dis_threshold)
{
    assert(dis_threshold > 0);
    final_to_init.clear();
    final_to_init.resize(lines.size());
    init_to_final.clear();
    init_to_final.resize(init_keylines.size());
    // 这是一个矩阵，外面两层的vector分别用来表示行列，矩阵里的每个元素是一个set
    // 这个矩阵大小和当前图像尺寸相同，每个像素位置上的元素代表着这个像素被哪些直线包含了
    vector<vector<set<uint16_t>>> final_line_occupy_mat;
    final_line_occupy_mat.resize(rows, vector<set<uint16_t>>(cols, *(new set<uint16_t>())));
    Equirectangular eq(rows, cols);
    for(uint16_t final_line_idx = 0; final_line_idx < lines.size(); final_line_idx++)
    {
        // 直线在全景图上是曲线，所以需要用一段段的直线近似，这里设置的直线长度低一些，可以更好的近似曲线
        const vector<cv::Point2f> segments = eq.BreakToSegments(lines[final_line_idx], 30);
        // 得到了一段段的直线后，就要把每一段的直线所占据的空间在final_line_occupy_mat上表示出来
        for(int i = 0; i < segments.size() - 1; i++)
        {
            if(abs(segments[i].x - segments[i+1].x) > 0.8 * cols)       
                continue;
            // 找到一个bounding box恰好能包含当前直线，而且要有一定的余量，余量就是 dis_threshold 的大小
            // 而且这个bounding box不能超出图像的尺寸
            cv::Point2f bounding_box_left_top(round(max(min(segments[i].x, segments[i+1].x) - dis_threshold, 0.f)), 
                                              round(max(min(segments[i].y, segments[i+1].y) - dis_threshold, 0.f)));
            cv::Point2f bounding_box_right_bottom(round(min(max(segments[i].x, segments[i+1].x) + dis_threshold, cols-1.f)), 
                                            round(min(max(segments[i].y, segments[i+1].y) + dis_threshold, rows-1.f)));
            // 用cv::line 把这条直线画出来，用255表示，其他区域就用0表示。所以在后面占据空间里就可以用画出的直线来表明
            // 当前直线经过了哪些像素。这里有个效率问题，因为实际上直线是很短的（30个像素左右长度），没必要在整张图片上
            // 把它画出来，所以就用了一个bounding box，只在这个小区域内画
            cv::Mat occupied_matrix = cv::Mat::zeros(static_cast<int>(bounding_box_right_bottom.y - bounding_box_left_top.y), 
                                                    static_cast<int>(bounding_box_right_bottom.x - bounding_box_left_top.x),
                                                    CV_8U);
            cv::line(occupied_matrix, segments[i] - bounding_box_left_top, segments[i+1] - bounding_box_left_top, 255, 2 * dis_threshold);
            for(int row = 0; row < occupied_matrix.rows; row++)
            {
                for(int col = 0; col < occupied_matrix.cols; col++)
                {
                    if(occupied_matrix.at<uchar>(row, col) > 0)
                        final_line_occupy_mat[row + int(bounding_box_left_top.y)][col + int(bounding_box_left_top.x)].insert(final_line_idx);
                }
            }
            
        } 
    }
    
    // 计算初始的直线和最终直线对应的平面参数
    vector<cv::Vec4f> final_planes, init_planes;
    vector<cv::Vec4f> init_lines;
    for(const cv::line_descriptor::KeyLine& key_line : init_keylines)
    {
        init_lines.push_back(cv::Vec4f(key_line.startPointX, key_line.startPointY, key_line.endPointX, key_line.endPointY));
    }
    LinesToPlane(init_lines, init_planes, *(new vector<cv::Point3f>()));
    LinesToPlane(lines, final_planes, *(new vector<cv::Point3f>()));


    for(uint16_t init_line_idx = 0; init_line_idx < init_keylines.size(); init_line_idx++ )
    {
        const cv::Vec4f& l = init_lines[init_line_idx];
        // 记录当前直线中对应于各个最终的直线的像素数量，key = final line id ， value=对应于该最终直线的像素数
        map<size_t, size_t> final_line_count;
        int point_all = 0;
        // 找到一个bounding box恰好能包含当前直线，而且要有一定的余量，余量就是 dis_threshold 的大小
        // 而且这个bounding box不能超出图像的尺寸
        cv::Point2f bounding_box_left_top(round(max(min(l[0], l[2]) - dis_threshold, 0.f)), 
                                            round(max(min(l[1], l[3]) - dis_threshold, 0.f)));
        cv::Point2f bounding_box_right_bottom(round(min(max(l[0], l[2]) + dis_threshold, cols-1.f)), 
                                        round(min(max(l[1], l[3]) + dis_threshold, rows-1.f)));
        // 用cv::line 把这条直线画出来，用255表示，其他区域就用0表示。所以在后面占据空间里就可以用画出的直线来表明
        // 当前直线经过了哪些像素。这里有个效率问题，因为实际上直线是很短的（30个像素左右长度），没必要在整张图片上
        // 把它画出来，所以就用了一个bounding box，只在这个小区域内画
        cv::Mat occupied_matrix = cv::Mat::zeros(static_cast<int>(bounding_box_right_bottom.y - bounding_box_left_top.y), 
                                                static_cast<int>(bounding_box_right_bottom.x - bounding_box_left_top.x),
                                                CV_8U);
        cv::line(occupied_matrix, cv::Point2f(l[0],l[1]) - bounding_box_left_top, cv::Point2f(l[2],l[3]) - bounding_box_left_top, 255);
        for(int row = 0; row < occupied_matrix.rows; row++)
        {
            for(int col = 0; col < occupied_matrix.cols; col++)
            {
                if(occupied_matrix.at<uchar>(row, col) == 0)
                    continue;
                point_all ++;
                for(const size_t& final_line_idx : final_line_occupy_mat[row + int(bounding_box_left_top.y)][col + int(bounding_box_left_top.x)])
                {
                    final_line_count[final_line_idx]++;
                }
            }
        }

        for(map<size_t, size_t>::const_iterator it = final_line_count.begin(); it != final_line_count.end(); it++)
        {
            if(it->second < 0.7 * point_all)
                continue;
            float angle = PlaneAngle(final_planes[it->first].val, init_planes[init_line_idx].val) * 180.0 / M_PI;
            if(angle > 5)
                continue;
            init_to_final[init_line_idx].push_back(it->first);
            final_to_init[it->first].push_back(init_line_idx);
        }
    }
    return true;
    // 以下为可视化，分别显示最终的直线对应于哪些初始直线
    vector<cv::Scalar> colors = {cv::Scalar(0,0,255), cv::Scalar(52,134,255),   // 红 橙
                                        cv::Scalar(20,230,255), cv::Scalar(0, 255,0),   // 黄 绿
                                        cv::Scalar(255,255,51), cv::Scalar(255, 0,0),   // 蓝 蓝
                                        cv::Scalar(255,0,255)};
    for(int i = 0; i < lines.size(); i++)
    {
        cv::Mat img_line;
        cv::cvtColor(img_gray, img_line, cv::COLOR_GRAY2BGR);
        DrawLine(img_line, lines[i], cv::Scalar(0,0,255), 5, true);
        cv::imwrite("final_line-" + num2str(i) + ".jpg", img_line);
        vector<cv::Vec4f> curr_init_lines;
        for(const size_t& idx : final_to_init[i])
            curr_init_lines.push_back(init_lines[idx]);
        img_line = DrawLinesOnImage(img_gray, curr_init_lines, colors, 3, true);
        cv::imwrite("final_line-" + num2str(i) + "_1.jpg", img_line);
    }
    return true;
}

inline bool IsInside(const cv::Point2i& pt, const cv::Mat& img) 
{
    return pt.x >= 0 && pt.y >= 0 && pt.x + 1 < img.cols && pt.y + 1 < img.rows;
}


float PanoramaLine::ComputeNCC(const cv::Mat& img1, const cv::Mat& img2, const cv::Point2f& point1, const cv::Point2f& point2, const int half_window)
{

    int nSizeWindow = 2 * half_window + 1;
    int nSizeStep = 2;
    const int nTexels = Square(nSizeWindow) / nSizeStep + (nSizeStep > 1);

    cv::Point2i x(round(point1.x), round(point1.y));
    float sigmaColor = -1.f / (2 * 0.2 * 0.2);
	float sigmaSpatial = -1.f / (2.f * half_window * half_window);
	const cv::Point2i lt0(x.x - half_window, x.y - half_window); // 以x为中心的图像块左上角
	const cv::Point2i rb0(x.x + half_window, x.y + half_window); // 以x为中心的图像块右下角
    if(! IsInside(lt0, img1) || !IsInside(rb0, img1))
        return -1;
	
	vector<float> texels0(nTexels), weight(nTexels), texelsMean(nTexels), texelsTmp(nTexels);
	uchar center = img1.at<uchar>(x);
	int k = 0;
	for (int m = 0; m < nSizeWindow; m += nSizeStep)
	{
		for (int n = 0; n < nSizeWindow; n += nSizeStep)
		{
			cv::Point2i ptn(lt0.x + n, lt0.y + m);
			float wColor = (img1.at<uchar>(ptn) - center) / 255.f;
			wColor = wColor * wColor * sigmaColor;
			cv::Point2f ptd((float)ptn.x - (float)x.x, (float)ptn.y - (float)x.y);
			float wSpatial = (ptd.x * ptd.x + ptd.y * ptd.y) * sigmaSpatial;
			weight[k] = std::exp(wColor + wSpatial);
			texels0[k] = img1.at<uchar>(ptn); // texels0存储当前图像中以x为中心的方形区域灰度值
			k++;
		}
	}
	float sum = accumulate(weight.begin(), weight.end(), 0.f);
    for(int i = 0; i < nTexels; i++)
        weight[i] /= sum;
    sum = 0;
    for(int i = 0; i < nTexels; i++)
        sum += weight[i] * texels0[i];
    for(int i = 0; i < nTexels; i++)
        texels0[i] -= sum;
    float sq0 = 0; 
    for(int i = 0; i < nTexels; i++)
    {
        float tmp = texels0[i] * weight[i];
        sq0 += texels0[i] * tmp;
        texels0[i] = tmp;             // texels0 存储的是加权的灰度
    }
	if (sq0 <= 0.f) // 如sq0过小，不再进行NCC计算
		return -1;
	
    x = cv::Point2i(round(point2.x), round(point2.y));
	const cv::Point2i lt1(x.x - half_window, x.y - half_window); // 以x为中心的图像块左上角
	const cv::Point2i rb1(x.x + half_window, x.y + half_window); // 以x为中心的图像块右下角
    if(! IsInside(lt1, img2) || !IsInside(rb1, img2))
        return -1;
    
	vector<float> texels1(nTexels);
    center = img2.at<uchar>(x);
	k = 0;
	for (int m = 0; m < nSizeWindow; m += nSizeStep)
	{
		for (int n = 0; n < nSizeWindow; n += nSizeStep)
		{
            cv::Point2i ptn(lt1.x + n, lt1.y + m);
			texels1[k] = img2.at<uchar>(ptn); // texels0存储当前图像中以x为中心的方形区域灰度值
			k++;
		}
	}
    for(int i = 0; i < nTexels; i++)
        sum += texels1[i] * weight[i];
    for(int i = 0; i < nTexels; i++)
        texels1[i] -= sum;
    float sq1 = 0, nrm = 0, sq01 = 0;
    for(int i = 0; i < nTexels; i++)
        sq1 += texels1[i] * texels1[i] * weight[i];
    
    nrm = sq0 * sq1;
    for(int i = 0; i < nTexels; i++)
        sq01 += texels0[i] * texels1[i];
    if (nrm <= 0.f) // 如nrm过小，则NCC分母过小，不再进行NCC计算
        return -1;
    float ncc = sq01 / sqrt(nrm); // 计算NCC
    return std::min(std::max(ncc, -1.f), 1.f) ;
}

void PanoramaLine::SetImageGray(const cv::Mat& gray)
{
    if(gray.channels() == 3)
        cv::cvtColor(gray, img_gray, cv::COLOR_BGR2GRAY);
    else 
        img_gray = gray.clone();
}

void PanoramaLine::SetName(const std::string& _name)
{
    name = _name;
}

const cv::Mat PanoramaLine::GetImageGray() const 
{
    if(!img_gray.empty())
        return img_gray;
    else 
        return cv::imread(name, cv::IMREAD_GRAYSCALE);
}

const std::vector<cv::Vec4f>& PanoramaLine::GetLines() const
{
    return lines;
}

const cv::Mat& PanoramaLine::GetLineDescriptor() const 
{
    return descriptor;
}

const std::vector<cv::line_descriptor::KeyLine>& PanoramaLine::GetInitLines() const
{
    return init_keylines;
}

void PanoramaLine::ReleaseDescriptor()
{
    if(!descriptor.empty())
        descriptor.release();
}

const std::vector<std::vector<size_t>>& PanoramaLine::GetInitToFinal() const
{
    return init_to_final;
}
const std::vector<std::vector<size_t>>& PanoramaLine::GetFinalToInit() const
{
    return final_to_init;
}

PanoramaLine::~PanoramaLine()
{
}

