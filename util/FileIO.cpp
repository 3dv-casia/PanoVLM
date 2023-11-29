/*
 * @Author: Diantao Tu
 * @Date: 2021-11-19 14:03:13
 */

#include "FileIO.h"

using namespace std;


bool ReadPoseT(std::string file_path, bool with_invalid, eigen_vector<Eigen::Matrix3d>& rotation_list, 
            eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list)
{
    ifstream in(file_path);
    if(!in.is_open())
    {
        LOG(ERROR) << "Fail to open " << file_path << endl;
        return false;
    }

    while (!in.eof())
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
        Eigen::Vector3d t = Eigen::Vector3d::Ones() * numeric_limits<double>::infinity();

        string str;
        getline(in, str);
        vector<string> sub_strings = SplitString(str, ' ')  ;
        string curr_name;
        bool pose_valid = true;
        // 如果分割出13个子串，那就说明是含有名字的，就把第一个子串作为名字保存下来
        if(sub_strings.size() == 13)
        {
            curr_name = sub_strings[0];
            sub_strings.erase(sub_strings.begin());
        }
        // 如果有12个子串，就说明恰好是R t
        if(sub_strings.size() == 12)
        {
            
            for(const string& s : sub_strings)
            {
                // 如果当前的位姿数据中包含了inf nan等数据，就说明当前的位姿数据是不可用的
                if(s.find("inf") != string::npos || s.find("nan") != string::npos)
                {
                    pose_valid = false;
                    break;
                }
            }
            if(pose_valid)
            {
                R(0,0) = str2num<double>(sub_strings[0]);
                R(0,1) = str2num<double>(sub_strings[1]);
                R(0,2) = str2num<double>(sub_strings[2]);
                t.x() = str2num<double>(sub_strings[3]);
                R(1,0) = str2num<double>(sub_strings[4]);
                R(1,1) = str2num<double>(sub_strings[5]);
                R(1,2) = str2num<double>(sub_strings[6]);
                t.y() = str2num<double>(sub_strings[7]);
                R(2,0) = str2num<double>(sub_strings[8]);
                R(2,1) = str2num<double>(sub_strings[9]);
                R(2,2) = str2num<double>(sub_strings[10]);
                t.z() = str2num<double>(sub_strings[11]);
            }
        }
        
        if(pose_valid || (!pose_valid && with_invalid))
        {
            rotation_list.push_back(R);
            trans_list.push_back(t);
            name_list.push_back(curr_name);
        }
        if(in.peek() == EOF)
            break;
    }
    in.close();
    return true;
}

bool ReadPoseQt(std::string file_path, eigen_vector<Eigen::Matrix3d>& rotation_list, 
            eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list)
{
    ifstream in(file_path);
    if(!in.is_open())
    {
        LOG(ERROR) << "Fail to open " << file_path << endl;
        return false;
    }
    while (!in.eof())
    {
        Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
        Eigen::Vector3d t = Eigen::Vector3d::Ones() * numeric_limits<double>::infinity();

        string str;
        getline(in, str);
        vector<string> sub_strings = SplitString(str, ' ');
        
        // 如果分割出8个子串，那就说明是含有名字的，就把第一个子串作为名字保存下来
        if(sub_strings.size() == 8)
        {
            name_list.push_back(sub_strings[0]);
            sub_strings.erase(sub_strings.begin());
        }
        // 如果有7个子串，就说明恰好是R t
        if(sub_strings.size() == 7)
        {
            for(const string& s : sub_strings)
            {
                if(s.find("inf") != string::npos || s.find("nan") != string::npos)
                {
                    goto end;
                }
            }
            double qx, qy, qz, qw;
            qx = str2num<double>(sub_strings[0]);
            qy = str2num<double>(sub_strings[1]);
            qz = str2num<double>(sub_strings[2]);
            qw = str2num<double>(sub_strings[3]);
            t.x() = str2num<double>(sub_strings[4]);
            t.y() = str2num<double>(sub_strings[5]);
            t.z() = str2num<double>(sub_strings[6]);
            R = Eigen::Matrix3d(Eigen::Quaterniond(qw, qx, qy, qz));
        }
        end:
        rotation_list.push_back(R);
        trans_list.push_back(t);
        if(in.peek() == EOF)
            break;
    }
    in.close();
    return true;
}

bool ReadGPS(const std::string file_path, eigen_vector<Eigen::Vector3d>& trans_list, std::vector<std::string>& name_list)
{
    ifstream in(file_path);
    if(!in.is_open())
    {
        LOG(ERROR) << "Fail to open " << file_path << endl;
        return false;
    }
    trans_list.clear();
    name_list.clear();
    while (!in.eof())
    {
        string str;
        getline(in, str);
        vector<string> sub_strings = SplitString(str, ' ');
        if(sub_strings.empty())
            continue;
        if(sub_strings.size() == 4)
            name_list.push_back(sub_strings[0]);
        // 最后的三个数字是xyz
        double x = str2num<double>(sub_strings[sub_strings.size() - 3]);
        double y = str2num<double>(sub_strings[sub_strings.size() - 2]);
        double z = str2num<double>(sub_strings[sub_strings.size() - 1]);
        if(isinf(x) || isinf(y) || isinf(z) || isnan(x) || isnan(y) || isnan(z))
            trans_list.push_back(Eigen::Vector3d::Ones() * std::numeric_limits<double>::infinity());
        else 
            trans_list.push_back(Eigen::Vector3d(x, y, z)); 
        
        if(in.peek() == EOF)
            break;
    }
    return true;
}

void ExportPoseT(const std::string file_path, const eigen_vector<Eigen::Matrix3d>& rotation_list,
                const eigen_vector<Eigen::Vector3d>& trans_list,
                const std::vector<string>& name_list)
{
    ofstream out(file_path);
    if(!out.is_open())
    {
        LOG(ERROR) << "Fail to write " << file_path << endl;
        return;
    }
    assert(rotation_list.size() == trans_list.size());
    for(size_t i = 0; i < rotation_list.size(); i++)
    {
        if(i < name_list.size())
            out << name_list[i] << " ";
        const Eigen::Matrix3d& R_wc = rotation_list[i];
        const Eigen::Vector3d& t_wc = trans_list[i];
        out << R_wc(0,0) << " " << R_wc(0,1) << " " << R_wc(0,2) << " " << t_wc(0) << " "
            << R_wc(1,0) << " " << R_wc(1,1) << " " << R_wc(1,2) << " " << t_wc(1) << " "
            << R_wc(2,0) << " " << R_wc(2,1) << " " << R_wc(2,2) << " " << t_wc(2) << endl;
    }
    out.close();
    return;
}

bool ExportMatchPair(const std::string folder, const std::vector<MatchPair>& pairs)
{
    LOG(INFO) << "Save match pair in " << folder;
    // 保存文件之前要先把之前的都删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directory(folder);

    for(int i = 0; i < pairs.size(); i++)
    {
        ofstream out(folder + "/" + num2str(i) + ".bin");
        boost::archive::binary_oarchive out_archive(out);
        out_archive << pairs[i];
        out.close();
    }
    return true;
}

bool ReadMatchPair(const std::string folder, std::vector<MatchPair>& pairs, const int num_threads)
{
    omp_set_num_threads(num_threads);
    LOG(INFO) << "Loading match pair from " << folder;
    if(!boost::filesystem::exists(folder))
    {
        LOG(ERROR) << "Fail to load match pair, " << folder << " doesn't exist";
        return false;
    }
    vector<string> names = IterateFiles(folder, ".bin");
    if(names.empty())
    {
        LOG(ERROR) << "Fail to load match pair, " << folder << " is empty";
        return false;
    }
    pairs.clear();
    #pragma omp parallel for
    for(size_t i = 0; i < names.size(); i++)
    {
        ifstream in(names[i]);
        boost::archive::binary_iarchive in_archive(in);
        MatchPair pair;
        in_archive >> pair;
        in.close();
        #pragma omp critical
        {
            pairs.push_back(pair);
        }
    }
    // 经过上面openmp的并行操作后，image_pair的顺序就被打乱了，重新按图像的索引排序，排列成
    // 0-1  0-2  0-3  0-4 ... 1-2  1-3  1-4 ... 2-3  2-4  ... 3-4 这样的顺序
    sort(pairs.begin(), pairs.end(), [](const MatchPair& mp1,const MatchPair& mp2)
        {
            if(mp1.image_pair.first == mp2.image_pair.first)
                return mp1.image_pair.second < mp2.image_pair.second;
            else 
                return mp1.image_pair.first < mp2.image_pair.first;
        }
        );
    return true;
}

bool ExportFrame(const std::string folder, const std::vector<Frame>& frames)
{
    LOG(INFO) << "Save frame data in " << folder;
    // 在保存之前要先把文件夹删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directory(folder);
    for(const Frame& f : frames)
    {
        // 从 /aaa/bbb/ccc/xxx.jpg 变成 xxx.bin
        string name = FileName(f.name);
        name += ".bin";
        ofstream out(folder + "/" + name);
        boost::archive::binary_oarchive archive_out(out);
        archive_out << f;
        out.close();
    }
    return true;
}

bool ReadFrames(const std::string frame_folder, const std::string image_folder, 
            std::vector<Frame>& frames, const int num_threads, const bool skip_descriptor )
{
    LOG(INFO) << "Load frames from " << frame_folder;
    vector<string> image_names = IterateFiles(image_folder, ".jpg");
    vector<string> frame_names = IterateFiles(frame_folder, ".bin");
    if(image_names.size() != frame_names.size())
    {
        LOG(ERROR) << "num of images != num of frames";
        return false;
    }
    frames.clear();
    bool success = true;
    // 这里目前假设所有图像都是相同的尺寸
    cv::Mat img = cv::imread(image_names[0]);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < image_names.size(); i++)
    {
        Frame frame(img.rows, img.cols, i, image_names[i]);
        // 记录一下初始的id和路径名，和读取后的对比，要求二者一致才行
        int old_id = frame.id;
        string old_name = frame.name;
        ifstream input(frame_names[i]);
        boost::archive::binary_iarchive ia(input);
        ia >> frame;
        // 这里只检查了图像的id是否一致，没有检查图像的路径名，因为可能存在一种情况是在 A电脑上跑了结果，然后
        // 又把结果放到了B电脑上继续跑，那这种情况下路径就不一样了，但是实际上图像是一样的，所以就不检查路径了。
        // 然而id需要一致，如果不一致就说明是图像顺序变了或者是数量变了等等
        // 同样检查了图像的行列数，保证是相同的图片
        if(frame.id != old_id || frame.GetImageRows() != img.rows || frame.GetImageCols() != img.cols)
        {
            success = false;
        }
        frame.name = old_name;
        if(skip_descriptor)
            frame.ReleaseDescriptor();
        #pragma omp critical 
        {
            frames.push_back(frame);
        }
    }
    if(frames[0].GetDescriptor().rows == 0)
    {
        LOG(WARNING) << "no descriptor in frame, may cause error in image matching"; 
    }
    // 对frame按id大小排列，因为经过openmp乱序执行
    sort(frames.begin(), frames.end(), [](Frame& a, Frame& b){return a.id < b.id;});
    if(success)
        LOG(INFO) << "Load " << frames.size() << " frames";
    else 
        LOG(INFO) << "Load frames failed";
    return success;    
}

bool ExportOpenCVMat(const std::string& file_path, const cv::Mat& mat)
{
    if(mat.empty())
        return false;
    ofstream out(file_path);
    boost::archive::binary_oarchive out_archive(out);
    out_archive << mat;
    out.close();
    return true;
}

bool ReadOpenCVMat(const std::string& file_path, cv::Mat& mat)
{
    if(!boost::filesystem::exists(file_path))
        return false;
    ifstream in(file_path);
    boost::archive::binary_iarchive in_archive(in);
    in_archive >> mat;
    in.close();
    return true;
}

bool ExportFrameDepthAll(string folder, const vector<Frame>& frames, bool use_filtered_depth)
{
    // 保存文件之前要先把之前的都删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directories(folder);
    uint32_t count = 0;
    #pragma omp parallel for
    for(int i = 0; i < frames.size(); i++)
    {
        // if(!ExportFrameDepth(folder + "/" + num2str(i) + ".bin", frames[i], use_filtered_depth))
        //     continue;
        if(!ExportOpenCVMat(folder + "/" + num2str(i) + ".bin", (use_filtered_depth ? frames[i].depth_filter : frames[i].depth_map)))
            continue;
        #pragma omp critical
        {
            count ++;
        }
    }
    LOG(INFO) << "successfully export " << count << " depth images to " << folder;
    return true;
}

bool ExportFrameNormalAll(string folder, const vector<Frame>& frames)
{
    // 保存文件之前要先把之前的都删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directories(folder);
    uint32_t count = 0;
    #pragma omp parallel for
    for(int i = 0; i < frames.size(); i++)
    {
        if(!ExportOpenCVMat(folder + "/" + num2str(i) + ".bin", frames[i].normal_map))
            continue;
        #pragma omp critical
        {
            count ++;
        }
    }
    LOG(INFO) << "successfully export " << count << " normal images to " << folder;
    return true;
}

bool ExportFrameConfAll(string folder, const vector<Frame>& frames)
{
    // 保存文件之前要先把之前的都删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directories(folder);
    uint32_t count = 0;
    #pragma omp parallel for
    for(int i = 0; i < frames.size(); i++)
    {
        if(!ExportConfMap(folder + "/" + num2str(i) + ".bin", frames[i].conf_map))
            continue;
        #pragma omp critical
        {
            count ++;
        }
    }
    LOG(INFO) << "successfully export " << count << " confidence images to " << folder;
    return true;
}

bool ReadFrameDepthAll(const string& folder, vector<Frame>& frames, const string& file_type, bool use_filtered_depth)
{
    if(!boost::filesystem::exists(folder))
    {
        LOG(ERROR) << "fail to load MVS depth images, " << folder << " not exists";
        return false;
    }
    vector<string> names = IterateFiles(folder, file_type);
    bool valid = true;
    #pragma omp parallel for shared(valid)
    for(int i = 0; i < names.size(); i++)
    {
        if(!valid)
            continue;
        const string& name = names[i];
        // /aaa/bbb/cccc/dddd.bin -> dddd
        int id = str2num<int>(FileName(name));
        valid = ReadFrameDepth(name, frames[id], use_filtered_depth);
    }
    return valid;
}

bool ReadFrameNormalAll(const string& folder, vector<Frame>& frames, const string& file_type)
{
    if(!boost::filesystem::exists(folder))
    {
        LOG(ERROR) << "fail to load MVS normal images, " << folder << " not exists";
        return false;
    }
    vector<string> names = IterateFiles(folder, file_type);
    bool valid = true;
    #pragma omp parallel for shared(valid)
    for(int i = 0; i < names.size(); i++)
    {
        if(!valid)
            continue;
        const string& name = names[i];
        int id = str2num<int>(FileName(name));
        valid = ReadFrameNormal(name, frames[id]);
    }
    return valid;
}

bool ReadFrameConfAll(const string& folder, vector<Frame>& frames, const string& file_type)
{
    if(!boost::filesystem::exists(folder))
    {
        LOG(ERROR) << "fail to load MVS conf images, " << folder << " not exists";
        return false;
    }
    vector<string> names = IterateFiles(folder, file_type);
    bool valid = true;
    #pragma omp parallel for shared(valid)
    for(int i = 0; i < names.size(); i++)
    {
        if(!valid)
            continue;
        const string& name = names[i];
        int id = str2num<int>(FileName(name));
        valid = ReadFrameConf(name, frames[id]);
    }
    return valid;
}

bool ExportPanoramaLines(const std::string folder, const std::vector<PanoramaLine>& image_lines_all)
{
    LOG(INFO) << "save panorama line in " << folder;
    // 在保存之前要先把文件夹删了
    if(boost::filesystem::exists(folder))
        boost::filesystem::remove_all(folder);
    boost::filesystem::create_directories(folder);
    for(size_t i = 0; i < image_lines_all.size(); i++)
    {
        ofstream f(folder + "/" + num2str(i) + ".bin");
        boost::archive::binary_oarchive archive_out(f);
        archive_out << image_lines_all[i];
        f.close();
    }
    return true;
}

bool ReadPanoramaLines(const std::string line_folder, const std::string image_folder, std::vector<PanoramaLine>& image_lines_all)
{
    if(!boost::filesystem::exists(line_folder))
    {
        LOG(ERROR) << "Fail to load line, " << line_folder << " not exists";
        return false;
    }
    vector<string> image_names = IterateFiles(image_folder, ".jpg");
    vector<string> line_names = IterateFiles(line_folder, ".bin");
    if(image_names.size() != line_names.size())
    {
        LOG(ERROR) << "num of images != num of lines";
        return false;
    }
    image_lines_all.clear();
    for(const string& name : line_names)
    {
        ifstream f(name);
        PanoramaLine line;
        boost::archive::binary_iarchive ia(f);
        ia >> line;
        f.close();
        image_lines_all.push_back(line);
    }
    LOG(INFO) << "Load " << image_lines_all.size() << " panorama lines";
    return true;
}

bool ExportPointTracks(const std::string file, const std::vector<PointTrack>& tracks)
{
    ofstream f(file);
    if(!f.is_open())
    {
        LOG(ERROR) << "Fail to open " << file << " to write data";
        return false;
    }
    boost::archive::binary_oarchive archive_out(f);
    archive_out << tracks;
    f.close();
    return true;
}

bool ReadPointTracks(const std::string file, std::vector<PointTrack>& tracks)
{
    if(!boost::filesystem::exists(file))
    {
        LOG(ERROR) << "Fail to load tracks, " << file << " not exists";
        return false;
    }
    tracks.clear();
    ifstream f(file);
    boost::archive::binary_iarchive ia(f);
    ia >> tracks;
    f.close();
    LOG(INFO) << "Load " << tracks.size() << " tracks";
    return true;
}

