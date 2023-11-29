/*
 * @Author: Diantao Tu
 * @Date: 2022-06-28 17:33:43
 */
#include "Util.h"

bool SetFramePose(std::vector<Frame>& frames, const std::vector<Velodyne>& lidars, const Eigen::Matrix4d& T_cl, const double time_offset, const double time_gap)
{
	assert(lidars.size() == frames.size());
    const double lidar_duration = 0.1;		// 每帧雷达持续的时间
    Eigen::Matrix4d T_lc = T_cl.inverse();
    for(int i = 0; i < int(lidars.size()); i++)
    {
        if(!lidars[i].IsPoseValid())
            continue;
        // 先给frame设置个初始的外参，后面再计算更精细的结果
        Eigen::Matrix4d frame_pose = lidars[i].GetPose() * T_lc;
        frames[i].SetPose(frame_pose);
		// 找到距离当前frame最近的两帧雷达,要求这两帧雷达一个在frame之前，一个在frame之后，frame的位姿就依靠它们插值得到
		// 只有对于第一帧或最后一帧frame的时候可能会出现两帧雷达都在frame之前或两帧雷达都在frame之后的情况
        double lidar_time = (lidar_duration + time_gap) * i;
        double frame_time = lidar_time + time_offset;
        int lidar_idx_before_frame = max(frame_time / (lidar_duration + time_gap), 0.0);
        int lidar_idx_after_frame = min(lidar_idx_before_frame + 1 , int(lidars.size()) -1);
        lidar_idx_before_frame = lidar_idx_after_frame - 1;
        
        while(lidar_idx_after_frame < lidars.size() - 1 && !lidars[lidar_idx_after_frame].IsPoseValid())
            lidar_idx_after_frame++;
        if(lidar_idx_after_frame >= lidars.size())
            continue;
        while(lidar_idx_before_frame >= 0 && !lidars[lidar_idx_after_frame].IsPoseValid())
            lidar_idx_before_frame--;
        if(lidar_idx_before_frame < 0)
            continue;

        double lidar_time_before_frame = lidar_idx_before_frame * (lidar_duration + time_gap);
        double lidar_time_after_frame = lidar_idx_after_frame * (lidar_duration + time_gap);
        double ratio = (frame_time - lidar_time_before_frame) / (lidar_time_after_frame - lidar_time_before_frame);
        frame_pose = SlerpPose(lidars[lidar_idx_before_frame].GetPose(), lidars[lidar_idx_after_frame].GetPose(), ratio);
        frame_pose = frame_pose * T_lc;
        frames[i].SetPose(frame_pose);
    }
    return true;
}

bool SetLidarPose(const std::vector<Frame>& frames, std::vector<Velodyne>& lidars, const Eigen::Matrix4d& T_cl, const double time_offset, const double time_gap)
{
    assert(lidars.size() == frames.size());
    const double frame_duration = 0.1;
    Eigen::Matrix4d T_lc = T_cl.inverse();
    for(int i = 0; i < int(lidars.size()); i++)
    {
        if(!frames[i].IsPoseValid())
            continue;

        Eigen::Matrix4d lidar_pose = frames[i].GetPose() * T_cl;
        lidars[i].SetPose(lidar_pose);
        double frame_time = (frame_duration + time_gap) * i;
        double lidar_time = frame_time - time_offset;
        int frame_idx_before_lidar = max(lidar_time / (frame_duration + time_gap), 0.0);
        int frame_idx_after_lidar = min(frame_idx_before_lidar + 1, int(frames.size()) - 1);
        frame_idx_before_lidar = frame_idx_after_lidar - 1;
        while(frame_idx_after_lidar < frames.size() - 1 && !frames[frame_idx_after_lidar].IsPoseValid())
            frame_idx_after_lidar++;
        if(frame_idx_after_lidar >= frames.size())
            continue;
        while(frame_idx_before_lidar >= 0 && !frames[frame_idx_before_lidar].IsPoseValid())
            frame_idx_before_lidar--;
        if(frame_idx_before_lidar < 0)
            continue;
        double frame_time_before_lidar = frame_idx_before_lidar * (frame_duration + time_gap);
        double frame_time_after_lidar = frame_idx_after_lidar * (frame_duration + time_gap);
        double ratio = (lidar_time - frame_time_before_lidar) / (frame_time_after_lidar - frame_time_before_lidar);
        lidar_pose = SlerpPose(frames[frame_idx_before_lidar].GetPose(), frames[frame_idx_after_lidar].GetPose(), ratio);
        lidar_pose = lidar_pose * T_cl;
        lidars[i].SetPose(lidar_pose);

    }
    return true;
}

bool LoadFramePose(std::vector<Frame>& frames, const std::string pose_file)
{
    eigen_vector<Eigen::Matrix3d> R_wc_list;
    eigen_vector<Eigen::Vector3d> t_wc_list;
    vector<string> image_names;
    if(!ReadPoseT(pose_file, false, R_wc_list, t_wc_list, image_names))
        return false;
    
    for(size_t i = 0, frame_idx = 0; i < frames.size(); i++)
    {
        if(frames[i].name == image_names[frame_idx])
        {
            frames[i].SetRotation(R_wc_list[frame_idx]);
            frames[i].SetTranslation(t_wc_list[frame_idx]);
            frame_idx++;
        }
    }
    return true;
}

bool LoadLidarPose(std::vector<Velodyne>& lidars, const std::string pose_file, const bool with_invalid)
{
    eigen_vector<Eigen::Matrix3d> R_wl_list;
    eigen_vector<Eigen::Vector3d> t_wl_list;
    vector<string> lidar_names;
    if(!ReadPoseT(pose_file, with_invalid, R_wl_list, t_wl_list, lidar_names))
        return false;
    for(size_t i = 0; i < lidar_names.size(); i++)
    {
        Velodyne lidar(16, i);
        lidar.SetName(lidar_names[i]);
        lidar.SetRotation(R_wl_list[i]);
        lidar.SetTranslation(t_wl_list[i]);
        lidars.push_back(lidar);
    }
    LOG(INFO) << "Load " << lidars.size() << " lidars";
    return true;
}