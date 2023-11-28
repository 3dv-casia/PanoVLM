<!--
 * @Author: Diantao Tu
 * @Date: 2023-11-28 18:06:03
-->
# Build 
1. Clone this repository
```bash
git clone git@github.com:3dv-casia/PanoVLM.git
```

2. Compile the code
```bash
cd PanoVLM
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

# Run
All of the configuration files are in `PanoVLM/config/`. You can modify the configuration files to run the code.
Here, we use the Room dataset as an example.

## 1. Prepare the dataset
Download the dataset from [here](https://drive.google.com/file/d/1EjQnuHemL7kW2ORFc0AnFhIdQqvi32fo/view?usp=sharing) and unzip it to `path_to_dataset/Room/`.
By default, PanoVLM outputs many intermediate results, such as image pairs, image lines, depth maps, etc. Please make sure that the disk space is sufficient (about 20GB for Room).

## 2. Modify the configuration file 
The configuration file is in `PanoVLM/config/Room.txt`, please modify the following paths according to your own dataset.
```bash
image_path = path_to_dataset/Room/images/  
lidar_path = path_to_dataset/Room/lidar/    
depth_path = path_to_dataset/Room/depth/ 
frame_path = path_to_dataset/Room/frame/
match_pair_path = path_to_dataset/Room/image_pair/
match_pair_joint_path = path_to_dataset/Room/image_pair_joint/
image_line_path = path_to_dataset/Room/image_line/
lidar_path_undistort = path_to_dataset/Room/lidar_undis/
mvs_data_path = path_to_dataset/Room/mvs/
result_path = path_to_dataset/Room/result/
mask_path = path_to_dataset/Room/mask.jpg
```

## 3. Initial camera pose estimation
```bash
cd PanoVLM
./build/PanoVLM init_camera_pose ./config/Room.txt
```
The result is saved in `path_to_dataset/Room/result/sfm/`.

## 4. Initial LiDAR pose estimation
```bash
./build/PanoVLM init_lidar_pose ./config/Room.txt
```
The result is saved in `path_to_dataset/Room/result/odometry`.

## 5. Joint pose estimation
Change the `angle_residual` to `true` in configuration file.
```bash
./build/PanoVLM joint_optimization ./config/Room.txt
```
The result is saved in `path_to_dataset/Room/result/joint`.

## 6. Colorize LiDAR map (optional)
```bash
./build/PanoVLM colorize_lidar_map ./config/Room.txt
```
The result is saved in `path_to_dataset/Room/result/texture`.

## 7. Joint MVS 
```bash
./build/PanoVLM joint_mvs ./config/Room.txt
```
The result is saved in `path_to_dataset/Room/result/mvs`.

# Run on your own dataset
## LiDAR data pre-processing 
The LiDAR data originally collected by the VLP-16 is in the order of col-major, and the rows are not continuous. 
The scan line order is 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15.
In PanoVLM, we utilize the scan order to map the LiDAR data into a 2D image.
If your LiDAR is not in the order, i.e., the KITTI dataset, please implement the corresponding mapping function in `PanoVLM/sensor/Velodyne.cpp`.


## Critical parameters
1. init_translation_DLT. This parameter is used to determine whether to use the DLT method to initialize the gloabl translation in translation averaging. Theoritically, using DLT to give an initial value can improve the accuracy of translation averaging. However, DLT is time comsuming in large-scale datasets. Therefore, we recommend that you set this parameter to `false` when you run the code for large-scale datasets (more than 500 images). 

2. lidar_segmentation. This parameter is used to determine whether to perform LiDAR segmentation, which is used to remove small objects in the LiDAR point cloud (adopted from LeGO-LOAM). However, the segmentation may lead to worse results in some cases. You can set this parameter to `false` to disable the segmentation.

3. keep_lidar_constant. This parameter is used to determine whether to keep the LiDAR depth constant during mvs propagation. Setting this parameter to `true` can improve the completeness of the final result, leading to better visualization results. However, this may lead to worse accuracy in some cases. You can set this parameter to `false` to disable the constant depth constraint.
