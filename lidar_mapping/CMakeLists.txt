aux_source_directory(. lidar_mapping_files)
add_library(panovlm_lidar_mapping ${lidar_mapping_files})
target_link_libraries(panovlm_lidar_mapping
                      panovlm_base 
                      panovlm_util
                      panovlm_sensors
                      ${CERES_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ) 