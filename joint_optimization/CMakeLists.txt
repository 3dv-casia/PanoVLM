aux_source_directory(. joint_optimization_files)
add_library(panovlm_joint_optimization ${joint_optimization_files})
target_link_libraries(panovlm_joint_optimization 
                      panovlm_base 
                      panovlm_util
                      panovlm_sensors
                      ${OpenCV_LIBRARIES}
                      )   