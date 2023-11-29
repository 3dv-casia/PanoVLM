aux_source_directory(. mvs_files)
add_library(panovlm_mvs ${mvs_files})
target_link_libraries(panovlm_mvs
                      panovlm_base 
                      panovlm_util
                      panovlm_sensors
                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      )