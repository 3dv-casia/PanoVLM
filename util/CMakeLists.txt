aux_source_directory(. util_files)
add_library(panovlm_util ${util_files})
target_link_libraries(panovlm_util 
                      panovlm_base 
                      panovlm_sensors
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES}
                      ${PCL_LIBRARIES}
                      )