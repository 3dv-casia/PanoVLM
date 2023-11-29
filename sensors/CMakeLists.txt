aux_source_directory(. sensors_files)
add_library(panovlm_sensors ${sensors_files})
target_link_libraries(panovlm_sensors
                      panovlm_base 
                      panovlm_util 
                      ${OpenCV_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${Boost_LIBRARIES}
                      )