
aux_source_directory(. sfm_files)

add_library(panovlm_sfm ${sfm_files})
target_link_libraries(panovlm_sfm
                      panovlm_base 
                      panovlm_util
                      ${CERES_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${PCL_LIBRARIES}
                      ${LEMON_LIBRARY}
                        ${CLP_LIBRARIES}     # clp + solver wrapper
                        ${COINUTILS_LIBRARY} # container tools
                        ${OSI_LIBRARY}       # generic LP
                      )