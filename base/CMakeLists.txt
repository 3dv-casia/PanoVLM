aux_source_directory(. base_files)
add_library(panovlm_base ${base_files})
target_link_libraries(panovlm_base 
                      ${CERES_LIBRARIES})