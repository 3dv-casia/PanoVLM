#---------------------
#Third party libraries
#---------------------



# Add graph library
add_subdirectory(lemon)
set(LEMON_LIBRARY openMVG_lemon PARENT_SCOPE)
add_subdirectory(osi_clp)


