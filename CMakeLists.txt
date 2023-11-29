cmake_minimum_required(VERSION 3.4)
project(PanoVLM)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-fopenmp")

IF (CMAKE_BUILD_TYPE MATCHES "Debug")
        MESSAGE(STATUS "CMAKE_BUILD_TYPE = Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
ELSEIF (CMAKE_BUILD_TYPE MATCHES "Release")
        MESSAGE(STATUS "CMAKE_BUILD_TYPE = Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
ELSE()
        set(CMAKE_BUILD_TYPE "Release")
        MESSAGE(STATUS "Set defalult CMAKE_BUILD_TYPE = Release")
ENDIF()

include_directories("/usr/local/include/eigen3/")

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "find opencv version:" ${OpenCV_VERSION})

# 不知道为什么，一定要先找CGAL，然后再找Boost 
# 不然就会报错，报错是关于boost::serialization的
# 让CGAL不要报关于当前是debug模式的警告
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

find_package(CGAL REQUIRED)
include_directories(${CGAL_INCLUDE_DIRS})
add_definitions(${CGAL_DEFINITIONS})
link_directories(${CGAL_LIBRARY_DIRS})
message(STATUS "find CGAL version:" ${CGAL_VERSION})

find_package(Boost REQUIRED COMPONENTS system filesystem program_options serialization)
include_directories(${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

find_package(CUDA)
if(CUDA_FOUND)
  add_definitions(-DUSE_CUDA)
endif()

add_subdirectory(third_party)

# ==============================================================================
# Lemon
# ==============================================================================
find_package(Lemon QUIET)
if (NOT LEMON_FOUND)
  	set(LEMON_INCLUDE_DIRS
		${CMAKE_CURRENT_SOURCE_DIR}/third_party/lemon
		${PROJECT_BINARY_DIR}/third_party/lemon)
	message(STATUS "Use internal LEMON, version: " ${LEMON_VERSION})
endif()
message(STATUS "LEMON: " ${LEMON_VERSION} " (external)")
include_directories(${LEMON_INCLUDE_DIRS})

# ==============================================================================
# CoinUtils
# ==============================================================================
# - internal by default (CoinUtils),
# - external if COINUTILS_INCLUDE_DIR_HINTS and a valid CoinUtils setup is found
# ==============================================================================
if (NOT DEFINED COINUTILS_INCLUDE_DIR_HINTS)
  set(COINUTILS_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/CoinUtils/src/)
  set(OpenMVG_USE_INTERNAL_COINUTILS ON)
  find_package(CoinUtils QUIET)
  set(COINUTILS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/CoinUtils/src/)
  set(COINUTILS_LIBRARY lib_CoinUtils)
  message(STATUS "Use internal CoinUtils, version: " ${COINUTILS_VERSION})
else()
  find_package(CoinUtils QUIET)
endif()
include_directories(${COINUTILS_INCLUDE_DIRS})

## ==============================================================================
## Clp
## ==============================================================================
## - internal by default (Clp),
## - external if CLP_INCLUDE_DIR_HINTS and a valid Clp setup is found
## ==============================================================================
if (NOT DEFINED CLP_INCLUDE_DIR_HINTS)
  set(CLP_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/Clp/src/)
  set(OpenMVG_USE_INTERNAL_CLP ON)
  find_package(Clp QUIET)
  set(CLP_INCLUDE_DIRS
     ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/Clp/src/
     ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/Clp/src/OsiClp/)
  set(CLP_LIBRARIES lib_clp lib_OsiClpSolver)
  message(STATUS "Use internal CLP, version: " ${CLP_VERSION})
else()
  find_package(Clp QUIET)
endif()
include_directories(${CLP_INCLUDE_DIRS})

# ==============================================================================
# Osi
# ==============================================================================
# - internal by default (Osi),
# - external if OSI_INCLUDE_DIR_HINTS and a valid Osi setup is found
# ==============================================================================
if (NOT DEFINED OSI_INCLUDE_DIR_HINTS)
  set(OSI_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/Osi/src/)
  set(OpenMVG_USE_INTERNAL_OSI ON)
  find_package(Osi QUIET)
  set(OSI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/osi_clp/Osi/src/Osi/)
  set(OSI_LIBRARY lib_Osi)
  message(STATUS "Use internal OSI, version: " ${OSI_VERSION})

else()
  find_package(Osi QUIET)
endif()
include_directories(${OSI_INCLUDE_DIRS})

              
add_subdirectory(base)
add_subdirectory(sensors)
add_subdirectory(util) 
add_subdirectory(joint_optimization)    
add_subdirectory(lidar_mapping)
add_subdirectory(mvs)
add_subdirectory(sfm)



# 主函数
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
                        ${PCL_LIBRARIES}  
                        ${Boost_LIBRARIES} 
                        ${OpenCV_LIBRARIES}
                        glog
                        panovlm_base
                        panovlm_joint_optimization
                        panovlm_lidar_mapping
                        panovlm_mvs
                        panovlm_sensors
                        panovlm_sfm
                        panovlm_util
                        )

