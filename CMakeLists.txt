cmake_minimum_required(VERSION 2.8)
project(superStereo)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# Check C++17 or C++11 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX17)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   add_definitions(-DCOMPILEDWITHC17)
   message(STATUS "Using flag -std=c++17.")
elseif(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 3 not found.")
endif()

find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
   message(FATAL_ERROR "Eigen3 not found.")
endif()
# find_package(Pangolin REQUIRED)
# if(NOT Pangolin_FOUND)
#    message(FATAL_ERROR "Pangolin not found.")
# endif()
# message(STATUS ${EIGEN3_INCLUDE_DIRS})

#LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#FIND_PACKAGE(G2O REQUIRED)
#find_package(CSparse REQUIRED)
#find_package(Cholmod REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
# ${Pangolin_INCLUDE_DIRS}
#${G2O_INCLUDE_DIR}
#${CSPARSE_INCLUDE_DIR}
#${CHOLMOD_INCLUDE_DIR}
)

#message(STATUS "G2O include path: " ${G2O_INCLUDE_DIR})

link_directories(
/usr/local/lib
#/mine/g2o/lib
)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

aux_source_directory(src/ PROJ_SRC)

add_executable(matchingTest
matchingTest.cc
${PROJ_SRC}
)

target_link_libraries(matchingTest
# ${Pangolin_LIBRARIES}
${OpenCV_LIBS}
#${G2O_LIBS}
#${CHOLMOD_LIBRARIES}
)
