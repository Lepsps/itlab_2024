cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 11)

set(ProjectName "itlab")
project(${ProjectName})

include_directories(include)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    v1.14.0
)
FetchContent_MakeAvailable(googletest)


#add_subdirectory(3rdparty/googletest)
add_subdirectory(src)
add_subdirectory(test)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message( STATUS "")