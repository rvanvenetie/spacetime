cmake_minimum_required(VERSION 3.15)

project(SpaceTime)

# Set the compile type to debug.
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-O3")
#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Compiler properties.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMake extra options
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set CCache to compile.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

### Download GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

## Ensure that a proper version of boost is installed
set(Boost_USE_STATIC_RUNTIME ON)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost 1.69.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Install eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

## Enable testing.
enable_testing()

## Add test macro.
macro(package_add_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

## Add all the subdirectories.
include_directories(.)
add_subdirectory(datastructures)
add_subdirectory(space)
add_subdirectory(spacetime)
add_subdirectory(time)
add_subdirectory(tools)
