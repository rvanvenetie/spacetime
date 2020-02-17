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
FetchContent_Populate(googletest)
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

## Downloads (a subset of) boost. Only works for header only parts.
FetchContent_Declare(
    boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_SUBMODULES libs/array libs/assert libs/bind libs/concept_check libs/config libs/container
    libs/core libs/detail libs/intrusive libs/iterator libs/move libs/mpl libs/predef
    libs/preprocessor libs/range libs/smart_ptr libs/static_assert libs/type_traits
    libs/throw_exception libs/utility
    GIT_TAG boost-1.72.0
)   
FetchContent_Populate(boost)
file(GLOB boost_includes ${boost_SOURCE_DIR}/libs/*/include)
include_directories(${boost_includes})

### Download Eigen
FetchContent_Declare(
  eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.3.7
)
FetchContent_Populate(eigen3)
include_directories(${eigen3_SOURCE_DIR})

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
add_subdirectory(tools)
add_subdirectory(space)
add_subdirectory(time)
add_subdirectory(applications)
add_subdirectory(spacetime)
