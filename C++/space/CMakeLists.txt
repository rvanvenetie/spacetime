add_library(triangulation STATIC triangulation.cpp triangulation.hpp)
target_include_directories(triangulation PUBLIC ${CMAKE_SOURCE_DIR}/datastructures)

# Create tests
package_add_test(triangulation_test triangulation_test.cpp)
target_link_libraries(triangulation_test triangulation)
