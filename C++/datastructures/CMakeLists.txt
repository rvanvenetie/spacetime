add_library(triangulation STATIC triangulation.cpp triangulation.hpp)

# Create tests
package_add_test(triangulation_test triangulation_test.cpp)
target_link_libraries(triangulation_test triangulation)
