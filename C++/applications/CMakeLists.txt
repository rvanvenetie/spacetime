add_library(applications SHARED heat_equation.cpp)
target_link_libraries(applications spacetime)

# Create tests
package_add_test(heat_equation_test heat_equation_test.cpp)
target_link_libraries(heat_equation_test applications)
