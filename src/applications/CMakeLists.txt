add_library(applications STATIC heat_equation.cpp adaptive_heat_equation.cpp error_estimator.cpp)
target_precompile_headers(applications PUBLIC heat_equation.hpp)
target_link_libraries(applications spacetime)

add_executable(heat_equation_performance heat_equation_performance.cpp)
target_link_libraries(heat_equation_performance applications)

add_executable(adaptive adaptive.cpp)
target_link_libraries(adaptive applications BoostProgramOptions)

add_executable(uniform uniform.cpp)
target_link_libraries(uniform applications BoostProgramOptions)

# Create tests
package_add_test(heat_equation_test heat_equation_test.cpp)
target_link_libraries(heat_equation_test applications)

package_add_test(adaptive_heat_equation_test adaptive_heat_equation_test.cpp)
target_link_libraries(adaptive_heat_equation_test applications)
