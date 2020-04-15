add_library(heat_equation STATIC heat_equation.cpp)
add_library(adaptive_heat_equation STATIC adaptive_heat_equation.cpp)
add_executable(adaptive_heat_equation_test adaptive_heat_equation_test.cpp)
target_link_libraries(adaptive_heat_equation_test heat_equation adaptive_heat_equation space time gtest gmock gtest_main)
#3target_link_libraries(applications space time heat_equation)
#3
#3add_executable(heat_equation_performance heat_equation_performance.cpp)
#3target_link_libraries(heat_equation_performance applications)
#3
#3add_executable(adaptive adaptive.cpp)
#3target_link_libraries(adaptive applications)
#3
#3# Create tests
#3package_add_test(heat_equation_test heat_equation_test.cpp)
#3target_link_libraries(heat_equation_test applications)
#3
#3package_add_test(adaptive_heat_equation_test adaptive_heat_equation_test.cpp)
#3target_link_libraries(adaptive_heat_equation_test applications)
