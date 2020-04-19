add_library(tools STATIC integration.cpp)
target_link_libraries(tools time space)

package_add_test(integration_test integration_test.cpp)
target_link_libraries(integration_test tools space time)

package_add_test(linalg_test linalg_test.cpp)
