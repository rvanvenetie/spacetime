add_library(tools SHARED integration.hpp)

package_add_test(integration_test integration_test.cpp)
target_link_libraries(integration_test tools)
