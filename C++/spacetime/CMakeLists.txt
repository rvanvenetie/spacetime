add_library(spacetime SHARED basis.cpp)
target_link_libraries(spacetime space time)

package_add_test(spacetime_basis_test basis_test.cpp)
target_link_libraries(spacetime_basis_test spacetime space time)
