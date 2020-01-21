add_library(time SHARED basis.cpp haar_basis.cpp three_point_basis.cpp)

# Create tests
package_add_test(basis_test basis_test.cpp)
target_link_libraries(basis_test time)
