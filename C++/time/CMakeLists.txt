add_library(time SHARED basis.cpp haar_basis.cpp orthonormal_basis.cpp three_point_basis.cpp)

# Create tests
package_add_test(basis_test basis_test.cpp)
target_link_libraries(basis_test time)

package_add_test(linear_operator_test linear_operator_test.cpp)
target_link_libraries(linear_operator_test time)

package_add_test(sparse_vector_test sparse_vector_test.cpp)
target_link_libraries(sparse_vector_test time)

package_add_test(bilinear_form_test bilinear_form_test.cpp)
target_link_libraries(bilinear_form_test time)
