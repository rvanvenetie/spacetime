#pragma once
#include <utility>

#include "operators.hpp"
#include "triangulation_view.hpp"

namespace space {

template <typename Operator, typename I_in, typename I_out = I_in>
class BilinearForm {
 public:
  BilinearForm(I_in* root_vec_in, I_out* root_vec_out,
               bool dirichlet_boundary = true);

  void Apply();
  Eigen::MatrixXd ToMatrix();

 protected:
  I_in* vec_in_;
  I_out* vec_out_;
  bool symmetric_;

  std::unique_ptr<datastructures::TreeVector<HierarchicalBasisFn>> vec_union_;
  std::unique_ptr<TriangulationView> triang_;
  std::unique_ptr<Operator> operator_;
};

// Helper functions .
template <typename Operator, typename I_in, typename I_out>
BilinearForm<Operator, I_in, I_out> CreateBilinearForm(
    I_in* root_vec_in, I_out* root_vec_out, bool dirichlet_boundary = true) {
  return BilinearForm<Operator, I_in, I_out>(root_vec_in, root_vec_out,
                                             dirichlet_boundary);
}

// Helper function.
template <typename Operator>
auto CreateBilinearForm(
    const datastructures::TreeVector<HierarchicalBasisFn>& vec_in,
    const datastructures::TreeVector<HierarchicalBasisFn>& vec_out,
    bool dirichlet_boundary = true) {
  return BilinearForm<Operator,
                      datastructures::NodeVector<HierarchicalBasisFn>>(
      vec_in.root(), vec_out.root(), dirichlet_boundary);
}

}  // namespace space

#include "bilinear_form.ipp"
