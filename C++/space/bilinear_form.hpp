#pragma once
#include <utility>

#include "operators.hpp"
#include "triangulation_view.hpp"

namespace space {

template <typename Operator>
class BilinearForm {
 public:
  BilinearForm(const datastructures::TreeVector<HierarchicalBasisFn> &vec_in,
               datastructures::TreeVector<HierarchicalBasisFn> *vec_out,
               bool dirichlet_boundary = true);

  void Apply();
  Eigen::MatrixXd ToMatrix();

 protected:
  const datastructures::TreeVector<HierarchicalBasisFn> &vec_in_;
  datastructures::TreeVector<HierarchicalBasisFn> *vec_out_;
  bool symmetric_;

  std::unique_ptr<datastructures::TreeVector<HierarchicalBasisFn>> vec_union_;
  std::unique_ptr<TriangulationView> triang_;
  std::unique_ptr<Operator> operator_;
};

}  // namespace space

#include "bilinear_form.ipp"
