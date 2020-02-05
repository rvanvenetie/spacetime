#pragma once
#include <utility>

#include "../datastructures/double_tree_view.hpp"
#include "../space/basis.hpp"

namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
class BilinearForm {
 protected:
  template <typename T0, typename T1>
  using DoubleTreeVector = datastructures::DoubleTreeVector<T0, T1>;
  using BasisSpace = space::HierarchicalBasisFn;

 public:
  BilinearForm(const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in,
               DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out);
  void Apply() const;

 protected:
  const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out_;

  DoubleTreeVector<BasisTimeIn, BasisSpace> sigma_;
  DoubleTreeVector<BasisTimeOut, BasisSpace> theta_;
};

// Helper function.
template <template <typename, typename> class OpTime, typename OpSpace,
          typename BTimeIn, typename BTimeOut>
BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut> CreateBilinearForm(
    const datastructures::DoubleTreeVector<BTimeIn, space::HierarchicalBasisFn>
        &vec_in,
    datastructures::DoubleTreeVector<BTimeOut, space::HierarchicalBasisFn>
        *vec_out) {
  return BilinearForm<OpTime, OpSpace, BTimeIn, BTimeOut>(vec_in, vec_out);
}

}  // namespace spacetime

#include "bilinear_form.ipp"
