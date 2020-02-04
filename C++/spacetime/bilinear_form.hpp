#pragma once
#include <utility>

#include "../datastructures/double_tree_view.hpp"

namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename DblTreeIn, typename DblTreeOut>
class BilinearForm {
 public:
  BilinearForm(const DblTreeIn &vec_in, DblTreeOut *vec_out);
  void Apply() const;

 protected:
  const DblTreeIn &vec_in_;
  DblTreeOut *vec_out_;

  datastructures::DoubleTreeVector<typename DblTreeIn::T0,
                                   typename DblTreeOut::T1>
      sigma_;
  datastructures::DoubleTreeVector<typename DblTreeOut::T0,
                                   typename DblTreeIn::T1>
      theta_;
};

}  // namespace spacetime

#include "bilinear_form.ipp"
