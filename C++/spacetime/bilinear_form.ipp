#include "bilinear_form.hpp"
namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename DblTreeIn, typename DblTreeOut>
BilinearForm<OperatorTime, OperatorSpace, DblTreeIn, DblTreeOut>::BilinearForm(
    const DblTreeIn &vec_in, DblTreeOut *vec_out)
    : vec_in_(vec_in), vec_out_(vec_out){};

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename DblTreeIn, typename DblTreeOut>
void BilinearForm<OperatorTime, OperatorSpace, DblTreeIn, DblTreeOut>::Apply()
    const {}

};  // namespace spacetime
