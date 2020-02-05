#include "../space/bilinear_form.hpp"
#include "basis.hpp"
#include "bilinear_form.hpp"
namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BilinearForm(const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in,
                 DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out)
    : vec_in_(vec_in),
      vec_out_(vec_out),
      sigma_(GenerateSigma(vec_in, *vec_out)),
      theta_(GenerateTheta(vec_in, *vec_out)) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                  BasisTimeOut>::Apply() const {
  for (auto psi_in_labda : sigma_.Project_0()->Bfs()) {
    auto fiber_in = vec_in_.Fiber_1(psi_in_labda->node());
    auto fiber_out = sigma_.Fiber_1(psi_in_labda->node());
    space::BilinearForm<OperatorSpace>(fiber_in, &fiber_out).Apply();
  }
}

}  // namespace spacetime
