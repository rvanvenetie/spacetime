#include "../space/bilinear_form.hpp"
#include "../time/bilinear_form.hpp"
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
      vec_out_low_(vec_out->template DeepCopy<
                   DoubleTreeVector<BasisTimeOut, BasisSpace>>()),
      sigma_(GenerateSigma(vec_in, *vec_out)),
      theta_(GenerateTheta(vec_in, *vec_out)) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                  BasisTimeOut>::Apply() {
  // Reset the necessary DoubleTrees.
  vec_out_->Reset();
  vec_out_low_.Reset();
  sigma_.Reset();
  theta_.Reset();

  // Calculate R_sigma(Id x A_1)I_Lambda
  for (auto psi_in_labda : sigma_.Project_0()->Bfs()) {
    auto fiber_in = vec_in_.Fiber_1(psi_in_labda->node());
    auto fiber_out = sigma_.Fiber_1(psi_in_labda->node());
    space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out).Apply();
  }

  // Calculate R_Lambda(L_0 x Id)I_Sigma
  for (auto psi_out_labda : vec_out_low_.Project_1()->Bfs()) {
    auto fiber_in = sigma_.Fiber_0(psi_out_labda->node());
    auto fiber_out = vec_out_low_.Fiber_0(psi_out_labda->node());
    Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out).ApplyLow();
  }

  // Calculate R_Theta(U_1 x Id)I_Lambda
  for (auto psi_in_labda : theta_.Project_1()->Bfs()) {
    auto fiber_in = vec_in_.Fiber_0(psi_in_labda->node());
    auto fiber_out = theta_.Fiber_0(psi_in_labda->node());
    Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out).ApplyUpp();
  }

  // Calculate R_Lambda(Id x A2)I_Theta
  for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
    auto fiber_in = theta_.Fiber_1(psi_out_labda->node());
    auto fiber_out = vec_out_->Fiber_1(psi_out_labda->node());
    space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out).Apply();
  }

  // Add the lower part to the output.
  *vec_out_ += vec_out_low_;
}

}  // namespace spacetime
