#include "basis.hpp"
#include "bilinear_form.hpp"
namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BilinearForm(const DoubleTreeVector<BasisTimeIn, BasisSpace> &vec_in,
                 DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
                 bool use_cache)
    : vec_in_(vec_in),
      vec_out_(vec_out),
      sigma_(GenerateSigma(vec_in, *vec_out)),
      theta_(GenerateTheta(vec_in, *vec_out)),
      use_cache_(use_cache) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                  BasisTimeOut>::Apply() {
  // Reset the necessary DoubleTrees.
  vec_out_->Reset();
  sigma_.Reset();
  theta_.Reset();
  Eigen::VectorXd v;

  // Check whether we have to recalculate the bilinear forms.
  if (!use_cache_ || (bil_space_low_.empty() && bil_time_upp_.empty())) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : sigma_.Project_0()->Bfs()) {
      auto fiber_in = vec_in_.Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form =
          space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
      bil_form.Apply();
      if (use_cache_) bil_space_low_.emplace_back(std::move(bil_form));
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_out_->Project_1()->Bfs()) {
      auto fiber_in = sigma_.Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyLow();
      if (use_cache_) bil_time_low_.emplace_back(std::move(bil_form));
    }

    // Store the lower output.
    v = vec_out_->ToVectorContainer();
    vec_out_->Reset();

    // Calculate R_Theta(U_1 x Id)I_Lambda.
    for (auto psi_in_labda : theta_.Project_1()->Bfs()) {
      auto fiber_in = vec_in_.Fiber_0(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyUpp();
      if (use_cache_) bil_time_upp_.emplace_back(std::move(bil_form));
    }

    // Calculate R_Lambda(Id x A2)I_Theta.
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = theta_.Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
      bil_form.Apply();
      if (use_cache_) bil_space_upp_.emplace_back(std::move(bil_form));
    }
  } else
#pragma omp parallel
  {
    // Apply the lower part using cached bil forms.
#pragma omp for
    for (int i = 0; i < bil_space_low_.size(); ++i) bil_space_low_[i].Apply();
#pragma omp for
    for (int i = 0; i < bil_time_low_.size(); ++i) bil_time_low_[i].ApplyLow();

      // Store the lower output.
#pragma single
    {
      v = vec_out_->ToVectorContainer();
      vec_out_->Reset();
    }

// Apply the upper part using cached bil forms.
#pragma omp for
    for (int i = 0; i < bil_time_upp_.size(); ++i) bil_time_upp_[i].ApplyUpp();
#pragma omp for
    for (int i = 0; i < bil_space_upp_.size(); ++i) bil_space_upp_[i].Apply();
  }

  // Add the upper part to the output.
  v += vec_out_->ToVectorContainer();

  // Set the output.
  vec_out_->FromVectorContainer(v);
}

}  // namespace spacetime
