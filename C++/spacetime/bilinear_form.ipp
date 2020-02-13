#include "basis.hpp"
#include "bilinear_form.hpp"
namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BilinearForm(
        DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
        DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
        std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma,
        std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta,
        bool use_cache)
    : vec_in_(vec_in),
      vec_out_(vec_out),
      sigma_(sigma),
      theta_(theta),
      use_cache_(use_cache) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::Apply() {
  // Reset the necessary DoubleTrees.
  vec_out_->Reset();
  sigma_->Reset();
  theta_->Reset();
  Eigen::VectorXd v;

  // Check whether we have to recalculate the bilinear forms.
  if (!use_cache_ || !is_cached_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form =
          space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
      bil_form.Apply();
      if (use_cache_) bil_space_low_.emplace_back(std::move(bil_form));
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_out_->Project_1()->Bfs()) {
      auto fiber_in = sigma_->Fiber_0(psi_out_labda->node());
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
    for (auto psi_in_labda : theta_->Project_1()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_0(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyUpp();
      if (use_cache_) bil_time_upp_.emplace_back(std::move(bil_form));
    }

    // Calculate R_Lambda(Id x A2)I_Theta.
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = theta_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
      bil_form.Apply();
      if (use_cache_) bil_space_upp_.emplace_back(std::move(bil_form));
    }
    is_cached_ = true;
  } else {
    // Apply the lower part using cached bil forms.
    for (auto &bil_form : bil_space_low_) bil_form.Apply();
    for (auto &bil_form : bil_time_low_) bil_form.ApplyLow();

    // Store the lower output.
    v = vec_out_->ToVectorContainer();
    vec_out_->Reset();

    // Apply the upper part using cached bil forms.
    for (auto &bil_form : bil_time_upp_) bil_form.ApplyUpp();
    for (auto &bil_form : bil_space_upp_) bil_form.Apply();
  }

  // Add the upper part to the output, and store the result in the tree.
  size_t i = 0;
  for (auto &node : vec_out_->container()) {
    v[i] += node.value();
    node.set_value(v[i++]);
  }

  // Return vectorized output.
  return v;
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                             BasisTimeOut>::ApplyTranspose() {
  // ApplyTranspose only works with if we have cached the bil forms.
  assert(use_cache_ && is_cached_);

  // Reset the necessary DoubleTrees.
  vec_in_->Reset();
  sigma_->Reset();
  theta_->Reset();
  Eigen::VectorXd v;

  // Apply the lower part using cached bil forms.
  for (auto &bil_form : bil_time_low_) bil_form.Transpose().ApplyUpp();
  for (auto &bil_form : bil_space_low_) bil_form.Transpose().Apply();

  // Store the lower output.
  v = vec_in_->ToVectorContainer();
  vec_in_->Reset();

  // Apply the upper part using cached bil forms.
  for (auto &bil_form : bil_space_upp_) bil_form.Transpose().Apply();
  for (auto &bil_form : bil_time_upp_) bil_form.Transpose().ApplyLow();

  // Add the upper part to the output, and store the result in the tree.
  size_t i = 0;
  for (auto &node : vec_in_->container()) {
    v[i] += node.value();
    node.set_value(v[i++]);
  }

  // Return vectorized output.
  return v;
}
}  // namespace spacetime
