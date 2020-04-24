#include "basis.hpp"
#include "bilinear_form.hpp"
namespace spacetime {

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BilinearForm(DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
                 DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
                 bool use_cache, space::OperatorOptions space_opts)
    : BilinearForm(vec_in, vec_out, GenerateSigma(*vec_in, *vec_out),
                   GenerateTheta(*vec_in, *vec_out), use_cache) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BilinearForm(
        DoubleTreeVector<BasisTimeIn, BasisSpace> *vec_in,
        DoubleTreeVector<BasisTimeOut, BasisSpace> *vec_out,
        std::shared_ptr<DoubleTreeVector<BasisTimeIn, BasisSpace>> sigma,
        std::shared_ptr<DoubleTreeVector<BasisTimeOut, BasisSpace>> theta,
        bool use_cache, space::OperatorOptions space_opts)
    : vec_in_(vec_in),
      vec_out_(vec_out),
      sigma_(sigma),
      theta_(theta),
      use_cache_(use_cache),
      space_opts_(std::move(space_opts)) {}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                             BasisTimeOut>::Apply(const Eigen::VectorXd &v_in) {
  sigma_->Reset();
  theta_->Reset();
  Eigen::VectorXd v_lower;

  // Store the input in the double tree.
  vec_in_->FromVectorContainer(v_in);

  // Check whether we have to recalculate the bilinear forms.
  if (!use_cache_ || !is_cached_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
      if (use_cache_) bil_space_low_.emplace_back(std::move(bil_form));
    }

    // Reset the output.
    vec_out_->Reset();

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
    v_lower = vec_out_->ToVectorContainer();

    // Reset the input, if necessary.
    if constexpr (std::is_same_v<BasisTimeIn, BasisTimeOut>)
      if (vec_in_ == vec_out_) vec_in_->FromVectorContainer(v_in);

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

    // Reset the output.
    vec_out_->Reset();

    // Calculate R_Lambda(Id x A2)I_Theta.
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = theta_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
      if (use_cache_) bil_space_upp_.emplace_back(std::move(bil_form));
    }
    is_cached_ = true;
  } else {
    // Apply the lower part using cached bil forms.
    for (auto &bil_form : bil_space_low_) bil_form.Apply();
    vec_out_->Reset();
    for (auto &bil_form : bil_time_low_) bil_form.ApplyLow();

    // Store the lower output.
    v_lower = vec_out_->ToVectorContainer();

    // Reset the input, if necessary.
    if constexpr (std::is_same_v<BasisTimeIn, BasisTimeOut>)
      if (vec_in_ == vec_out_) vec_in_->FromVectorContainer(v_in);

    // Apply the upper part using cached bil forms.
    for (auto &bil_form : bil_time_upp_) bil_form.ApplyUpp();
    vec_out_->Reset();
    for (auto &bil_form : bil_space_upp_) bil_form.Apply();
  }

  // Return vectorized output.
  return v_lower + vec_out_->ToVectorContainer();
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
             BasisTimeOut>::ApplyTranspose(const Eigen::VectorXd &v_in) {
  // ApplyTranspose only works with if we have cached the bil forms.
  assert(use_cache_ && is_cached_);

  // Reset the necessary DoubleTrees.
  sigma_->Reset();
  theta_->Reset();
  Eigen::VectorXd v_lower;

  // Store the input in the double tree.
  vec_out_->FromVectorContainer(v_in);

  // NOTE: We are calculating the transpose, by transposing all the time/space
  // bilinear forms. This means that the order of evaluation changes, and that
  // ApplyLow becomes  ApplyUpp, and vice versa. Since L is the strict lower
  // diagonal, this only works in Sigma is larger than described in followup3.
  // (See also the note in GenerateSigma).

  // Apply the lower part using cached bil forms.
  for (auto &bil_form : bil_space_upp_) bil_form.Transpose().Apply();
  vec_in_->Reset();
  for (auto &bil_form : bil_time_upp_) bil_form.Transpose().ApplyLow();

  // Store the lower output.
  v_lower = vec_in_->ToVectorContainer();

  // Reset the input, if necessary.
  if constexpr (std::is_same_v<BasisTimeIn, BasisTimeOut>)
    if (vec_in_ == vec_out_) vec_out_->FromVectorContainer(v_in);

  // Apply the upper part using cached bil forms.
  for (auto &bil_form : bil_time_low_) bil_form.Transpose().ApplyUpp();
  vec_in_->Reset();
  for (auto &bil_form : bil_space_low_) bil_form.Transpose().Apply();

  // Return vectorized output.
  return v_lower + vec_in_->ToVectorContainer();
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTime>
SymmetricBilinearForm<OperatorTime, OperatorSpace, BasisTime>::
    SymmetricBilinearForm(
        DblVec *vec,
        std::shared_ptr<DoubleTreeVector<BasisTime, BasisSpace>> sigma,
        bool use_cache, space::OperatorOptions space_opts)
    : vec_(vec),
      sigma_(sigma),
      use_cache_(use_cache),
      space_opts_(std::move(space_opts)) {
  // If use cache, cache the bil forms here.
  if (use_cache_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_space_low_.emplace_back(std::move(bil_form));
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_->Project_1()->Bfs()) {
      auto fiber_in = sigma_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_time_low_.emplace_back(std::move(bil_form));
    }
  }
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTime>
Eigen::VectorXd
SymmetricBilinearForm<OperatorTime, OperatorSpace, BasisTime>::Apply(
    const Eigen::VectorXd &v_in) {
  Eigen::VectorXd v_lower;

  // Store the input in the double tree.
  vec_->FromVectorContainer(v_in);

  if (use_cache_) {
    // Apply the lower part using cached bil forms.
    sigma_->Reset();
    for (auto &bil_form : bil_space_low_) bil_form.Apply();
    vec_->Reset();
    for (auto &bil_form : bil_time_low_) bil_form.ApplyLow();

    v_lower = vec_->ToVectorContainer();
    vec_->FromVectorContainer(v_in);

    // Apply the upper part using cached bil forms.
    sigma_->Reset();
    for (auto &bil_form : bil_time_low_) bil_form.Transpose().ApplyUpp();
    vec_->Reset();
    for (auto &bil_form : bil_space_low_) bil_form.Transpose().Apply();
  } else {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    sigma_->Reset();
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    vec_->Reset();
    for (auto psi_out_labda : vec_->Project_1()->Bfs()) {
      auto fiber_in = sigma_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyLow();
    }

    v_lower = vec_->ToVectorContainer();
    vec_->FromVectorContainer(v_in);

    // Calculate R_Sigma(U_1 x Id)I_Lambda.
    sigma_->Reset();
    for (auto psi_in_labda : vec_->Project_1()->Bfs()) {
      auto fiber_out = sigma_->Fiber_0(psi_in_labda->node());
      if (fiber_out->children().empty()) continue;
      auto fiber_in = psi_in_labda->FrozenOtherAxis();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyUpp();
    }
    // Calculate R_Lambda(Id x A2)I_Sigma.
    vec_->Reset();
    for (auto psi_out_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_out = vec_->Fiber_1(psi_out_labda->node());
      auto fiber_in = psi_out_labda->FrozenOtherAxis();
      if (fiber_in->children().empty()) continue;
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
    }
  }

  // Return vectorized output.
  return v_lower + vec_->ToVectorContainer();
}

template <typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd
BlockDiagonalBilinearForm<OperatorSpace, BasisTimeIn, BasisTimeOut>::Apply(
    const Eigen::VectorXd &v_in) {
  // Store the input in the double tree.
  vec_in_->FromVectorContainer(v_in);

  if (!use_cache_ || !is_cached_) {
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      // Set the level of the time wavelet.
      space_opts_.time_level_ = std::get<0>(psi_out_labda->nodes())->level();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
      if (use_cache_) space_bilforms_.emplace_back(std::move(bil_form));
    }
    is_cached_ = true;
  } else {
    // Apply the space bilforms.
    for (auto &bil_form : space_bilforms_) bil_form.Apply();
  }
  return vec_out_->ToVectorContainer();
}
}  // namespace spacetime
