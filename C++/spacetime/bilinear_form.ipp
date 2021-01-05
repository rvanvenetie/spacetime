#pragma once
#include <boost/core/demangle.hpp>
#include <iomanip>

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
                   GenerateTheta(*vec_in, *vec_out), use_cache, space_opts) {}

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
      space_opts_(std::move(space_opts)) {
  auto time_start = std::chrono::steady_clock::now();
#ifdef VERBOSE
  std::cerr << std::left;
  std::cerr << std::endl
            << boost::core::demangle(typeid(*this).name()) << std::endl;
  std::cerr << space_opts << std::endl;
  std::cerr << "  vec_in:  #bfs = " << std::setw(10) << vec_in_->Bfs().size()
            << "#container = " << vec_in_->container().size() << std::endl;
  std::cerr << "  vec_out: #bfs = " << std::setw(10) << vec_out_->Bfs().size()
            << "#container = " << vec_out_->container().size() << std::endl;
  std::cerr << "  sigma:   #bfs = " << std::setw(10) << sigma_->Bfs().size()
            << "#container = " << sigma_->container().size() << std::endl;
  std::cerr << "  theta:   #bfs = " << std::setw(10) << theta_->Bfs().size()
            << "#container = " << theta_->container().size() << std::endl;
  std::cerr << std::right;
#endif

  // If use cache, cache the bil forms here.
  if (use_cache_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      bil_space_low_.emplace_back(fiber_in, fiber_out, space_opts_);
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_out_->Project_1()->Bfs()) {
      auto fiber_in = sigma_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      bil_time_low_.emplace_back(fiber_in, fiber_out);
    }

    // Calculate R_Theta(U_1 x Id)I_Lambda.
    for (auto psi_in_labda : theta_->Project_1()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_0(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      bil_time_upp_.emplace_back(fiber_in, fiber_out);
    }

    // Calculate R_Lambda(Id x A2)I_Theta.
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = theta_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      bil_space_upp_.emplace_back(fiber_in, fiber_out, space_opts_);
    }
  }
  time_construct_ = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                             BasisTimeOut>::Apply(const Eigen::VectorXd &v_in) {
  if (v_in.squaredNorm() == 0)
    return Eigen::VectorXd::Zero(vec_out_->container().size());

  // Debug information.
  auto time_start = std::chrono::steady_clock::now();
  num_apply_++;

  Eigen::VectorXd v_lower;

  // Store the input in the double tree.
  vec_in_->FromVectorContainer(v_in);

  // Check whether we have to recalculate the bilinear forms.
  if (!use_cache_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    auto time_compute = std::chrono::steady_clock::now();
    for (auto psi_in_labda : sigma_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
    }
    time_apply_split_[0] += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_compute);

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    time_compute = std::chrono::steady_clock::now();
    for (auto psi_out_labda : vec_out_->Project_1()->Bfs()) {
      auto fiber_in = sigma_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyLow();
    }
    time_apply_split_[1] += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_compute);

    // Store the lower output.
    v_lower = vec_out_->ToVectorContainer();

    // Reset the input, if necessary.
    if (vec_in_ == sigma_.get() ||
        static_cast<void *>(vec_in_) == static_cast<void *>(vec_out_))
      vec_in_->FromVectorContainer(v_in);

    // Calculate R_Theta(U_1 x Id)I_Lambda.
    time_compute = std::chrono::steady_clock::now();
    for (auto psi_in_labda : theta_->Project_1()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_0(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      bil_form.ApplyUpp();
    }
    time_apply_split_[2] += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_compute);

    // Calculate R_Lambda(Id x A2)I_Theta.
    time_compute = std::chrono::steady_clock::now();
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = theta_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      bil_form.Apply();
    }
    time_apply_split_[3] += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_compute);
  } else {
    // Apply the lower part using cached bil forms.
    for (auto &bil_form : bil_space_low_) bil_form.Apply();
    for (auto &bil_form : bil_time_low_) bil_form.ApplyLow();

    // Store the lower output.
    v_lower = vec_out_->ToVectorContainer();

    // Reset the input, if necessary.
    if (vec_in_ == sigma_.get() ||
        static_cast<void *>(vec_in_) == static_cast<void *>(vec_out_))
      vec_in_->FromVectorContainer(v_in);

    // Apply the upper part using cached bil forms.
    for (auto &bil_form : bil_time_upp_) bil_form.ApplyUpp();
    for (auto &bil_form : bil_space_upp_) bil_form.Apply();
  }

  // Return vectorized output.
  Eigen::VectorXd result = v_lower + vec_out_->ToVectorContainer();

  // Store timing results.
  time_apply_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);

  return result;
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd
BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
             BasisTimeOut>::ApplyTranspose(const Eigen::VectorXd &v_in) {
  if (v_in.squaredNorm() == 0)
    return Eigen::VectorXd::Zero(vec_in_->container().size());

  // ApplyTranspose only works with if we have cached the bil forms.
  assert(use_cache_);

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
  for (auto &bil_form : bil_time_upp_) bil_form.Transpose().ApplyLow();

  // Store the lower output.
  v_lower = vec_in_->ToVectorContainer();

  // Reset the input, if necessary.
  if (vec_out_ == theta_.get() ||
      static_cast<void *>(vec_out_) == static_cast<void *>(vec_in_))
    vec_out_->FromVectorContainer(v_in);

  // Apply the upper part using cached bil forms.
  for (auto &bil_form : bil_time_low_) bil_form.Transpose().ApplyUpp();
  for (auto &bil_form : bil_space_low_) bil_form.Transpose().Apply();

  // Return vectorized output.
  return v_lower + vec_in_->ToVectorContainer();
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTime>
SymmetricBilinearForm<OperatorTime, OperatorSpace, BasisTime>::
    SymmetricBilinearForm(DblVec *vec, bool use_cache,
                          space::OperatorOptions space_opts)
    : vec_(vec), use_cache_(use_cache), space_opts_(std::move(space_opts)) {
  auto time_start = std::chrono::steady_clock::now();
  // If use cache, cache the bil forms here.
  if (use_cache_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : vec_->Project_0()->Bfs()) {
      auto fiber_in = vec_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;
      bil_space_low_.emplace_back(fiber_in, fiber_out, space_opts_);
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_->Project_1()->Bfs()) {
      auto fiber_in = vec_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      bil_time_low_.emplace_back(fiber_in, fiber_out);
    }
  }
  time_construct_ = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTime>
Eigen::VectorXd
SymmetricBilinearForm<OperatorTime, OperatorSpace, BasisTime>::Apply(
    const Eigen::VectorXd &v_in) {
  if (v_in.squaredNorm() == 0)
    return Eigen::VectorXd::Zero(vec_->container().size());

  // Debug information.
  auto time_start = std::chrono::steady_clock::now();
  num_apply_++;

  Eigen::VectorXd v_lower;

  // Store the input in the double tree.
  vec_->FromVectorContainer(v_in);

  if (!use_cache_) {
    // Calculate R_sigma(Id x A_1)I_Lambda.
    for (auto psi_in_labda : vec_->Project_0()->Bfs()) {
      auto fiber_in = vec_->Fiber_1(psi_in_labda->node());
      auto fiber_out = psi_in_labda->FrozenOtherAxis();
      if (fiber_out->children().empty()) continue;

      auto time_compute = std::chrono::steady_clock::now();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      time_apply_split_[0] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      time_compute = std::chrono::steady_clock::now();
      bil_form.Apply();
      time_apply_split_[1] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }

    // Calculate R_Lambda(L_0 x Id)I_Sigma.
    for (auto psi_out_labda : vec_->Project_1()->Bfs()) {
      auto fiber_in = vec_->Fiber_0(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();

      auto time_compute = std::chrono::steady_clock::now();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      time_apply_split_[2] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      time_compute = std::chrono::steady_clock::now();
      bil_form.ApplyLow();
      time_apply_split_[3] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }

    v_lower = vec_->ToVectorContainer();
    vec_->FromVectorContainer(v_in);

    // Calculate R_Sigma(U_1 x Id)I_Lambda.
    for (auto psi_in_labda : vec_->Project_1()->Bfs()) {
      auto fiber_out = vec_->Fiber_0(psi_in_labda->node());
      if (fiber_out->children().empty()) continue;
      auto fiber_in = psi_in_labda->FrozenOtherAxis();

      auto time_compute = std::chrono::steady_clock::now();
      auto bil_form =
          Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
      time_apply_split_[4] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      time_compute = std::chrono::steady_clock::now();
      bil_form.ApplyUpp();
      time_apply_split_[5] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }

    // Calculate R_Lambda(Id x A2)I_Sigma.
    for (auto psi_out_labda : vec_->Project_0()->Bfs()) {
      auto fiber_out = vec_->Fiber_1(psi_out_labda->node());
      auto fiber_in = psi_out_labda->FrozenOtherAxis();
      if (fiber_in->children().empty()) continue;

      auto time_compute = std::chrono::steady_clock::now();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      time_apply_split_[6] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      time_compute = std::chrono::steady_clock::now();
      bil_form.Apply();
      time_apply_split_[7] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }
  } else {
    // Apply the lower part using cached bil forms.
    for (auto &bil_form : bil_space_low_) bil_form.Apply();
    for (auto &bil_form : bil_time_low_) bil_form.ApplyLow();

    v_lower = vec_->ToVectorContainer();
    vec_->FromVectorContainer(v_in);

    // Apply the upper part using cached bil forms.
    for (auto &bil_form : bil_time_low_) bil_form.Transpose().ApplyUpp();
    for (auto &bil_form : bil_space_low_) bil_form.Transpose().Apply();
  }

  // Return vectorized output.
  Eigen::VectorXd result = v_lower + vec_->ToVectorContainer();

  // Store timing results.
  time_apply_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);

  return result;
}
template <typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
BlockDiagonalBilinearForm<OperatorSpace, BasisTimeIn, BasisTimeOut>::
    BlockDiagonalBilinearForm(DblVecIn *vec_in, DblVecOut *vec_out,
                              bool use_cache, space::OperatorOptions space_opts)
    : vec_in_(vec_in),
      vec_out_(vec_out),
      use_cache_(use_cache),
      space_opts_(std::move(space_opts)) {
  auto time_start = std::chrono::steady_clock::now();
  assert(vec_in->container().size() == vec_out->container().size());
  // If use cache, cache the bil forms here.
  if (use_cache_) {
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      // Set the level of the time wavelet.
      space_opts_.time_level = std::get<0>(psi_out_labda->nodes())->level();
      space_bilforms_.emplace_back(fiber_in, fiber_out, space_opts_);
    }
  }
  time_construct_ = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);
}

template <typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
Eigen::VectorXd
BlockDiagonalBilinearForm<OperatorSpace, BasisTimeIn, BasisTimeOut>::Apply(
    const Eigen::VectorXd &v_in) {
  if (v_in.squaredNorm() == 0)
    return Eigen::VectorXd::Zero(vec_out_->container().size());

  // Debug information.
  auto time_start = std::chrono::steady_clock::now();
  num_apply_++;

  // Store the input in the double tree.
  vec_in_->FromVectorContainer(v_in);

  if (!use_cache_) {
    for (auto psi_out_labda : vec_out_->Project_0()->Bfs()) {
      auto fiber_in = vec_in_->Fiber_1(psi_out_labda->node());
      if (fiber_in->children().empty()) continue;
      auto fiber_out = psi_out_labda->FrozenOtherAxis();
      // Set the level of the time wavelet.
      space_opts_.time_level = std::get<0>(psi_out_labda->nodes())->level();

      // Create Bilform.
      auto time_compute = std::chrono::steady_clock::now();
      auto bil_form = space::CreateBilinearForm<OperatorSpace>(
          fiber_in, fiber_out, space_opts_);
      time_apply_split_[0] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      // Apply Bilform.
      time_compute = std::chrono::steady_clock::now();
      bil_form.Apply();
      time_apply_split_[1] += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }
  } else {
    // Apply the space bilforms.
    for (auto &bil_form : space_bilforms_) bil_form.Apply();
  }
  // Return vectorized output.
  Eigen::VectorXd result = vec_out_->ToVectorContainer();

  // Store timing results.
  time_apply_ += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - time_start);

  return result;
}
}  // namespace spacetime
