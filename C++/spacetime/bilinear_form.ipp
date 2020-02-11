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
      use_cache_(use_cache) {
  vec_in_.compute_fibers();
  vec_out_->compute_fibers();
  sigma_.compute_fibers();
  theta_.compute_fibers();
}

template <template <typename, typename> class OperatorTime,
          typename OperatorSpace, typename BasisTimeIn, typename BasisTimeOut>
void BilinearForm<OperatorTime, OperatorSpace, BasisTimeIn,
                  BasisTimeOut>::Apply() {
  // Reset the necessary DoubleTrees.
  vec_out_->Reset();
  sigma_.Reset();
  theta_.Reset();
  Eigen::VectorXd v;

  // clang-format off
  // Check whether we have to recalculate the bilinear forms.
  if (!use_cache_ || (bil_space_low_.empty() && bil_time_upp_.empty()))
  {
    // Load some variables in single threading.
    auto vec_out_proj_0 = vec_out_->Project_0()->Bfs();
    auto vec_out_proj_1 = vec_out_->Project_1()->Bfs();
    auto sigma_proj_0 = sigma_.Project_0()->Bfs();
    auto theta_proj_1 = theta_.Project_1()->Bfs();

    // Execute the rest parallel.
    #pragma omp parallel 
    {
        // Calculate R_sigma(Id x A_1)I_Lambda.
        #pragma omp for nowait schedule(dynamic, 1)
        for (int i = 0; i < sigma_proj_0.size(); ++i) {
          auto psi_in_labda = sigma_proj_0[i];
          auto fiber_in = vec_in_.Fiber_1(psi_in_labda->node());
          auto fiber_out = psi_in_labda->FrozenOtherAxis();
          if (fiber_out->children().empty()) continue;
          auto bil_form =
              space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
          bil_form.Apply();
          if (use_cache_)
            #pragma omp critical
            bil_space_low_.emplace_back(std::move(bil_form));
        }

        // Calculate R_Theta(U_1 x Id)I_Lambda.
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < theta_proj_1.size(); ++i) {
            auto psi_in_labda = theta_proj_1[i];
          auto fiber_in = vec_in_.Fiber_0(psi_in_labda->node());
          auto fiber_out = psi_in_labda->FrozenOtherAxis();
          if (fiber_out->children().empty()) continue;
          auto bil_form =
              Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
          bil_form.ApplyUpp();
          if (use_cache_)
            #pragma omp critical
            bil_time_upp_.emplace_back(std::move(bil_form));
        }

        // Calculate R_Lambda(L_0 x Id)I_Sigma.
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < vec_out_proj_1.size(); ++i) {
          auto psi_out_labda = vec_out_proj_1[i];
          auto fiber_in = sigma_.Fiber_0(psi_out_labda->node());
          if (fiber_in->children().empty()) continue;
          auto fiber_out = psi_out_labda->FrozenOtherAxis();
          auto bil_form =
              Time::CreateBilinearForm<OperatorTime>(fiber_in, fiber_out);
          bil_form.ApplyLow();
          if (use_cache_)
            #pragma omp critical
            bil_time_low_.emplace_back(std::move(bil_form));
        }

        #pragma omp single 
        {
            // Store the lower output.
            v = vec_out_->ToVectorContainer();
            vec_out_->Reset();
        }


        // Calculate R_Lambda(Id x A2)I_Theta.
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < vec_out_proj_0.size(); ++i) {
          auto psi_out_labda = vec_out_proj_0[i];
          auto fiber_in = theta_.Fiber_1(psi_out_labda->node());
          if (fiber_in->children().empty()) continue;
          auto fiber_out = psi_out_labda->FrozenOtherAxis();
          auto bil_form =
              space::CreateBilinearForm<OperatorSpace>(fiber_in, fiber_out);
          bil_form.Apply();
          if (use_cache_)
            #pragma omp critical
            bil_space_upp_.emplace_back(std::move(bil_form));
        }
    }
  } else
  #pragma omp parallel
  {
    #pragma omp for nowait schedule(dynamic, 1)
    for (int i = 0; i < bil_space_low_.size(); ++i) bil_space_low_[i].Apply();
    #pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < bil_time_upp_.size(); ++i) bil_time_upp_[i].ApplyUpp();
    #pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < bil_time_low_.size(); ++i) bil_time_low_[i].ApplyLow();

    // Store the lower output.
    #pragma omp single
    {
      v = vec_out_->ToVectorContainer();
      vec_out_->Reset();
    }
    #pragma omp for schedule(dynamic, 1)
    for (int i = 0; i < bil_space_upp_.size(); ++i) bil_space_upp_[i].Apply();
  }
  // clang-format on

  // Add the upper part to the output.
  v += vec_out_->ToVectorContainer();

  // Set the output.
  vec_out_->FromVectorContainer(v);
}

}  // namespace spacetime
