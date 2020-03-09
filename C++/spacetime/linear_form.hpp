#pragma once

#include "../datastructures/double_tree_view.hpp"
#include "../space/linear_form.hpp"
#include "../time/linear_form.hpp"

namespace spacetime {
template <typename TimeBasis, size_t space_order, class SpaceF>
class LinearForm {
 public:
  LinearForm(Time::LinearForm<TimeBasis> &&time_linform, SpaceF space_f)
      : time_linform_(std::move(time_linform)), space_f_(space_f) {}

  Eigen::VectorXd Apply(
      datastructures::DoubleTreeVector<TimeBasis, space::HierarchicalBasisFn>
          *vec) {
    vec->Reset();
    auto project_0 = vec->Project_0();
    auto project_1 = vec->Project_1();
    time_linform_.Apply(project_0);
    space::ApplyQuadrature<space_order>(space_f_, project_1,
                                        /*dirichlet_boundary*/ true);

    auto project_0_nodes = project_0->Bfs();
    auto project_1_nodes = project_1->Bfs();
    for (auto phi : project_0_nodes)
      phi->node()->template set_data<double>(
          const_cast<double *>(&phi->value()));
    for (auto phi : project_1_nodes)
      phi->node()->template set_data<double>(
          const_cast<double *>(&phi->value()));

    for (auto &dblnode : vec->container()) {
      if (dblnode.is_metaroot()) continue;
      auto [time_node, space_node] = dblnode.nodes();
      dblnode.set_value(*time_node->template data<double>() *
                        *space_node->template data<double>());
    }

    for (auto phi : project_0_nodes) {
      phi->node()->reset_data();
      phi->set_value(0.0);
    }
    for (auto phi : project_1_nodes) {
      phi->node()->reset_data();
      phi->set_value(0.0);
    }
    return vec->ToVectorContainer();
  }

 protected:
  Time::LinearForm<TimeBasis> time_linform_;
  SpaceF space_f_;
};

template <typename TimeBasis, typename LeftLinForm, typename RightLinForm>
class SumLinearForm {
 public:
  SumLinearForm(LeftLinForm &&left_lf, RightLinForm &&right_lf)
      : left_lf_(std::move(left_lf)), right_lf_(std::move(right_lf)) {}
  Eigen::VectorXd Apply(
      datastructures::DoubleTreeVector<TimeBasis, space::HierarchicalBasisFn>
          *vec) {
    auto vec_in = vec->ToVectorContainer();
    auto result = left_lf_.Apply(vec);
    vec->FromVectorContainer(result);
    result += right_lf_.Apply(vec);
    vec->FromVectorContainer(result);
    return result;
  }

 protected:
  LeftLinForm left_lf_;
  RightLinForm right_lf_;
};

template <typename TimeBasis, typename LeftLinForm, typename RightLinForm>
SumLinearForm<TimeBasis, LeftLinForm, RightLinForm> CreateSumLinearForm(
    LeftLinForm &&left_lf, RightLinForm &&right_lf) {
  return SumLinearForm<TimeBasis, LeftLinForm, RightLinForm>(
      std::move(left_lf), std::move(right_lf));
}

template <typename TimeBasis, size_t time_order, size_t space_order,
          class TimeF, class SpaceF>
LinearForm<TimeBasis, space_order, SpaceF> CreateQuadratureLinearForm(
    const TimeF &time_f, const SpaceF &space_f) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return LinearForm<TimeBasis, space_order, SpaceF>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<
              Time::QuadratureFunctional<TimeScalingBasis, time_order, TimeF>>(
              time_f)),
      space_f);
}

template <typename TimeBasis, size_t space_order, class SpaceF>
LinearForm<TimeBasis, space_order, SpaceF> CreateZeroEvalLinearForm(
    const SpaceF &space_f) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return LinearForm<TimeBasis, space_order, SpaceF>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<Time::ZeroEvalFunctional<TimeScalingBasis>>()),
      space_f);
}
}  // namespace spacetime
