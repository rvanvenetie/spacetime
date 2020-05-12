#pragma once

#include "../datastructures/double_tree_view.hpp"
#include "../space/linear_form.hpp"
#include "../time/linear_form.hpp"

namespace spacetime {

template <typename TimeBasis>
class LinearFormBase {
 public:
  using DblVec =
      typename datastructures::DoubleTreeVector<TimeBasis,
                                                space::HierarchicalBasisFn>;
  virtual ~LinearFormBase(){};

  // Function that must be implemented.
  virtual Eigen::VectorXd Apply(DblVec *vec) = 0;
  virtual const space::LinearForm &SpaceLF() const = 0;
};

template <typename TimeBasis>
class LinearForm : public LinearFormBase<TimeBasis> {
 public:
  using DblVec = typename LinearFormBase<TimeBasis>::DblVec;

  LinearForm(Time::LinearForm<TimeBasis> &&time_linform,
             space::LinearForm &&space_linform)
      : time_linform_(std::move(time_linform)),
        space_linform_(std::move(space_linform)) {}

  Eigen::VectorXd Apply(DblVec *vec) final {
    vec->Reset();
    auto project_0 = vec->Project_0();
    auto project_1 = vec->Project_1();
    time_linform_.Apply(project_0);
    space_linform_.Apply(project_1);

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

  const Time::LinearForm<TimeBasis> &TimeLF() const { return time_linform_; }
  const space::LinearForm &SpaceLF() const final { return space_linform_; }

 protected:
  Time::LinearForm<TimeBasis> time_linform_;
  space::LinearForm space_linform_;
};

template <typename TimeBasis>
class NoOpLinearForm : public LinearFormBase<TimeBasis> {
 public:
  using DblVec = typename LinearFormBase<TimeBasis>::DblVec;

  NoOpLinearForm()
      : space_linform_(std::make_unique<space::QuadratureFunctional>(
            /* space_f */ [](double x, double y) { return 0; },
            /* space_order */ 0)) {}

  Eigen::VectorXd Apply(DblVec *vec) final {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(vec->container().size());
    vec->FromVectorContainer(result);
    return result;
  }

  const space::LinearForm &SpaceLF() const final { return space_linform_; }

 protected:
  space::LinearForm space_linform_;
};

template <typename TimeBasis>
class SumLinearForm : public LinearFormBase<TimeBasis> {
 public:
  using DblVec = typename LinearFormBase<TimeBasis>::DblVec;

  SumLinearForm(std::unique_ptr<LinearForm<TimeBasis>> &&a,
                std::unique_ptr<LinearForm<TimeBasis>> &&b)
      : a_(std::move(a)), b_(std::move(b)) {}

  Eigen::VectorXd Apply(DblVec *vec) final {
    auto result = a_->Apply(vec);
    result += b_->Apply(vec);
    vec->FromVectorContainer(result);
    return result;
  }
  const space::LinearForm &SpaceLF() const final {
    throw std::logic_error("SpaceLF is not implemented for sum linear forms.");
  }

 protected:
  std::unique_ptr<LinearForm<TimeBasis>> a_;
  std::unique_ptr<LinearForm<TimeBasis>> b_;
};

template <typename TimeBasis>
std::unique_ptr<LinearForm<TimeBasis>> CreateQuadratureLinearForm(
    std::function<double(double)> time_f,
    std::function<double(double, double)> space_f, size_t time_order,
    size_t space_order) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return std::make_unique<LinearForm<TimeBasis>>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<Time::QuadratureFunctional<TimeScalingBasis>>(
              time_f, time_order)),
      space::LinearForm(
          std::make_unique<space::QuadratureFunctional>(space_f, space_order)));
}

template <typename TimeBasis>
std::unique_ptr<LinearForm<TimeBasis>> CreateZeroEvalLinearForm(
    std::function<double(double, double)> space_f, size_t space_order) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return std::make_unique<LinearForm<TimeBasis>>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<Time::ZeroEvalFunctional<TimeScalingBasis>>()),
      space::LinearForm(
          std::make_unique<space::QuadratureFunctional>(space_f, space_order)));
}
}  // namespace spacetime
