#pragma once

#include "../datastructures/double_tree_view.hpp"
#include "../space/linear_form.hpp"
#include "../time/linear_form.hpp"
#include "bilinear_form.hpp"
#include "linear_operator.hpp"

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
class TensorLinearForm : public LinearFormBase<TimeBasis> {
 public:
  using DblVec = typename LinearFormBase<TimeBasis>::DblVec;

  TensorLinearForm(Time::LinearForm<TimeBasis> &&time_linform,
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
      : space_linform_(/* space_f */ [](double x, double y) { return 0; },
                       /* apply_quadrature*/ true,
                       /* space_order */ 0) {}

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
class SumTensorLinearForm : public LinearFormBase<TimeBasis> {
 public:
  using DblVec = typename LinearFormBase<TimeBasis>::DblVec;

  SumTensorLinearForm(std::unique_ptr<TensorLinearForm<TimeBasis>> &&a,
                      std::unique_ptr<TensorLinearForm<TimeBasis>> &&b)
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
  std::unique_ptr<TensorLinearForm<TimeBasis>> a_;
  std::unique_ptr<TensorLinearForm<TimeBasis>> b_;
};

class InterpolationLinearForm
    : public LinearFormBase<Time::OrthonormalWaveletFn> {
 public:
  using DblVecX =
      typename datastructures::DoubleTreeVector<Time::ThreePointWaveletFn,
                                                space::HierarchicalBasisFn>;
  using DblVecY =
      typename datastructures::DoubleTreeVector<Time::OrthonormalWaveletFn,
                                                space::HierarchicalBasisFn>;
  using DblVecZ =
      typename datastructures::DoubleTreeVector<Time::HierarchicalWaveletFn,
                                                space::HierarchicalBasisFn>;

  InterpolationLinearForm(std::shared_ptr<DblVecX> X_delta,
                          std::function<double(double, double, double)> g)
      : X_delta_(X_delta),
        g_(g),
        vec_Z_(X_delta->root()
                   ->node_0()
                   ->children()
                   .at(0)
                   ->support()
                   .at(0)
                   ->parent()
                   ->RefinePsiHierarchical(),
               X_delta->root()->node_1()) {}

  Eigen::VectorXd Apply(DblVecY *vec_Y) final {
    // Grow Z_delta and interpolate.
    GenerateZDelta(*X_delta_, &vec_Z_);
    auto g_vec_Z = Interpolate(g_, vec_Z_);

    // Apply the mass operator to fill the inner products between
    // the interpolant of g, and the given vector Y.
    space::OperatorOptions space_opts(
        {.dirichlet_boundary = false, .build_mat = false});
    spacetime::BilinearForm<Time::MassOperator, space::MassOperator,
                            Time::HierarchicalWaveletFn,
                            Time::OrthonormalWaveletFn>
        mass_bil_form(&vec_Z_, vec_Y, /* use_cache */ false, space_opts);
    auto result = mass_bil_form.Apply(g_vec_Z);

    // We must manually set the boundary dofs in vec_Y to zero.
    size_t i = 0;
    for (const auto &node : vec_Y->container()) {
      if (node.node_1()->on_domain_boundary()) result[i] = 0;
      i++;
    }

    return result;
  }

  const space::LinearForm &SpaceLF() const final {
    throw std::logic_error(
        "SpaceLF is not implemented for interpolation linear form.");
  }

 protected:
  std::shared_ptr<DblVecX> X_delta_;
  std::function<double(double, double, double)> g_;
  DblVecZ vec_Z_;
};

template <typename TimeBasis>
std::unique_ptr<TensorLinearForm<TimeBasis>> CreateQuadratureTensorLinearForm(
    std::function<double(double)> time_f,
    std::function<double(double, double)> space_f, size_t time_order,
    size_t space_order) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return std::make_unique<TensorLinearForm<TimeBasis>>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<Time::QuadratureFunctional<TimeScalingBasis>>(
              time_f, time_order)),
      space::LinearForm(space_f, /* apply_quadrature*/ true, space_order));
}

template <typename TimeBasis>
std::unique_ptr<TensorLinearForm<TimeBasis>> CreateZeroEvalLinearForm(
    std::function<double(double, double)> space_f, bool space_apply_quadrature,
    size_t space_order) {
  using TimeScalingBasis = typename Time::FunctionTrait<TimeBasis>::Scaling;
  return std::make_unique<TensorLinearForm<TimeBasis>>(
      Time::LinearForm<TimeBasis>(
          std::make_unique<Time::ZeroEvalFunctional<TimeScalingBasis>>()),
      space::LinearForm(space_f, space_apply_quadrature, space_order));
}

}  // namespace spacetime
