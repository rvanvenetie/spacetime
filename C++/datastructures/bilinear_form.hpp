#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <memory>
#include <tuple>
#include <unsupported/Eigen/IterativeSolvers>

template <typename BilForm>
Eigen::MatrixXd ToMatrix(BilForm &bilform) {
  auto nodes_in = bilform.vec_in()->Bfs();
  auto nodes_out = bilform.vec_out()->Bfs();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nodes_out.size(), nodes_in.size());
  for (int i = 0; i < nodes_in.size(); ++i) {
    bilform.vec_in()->Reset();
    nodes_in[i]->set_value(1);
    bilform.Apply();
    for (int j = 0; j < nodes_out.size(); ++j) {
      A(j, i) = nodes_out[j]->value();
    }
  }
  return A;
}

// This class represents the adjoint of a bilinear form.
template <typename BilForm>
class TransposeBilinearForm {
 public:
  using DblVecIn = typename BilForm::DblVecOut;
  using DblVecOut = typename BilForm::DblVecIn;

  TransposeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  Eigen::VectorXd Apply() { return bil_form_->ApplyTranspose(); }
  auto Transpose() { return bil_form_; }

  DblVecIn *vec_in() const { return bil_form_->vec_out(); }
  DblVecOut *vec_out() const { return bil_form_->vec_in(); }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This class represents the sum of two bilinear forms.
template <typename BilFormA, typename BilFormB>
class SumBilinearForm {
 public:
  using DblVecIn = typename BilFormA::DblVecIn;
  using DblVecOut = typename BilFormA::DblVecOut;

  SumBilinearForm(std::shared_ptr<BilFormA> a, std::shared_ptr<BilFormB> b)
      : a_(a), b_(b) {
    assert(a->vec_in() == b->vec_in());
    assert(a->vec_out() == b->vec_out());
  }

  Eigen::VectorXd Apply() {
    // Apply both operators and store the vectorized result.
    auto v = a_->Apply();
    v += b_->Apply();
    a_->vec_out()->FromVectorContainer(v);
    return v;
  }

  auto Transpose() {
    auto a_t = a_->Transpose();
    auto b_t = b_->Transpose();
    return std::make_shared<
        SumBilinearForm<typename decltype(a_t)::element_type,
                        typename decltype(b_t)::element_type>>(a_t, b_t);
  }

  DblVecIn *vec_in() const { return a_->vec_in(); }
  DblVecOut *vec_out() const { return a_->vec_out(); }

 protected:
  std::shared_ptr<BilFormA> a_;
  std::shared_ptr<BilFormB> b_;
};

// This class represents a negative bilinear form (-BilForm).
template <typename BilForm>
class NegativeBilinearForm {
 public:
  using DblVecIn = typename BilForm::DblVecIn;
  using DblVecOut = typename BilForm::DblVecOut;
  NegativeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  Eigen::VectorXd Apply() {
    auto v = -bil_form_->Apply();
    bil_form_->vec_out()->FromVectorContainer(v);
    return v;
  }
  auto Transpose() {
    auto b_t = bil_form_->Transpose();
    return std::make_shared<
        NegativeBilinearForm<typename decltype(b_t)::element_type>>(bil_form_);
  }

  DblVecIn *vec_in() const { return bil_form_->vec_in(); }
  DblVecOut *vec_out() const { return bil_form_->vec_out(); }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This operator remaps the intput/output vectors of a given bilinear form.
template <typename BilForm, typename DblVecIn = typename BilForm::DblVecIn,
          typename DblVecOut = typename BilForm::DblVecOut>
class RemapBilinearForm {
 public:
  RemapBilinearForm(std::shared_ptr<BilForm> bil_form, DblVecIn *vec_in,
                    DblVecOut *vec_out)
      : bil_form_(bil_form), vec_in_(vec_in), vec_out_(vec_out) {
    assert(vec_in != bil_form_->vec_in());
    assert(vec_in->container().size() ==
           bil_form_->vec_in()->container().size());
    assert(vec_out != bil_form_->vec_out());
    assert(vec_out->container().size() ==
           bil_form_->vec_out()->container().size());
  }

  Eigen::VectorXd Apply() {
    // Backup the original input/output of the bilinear form.
    auto v_in = bil_form_->vec_in()->ToVectorContainer();
    auto v_out = bil_form_->vec_out()->ToVectorContainer();

    // Overwrite the input of the bil form with the correct input.
    bil_form_->vec_in()->FromVectorContainer(vec_in_->ToVectorContainer());

    // Apply the operator
    auto result = bil_form_->Apply();
    vec_out_->FromVectorContainer(result);

    // Restore the backedup values.
    bil_form_->vec_in()->FromVectorContainer(v_in);
    bil_form_->vec_out()->FromVectorContainer(v_out);
    return result;
  }

  DblVecIn *vec_in() const { return vec_in_; }
  DblVecOut *vec_out() const { return vec_out_; }

 protected:
  std::shared_ptr<BilForm> bil_form_;
  DblVecIn *vec_in_;
  DblVecOut *vec_out_;
};

// Below the block bilinear form is implemented as an eigen matrix-free matrix.
template <typename B00, typename B01, typename B10, typename B11>
class BlockBilinearForm;

namespace Eigen {
namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <typename... B>
struct traits<BlockBilinearForm<B...>>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

// This class represents a 2x2 block diagonal bilinear form.
template <typename B00, typename B01, typename B10, typename B11>
class BlockBilinearForm
    : public Eigen::EigenBase<BlockBilinearForm<B00, B01, B10, B11>> {
 public:
  // Ordered like B00, B01, B10, B11
  BlockBilinearForm(std::shared_ptr<B00> b00, std::shared_ptr<B01> b01,
                    std::shared_ptr<B10> b10, std::shared_ptr<B11> b11)
      : b00_(b00), b01_(b01), b10_(b10), b11_(b11) {
    assert(b00_->vec_in() == b10_->vec_in());
    assert(b01_->vec_in() == b11_->vec_in());
    assert(b00_->vec_out() == b01_->vec_out());
    assert(b10_->vec_out() == b11_->vec_out());
  }

  std::array<Eigen::VectorXd, 2> Apply() const {
    Eigen::VectorXd v0, v1;

    // Apply bil forms in the top row.
    v0 = b00_->Apply();
    v0 += b01_->Apply();

    // Apply bil forms in the bottom row.
    v1 = b10_->Apply();
    v1 += b11_->Apply();

    b00_->vec_out()->FromVectorContainer(v0);
    b10_->vec_out()->FromVectorContainer(v1);
    return {v0, v1};
  }

  // Little helper functions for getting the vectorized output/intput
  Eigen::VectorXd ToVector(const std::array<Eigen::VectorXd, 2> &vecs) const {
    assert(vecs[0].size() == b00_->vec_in()->container().size());
    assert(vecs[1].size() == b01_->vec_in()->container().size());
    Eigen::VectorXd result(vecs[0].size() + vecs[1].size());
    result << vecs[0], vecs[1];
    return result;
  }

  // Create an apply that works entirely on vectors.
  Eigen::VectorXd MatVec(const Eigen::VectorXd &rhs) const {
    assert(rhs.size() == cols());
    size_t i = 0;

    // Fill the input vectors with the rhs.
    for (auto &node : b00_->vec_in()->container()) {
      if (node.is_metaroot() || node.node_1()->on_domain_boundary())
        assert(rhs(i) == 0);
      node.set_value(rhs(i++));
    }
    for (auto &node : b01_->vec_in()->container()) {
      if (node.is_metaroot() || node.node_1()->on_domain_boundary())
        assert(rhs(i) == 0);
      node.set_value(rhs(i++));
    }

    // Apply and retrieve the seperate vectors.
    return ToVector(Apply());
  }

  // Eigen stuff
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  Eigen::Index rows() const {
    return b00_->vec_out()->container().size() +
           b10_->vec_out()->container().size();
  }
  Eigen::Index cols() const {
    return b00_->vec_in()->container().size() +
           b01_->vec_in()->container().size();
  }
  template <typename Rhs>
  Eigen::Product<BlockBilinearForm<B00, B01, B10, B11>, Rhs,
                 Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<BlockBilinearForm<B00, B01, B10, B11>, Rhs,
                          Eigen::AliasFreeProduct>(*this, x.derived());
  }

 protected:
  std::shared_ptr<B00> b00_;
  std::shared_ptr<B01> b01_;
  std::shared_ptr<B10> b10_;
  std::shared_ptr<B11> b11_;
};

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template <typename Rhs, typename... B>
struct generic_product_impl<BlockBilinearForm<B...>, Rhs, SparseShape,
                            DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          BlockBilinearForm<B...>, Rhs,
          generic_product_impl<BlockBilinearForm<B...>, Rhs>> {
  typedef typename Product<BlockBilinearForm<B...>, Rhs>::Scalar Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const BlockBilinearForm<B...> &lhs,
                            const Rhs &rhs, const Scalar &alpha) {
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);
    dst.noalias() += lhs.MatVec(rhs);
  }
};
}  // namespace internal
}  // namespace Eigen
