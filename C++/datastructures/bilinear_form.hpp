#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <memory>
#include <tuple>
#include <unsupported/Eigen/IterativeSolvers>

// This is a base class for turning a bilinear form into an Eigen matvec.
class EigenBilinearForm;

// Define a necessary Eigen trait.
namespace Eigen {
namespace internal {
template <>
struct traits<EigenBilinearForm>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

class EigenBilinearForm : public Eigen::EigenBase<EigenBilinearForm> {
 public:
  // These are the functions that must be implemented.
  virtual Eigen::Index rows() const { assert(false); }
  virtual Eigen::Index cols() const { assert(false); }
  virtual Eigen::VectorXd MatVec(const Eigen::VectorXd &rhs) const {
    assert(false);
  }
  virtual ~EigenBilinearForm() {}

  // Eigen related stuff.
  using Scalar = double;
  using RealScalar = double;
  using StorageIndex = int;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  template <typename Rhs>
  Eigen::Product<EigenBilinearForm, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<EigenBilinearForm, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }
};

// Define a necessary eigen product overload, simply uses matvec.
namespace Eigen {
namespace internal {
template <typename Rhs>
struct generic_product_impl<EigenBilinearForm, Rhs, SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<EigenBilinearForm, Rhs,
                                generic_product_impl<EigenBilinearForm, Rhs>> {
  using Scalar = typename Product<EigenBilinearForm, Rhs>::Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const EigenBilinearForm &lhs,
                            const Rhs &rhs, const Scalar &alpha) {
    assert(alpha == Scalar(1));
    dst.noalias() += lhs.MatVec(rhs);
  }
};
}  // namespace internal
}  // namespace Eigen

namespace datastructures {
// This class represents the adjoint of a bilinear form.
template <typename BilForm>
class TransposeBilinearForm {
 public:
  using DblVecIn = typename BilForm::DblVecOut;
  using DblVecOut = typename BilForm::DblVecIn;

  TransposeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  ::Eigen::VectorXd Apply() { return bil_form_->ApplyTranspose(); }
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

  ::Eigen::VectorXd Apply() {
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

  ::Eigen::VectorXd Apply() {
    // Calculate and negate the outpt.
    auto v = bil_form_->Apply();
    v = -v;

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
template <typename BilForm, typename DblVecInType = typename BilForm::DblVecIn,
          typename DblVecOutType = typename BilForm::DblVecOut>
class RemapBilinearForm {
 public:
  using DblVecIn = DblVecInType;
  using DblVecOut = DblVecOutType;
  RemapBilinearForm(std::shared_ptr<BilForm> bil_form, DblVecIn *vec_in,
                    DblVecOut *vec_out)
      : bil_form_(bil_form), vec_in_(vec_in), vec_out_(vec_out) {
    assert(vec_in != bil_form_->vec_in());
    assert(vec_in->container().size() ==
           bil_form_->vec_in()->container().size());
    assert(vec_out != bil_form_->vec_out());
    assert(vec_out->container().size() ==
           bil_form_->vec_out()->container().size());

    // Be sure that the ordering of the nodes inside the container coincides!
    // NOTE: Expensive check, remove in a later phase.
    for (size_t i = 0; i < vec_in->container().size(); ++i)
      assert(vec_in->container()[i].nodes() ==
             bil_form_->vec_in()->container()[i].nodes());
    for (size_t i = 0; i < vec_out->container().size(); ++i)
      assert(vec_out->container()[i].nodes() ==
             bil_form_->vec_out()->container()[i].nodes());
  }

  ::Eigen::VectorXd Apply() {
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

// This class represents a 2x2 block diagonal bilinear form.
template <typename B00, typename B01, typename B10, typename B11>
class BlockBilinearForm : public EigenBilinearForm {
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

  std::array<::Eigen::VectorXd, 2> Apply() const {
    ::Eigen::VectorXd v0, v1;

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
  ::Eigen::VectorXd ToVector(
      const std::array<::Eigen::VectorXd, 2> &vecs) const {
    assert(vecs[0].size() == b00_->vec_in()->container().size());
    assert(vecs[1].size() == b01_->vec_in()->container().size());
    ::Eigen::VectorXd result(vecs[0].size() + vecs[1].size());
    result << vecs[0], vecs[1];
    return result;
  }

  // Create an apply that works entirely on vectors.
  ::Eigen::VectorXd MatVec(const ::Eigen::VectorXd &rhs) const final {
    assert(rhs.size() == cols());

    // Fill the input vectors with the rhs.
    size_t i = 0;
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
  ::Eigen::Index rows() const final {
    return b00_->vec_out()->container().size() +
           b10_->vec_out()->container().size();
  }
  ::Eigen::Index cols() const final {
    return b00_->vec_in()->container().size() +
           b01_->vec_in()->container().size();
  }

 protected:
  std::shared_ptr<B00> b00_;
  std::shared_ptr<B01> b01_;
  std::shared_ptr<B10> b10_;
  std::shared_ptr<B11> b11_;
};

/**
 * A class that represents the Schur complement operator
 * x \mapsto (B.T A^{-1} B + gamma_0' gamma_0) x.
 */
template <typename Ainv, typename B, typename BT, typename G>
class SchurBilinearForm : public EigenBilinearForm {
 public:
  using DblVecIn = typename B::DblVecIn;
  using DblVecOut = typename BT::DblVecOut;
  SchurBilinearForm(std::shared_ptr<Ainv> a_inv, std::shared_ptr<B> b,
                    std::shared_ptr<BT> bt, std::shared_ptr<G> g)
      : a_inv_(a_inv), b_(b), bt_(bt), g_(g) {
    assert(b_->vec_in() == g_->vec_in());
    assert(a_inv_->vec_in() == b_->vec_out());
    assert(bt_->vec_in() == a_inv_->vec_out());
    assert(bt_->vec_out() == g_->vec_out());
    assert(g_->vec_in() != bt_->vec_out());
    assert(b_->vec_in() != g_->vec_out());
  }

  ::Eigen::VectorXd Apply() const {
    b_->Apply();            // from X_in to Y_out
    a_inv_->Apply();        // from Y_out to Y_in
    auto v = bt_->Apply();  // from Y_in to X_out
    v += g_->Apply();       // from X_in to X_out
    bt_->vec_out()->FromVectorContainer(v);
    return v;
  }

  ::Eigen::VectorXd MatVec(const ::Eigen::VectorXd &rhs) const final {
    g_->vec_in()->FromVectorContainer(rhs);
    return Apply();
  }

  DblVecIn *vec_in() const { return b_->vec_in(); }
  DblVecOut *vec_out() const { return bt_->vec_out(); }

  ::Eigen::Index rows() const final {
    return g_->vec_out()->container().size();
  }
  ::Eigen::Index cols() const final { return g_->vec_in()->container().size(); }

 protected:
  std::shared_ptr<Ainv> a_inv_;
  std::shared_ptr<B> b_;
  std::shared_ptr<BT> bt_;
  std::shared_ptr<G> g_;
};
}  // namespace datastructures
