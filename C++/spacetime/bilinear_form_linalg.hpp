#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <memory>
#include <tuple>

// This is a base class for turning a bilinear form into an Eigen matvec.
namespace spacetime {
template <typename DblVecIn, typename DblVecOut = DblVecIn>
class BilinearFormBase;
}

// Define a necessary Eigen trait.
namespace Eigen {
namespace internal {
template <typename DblVecIn, typename DblVecOut>
struct traits<spacetime::BilinearFormBase<DblVecIn, DblVecOut>>
    : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}  // namespace internal
}  // namespace Eigen

namespace spacetime {
template <typename DblVecIn, typename DblVecOut>
class BilinearFormBase
    : public Eigen::EigenBase<BilinearFormBase<DblVecIn, DblVecOut>> {
 public:
  virtual ~BilinearFormBase() {}

  // These are the BilinearForm functions that must be implemented.
  virtual Eigen::VectorXd Apply() = 0;
  virtual DblVecIn *vec_in() const = 0;
  virtual DblVecOut *vec_out() const = 0;

  // These are the functions that must be implemented for Eigen to work.
  virtual Eigen::Index rows() const {
    if constexpr (!std::is_same_v<DblVecOut, void>)
      return vec_out()->container().size();
    assert(false);
  }
  virtual Eigen::Index cols() const {
    if constexpr (!std::is_same_v<DblVecIn, void>)
      return vec_in()->container().size();
    assert(false);
  }
  virtual Eigen::VectorXd MatVec(const Eigen::VectorXd &rhs) {
    if constexpr (!std::is_same_v<DblVecIn, void>) {
      vec_in()->FromVectorContainer(rhs);
      return Apply();
    }
    assert(false);
  }

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
  Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return const_cast<BilinearFormBase *>(this)->MatVec(x);
  }
};

// This class represents the adjoint of a bilinear form.
template <typename BilForm>
class TransposeBilinearForm
    : public BilinearFormBase<typename BilForm::DblVecOut,
                              typename BilForm::DblVecIn> {
 public:
  using DblVecIn = typename BilForm::DblVecOut;
  using DblVecOut = typename BilForm::DblVecIn;

  TransposeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  Eigen::VectorXd Apply() final { return bil_form_->ApplyTranspose(); }
  DblVecIn *vec_in() const final { return bil_form_->vec_out(); }
  DblVecOut *vec_out() const final { return bil_form_->vec_in(); }

  auto Transpose() { return bil_form_; }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This class represents the sum of two bilinear forms.
template <typename BilFormA, typename BilFormB>
class SumBilinearForm : public BilinearFormBase<typename BilFormA::DblVecIn,
                                                typename BilFormA::DblVecOut> {
 public:
  using DblVecIn = typename BilFormA::DblVecIn;
  using DblVecOut = typename BilFormA::DblVecOut;

  SumBilinearForm(std::shared_ptr<BilFormA> a, std::shared_ptr<BilFormB> b)
      : a_(a), b_(b) {
    assert(a->vec_in() == b->vec_in());
    assert(a->vec_out() == b->vec_out());
  }

  Eigen::VectorXd Apply() final {
    // Apply both operators and store the vectorized result.
    auto v = a_->Apply();
    v += b_->Apply();
    a_->vec_out()->FromVectorContainer(v);
    return v;
  }
  DblVecIn *vec_in() const final { return a_->vec_in(); }
  DblVecOut *vec_out() const final { return a_->vec_out(); }

  auto Transpose() {
    auto a_t = a_->Transpose();
    auto b_t = b_->Transpose();
    return std::make_shared<
        SumBilinearForm<typename decltype(a_t)::element_type,
                        typename decltype(b_t)::element_type>>(a_t, b_t);
  }

 protected:
  std::shared_ptr<BilFormA> a_;
  std::shared_ptr<BilFormB> b_;
};

// This class represents a negative bilinear form (-BilForm).
template <typename BilForm>
class NegativeBilinearForm
    : public BilinearFormBase<typename BilForm::DblVecIn,
                              typename BilForm::DblVecOut> {
 public:
  using DblVecIn = typename BilForm::DblVecIn;
  using DblVecOut = typename BilForm::DblVecOut;
  NegativeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  Eigen::VectorXd Apply() final {
    // Calculate and negate the outpt.
    auto v = bil_form_->Apply();
    v = -v;

    bil_form_->vec_out()->FromVectorContainer(v);
    return v;
  }
  DblVecIn *vec_in() const final { return bil_form_->vec_in(); }
  DblVecOut *vec_out() const final { return bil_form_->vec_out(); }

  auto Transpose() {
    auto b_t = bil_form_->Transpose();
    return std::make_shared<
        NegativeBilinearForm<typename decltype(b_t)::element_type>>(bil_form_);
  }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This operator remaps the intput/output vectors of a given bilinear form.
template <typename BilForm, typename DblVecInType = typename BilForm::DblVecIn,
          typename DblVecOutType = typename BilForm::DblVecOut>
class RemapBilinearForm : public BilinearFormBase<DblVecInType, DblVecOutType> {
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

  Eigen::VectorXd Apply() final {
    // Backup the original input/output of the bilinear form.
    auto v_in = bil_form_->vec_in()->ToVectorContainer();
    auto v_out = bil_form_->vec_out()->ToVectorContainer();

    // Overwrite the input of the bil form with the correct input.
    bil_form_->vec_in()->FromVectorContainer(vec_in_->ToVectorContainer());

    // Apply the operator
    auto result = bil_form_->Apply();

    // Restore the backedup values.
    bil_form_->vec_in()->FromVectorContainer(v_in);
    bil_form_->vec_out()->FromVectorContainer(v_out);

    // Set the correct output.
    vec_out_->FromVectorContainer(result);
    return result;
  }

  DblVecIn *vec_in() const final { return vec_in_; }
  DblVecOut *vec_out() const final { return vec_out_; }

 protected:
  std::shared_ptr<BilForm> bil_form_;
  DblVecIn *vec_in_;
  DblVecOut *vec_out_;
};

// This class represents a 2x2 block diagonal bilinear form.
template <typename B00, typename B01, typename B10, typename B11>
class BlockBilinearForm : public BilinearFormBase<void, void> {
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

  // Little helper functions for getting the vectorized output/intput
  Eigen::VectorXd ToVector(const std::array<::Eigen::VectorXd, 2> &vecs) const {
    assert(vecs[0].size() == b00_->vec_in()->container().size());
    assert(vecs[1].size() == b01_->vec_in()->container().size());
    ::Eigen::VectorXd result(vecs[0].size() + vecs[1].size());
    result << vecs[0], vecs[1];
    return result;
  }

  Eigen::VectorXd Apply() final {
    ::Eigen::VectorXd v0, v1;

    // Apply bil forms in the top row.
    v0 = b00_->Apply();
    v0 += b01_->Apply();

    // Apply bil forms in the bottom row.
    v1 = b10_->Apply();
    v1 += b11_->Apply();

    b00_->vec_out()->FromVectorContainer(v0);
    b10_->vec_out()->FromVectorContainer(v1);
    return ToVector({v0, v1});
  }
  void *vec_in() const final { return nullptr; }
  void *vec_out() const final { return nullptr; }

  // Create an apply that works entirely on vectors.
  ::Eigen::VectorXd MatVec(const ::Eigen::VectorXd &rhs) final {
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
    return Apply();
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
class SchurBilinearForm
    : public BilinearFormBase<typename B::DblVecIn, typename BT::DblVecOut> {
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

  Eigen::VectorXd Apply() final {
    b_->Apply();            // from X_in to Y_out
    a_inv_->Apply();        // from Y_out to Y_in
    auto v = bt_->Apply();  // from Y_in to X_out
    v += g_->Apply();       // from X_in to X_out
    bt_->vec_out()->FromVectorContainer(v);
    return v;
  }

  DblVecIn *vec_in() const final { return b_->vec_in(); }
  DblVecOut *vec_out() const final { return bt_->vec_out(); }

 protected:
  std::shared_ptr<Ainv> a_inv_;
  std::shared_ptr<B> b_;
  std::shared_ptr<BT> bt_;
  std::shared_ptr<G> g_;
};
}  // namespace spacetime
