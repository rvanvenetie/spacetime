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
template <typename DblVecInType, typename DblVecOutType>
class BilinearFormBase
    : public Eigen::EigenBase<BilinearFormBase<DblVecInType, DblVecOutType>> {
 public:
  using DblVecIn = DblVecInType;
  using DblVecOut = DblVecOutType;
  virtual ~BilinearFormBase() {}

  // These are the BilinearForm functions that must be implemented.
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd &v) { assert(false); }
  virtual DblVecIn *vec_in() const { assert(false); }
  virtual DblVecOut *vec_out() const { assert(false); }

  // These are the functions that must be implemented for Eigen to work.
  Eigen::Index rows() const { return vec_out()->container().size(); }
  Eigen::Index cols() const { return vec_in()->container().size(); }

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
    return const_cast<BilinearFormBase *>(this)->Apply(x);
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

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    return bil_form_->ApplyTranspose(v);
  }
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

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    return a_->Apply(v) + b_->Apply(v);
  }
  DblVecIn *vec_in() const final { return a_->vec_in(); }
  DblVecOut *vec_out() const final { return a_->vec_out(); }
  auto sigma() { return a_->sigma(); }
  auto theta() { return a_->theta(); }

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

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    return -bil_form_->Apply(v);
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

/**
 * The Schur complement operator for the matrix [A B; B^t G].
 * x \mapsto (B.T A^{-1} B + G) x.
 */
template <typename Ainv, typename B, typename BT, typename G0>
class SchurBilinearForm
    : public BilinearFormBase<typename B::DblVecIn, typename BT::DblVecOut> {
 public:
  using DblVecIn = typename B::DblVecIn;
  using DblVecOut = typename BT::DblVecOut;
  SchurBilinearForm(std::shared_ptr<Ainv> a_inv, std::shared_ptr<B> b,
                    std::shared_ptr<BT> bt, std::shared_ptr<G0> g)
      : a_inv_(a_inv), b_(b), bt_(bt), g_(g) {
    assert(b_->vec_in() == g_->vec_in());
    assert(a_inv_->vec_in() == b_->vec_out());
    assert(bt_->vec_in() == a_inv_->vec_out());
    assert(bt_->vec_out() == g_->vec_out());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    Eigen::VectorXd result;
    result = b_->Apply(v);
    result = a_inv_->Apply(result);
    result = bt_->Apply(result);

    result += g_->Apply(v);
    return result;
  }

  DblVecIn *vec_in() const final { return b_->vec_in(); }
  DblVecOut *vec_out() const final { return bt_->vec_out(); }

 protected:
  std::shared_ptr<Ainv> a_inv_;
  std::shared_ptr<B> b_;
  std::shared_ptr<BT> bt_;
  std::shared_ptr<G0> g_;
};

/**
 * The Schur complement operator for the matrix [A C; C^t -(A_s + gT' gT)].
 * x \mapsto (C.T A_s^{-1} C + A_s + gT.T gT) x.
 */
template <typename AinvY, typename C, typename CT, typename AX, typename GT>
class NewSchurBilinearForm
    : public BilinearFormBase<typename C::DblVecIn, typename CT::DblVecOut> {
 public:
  using DblVecIn = typename C::DblVecIn;
  using DblVecOut = typename CT::DblVecOut;
  NewSchurBilinearForm(std::shared_ptr<AinvY> a_inv, std::shared_ptr<C> c,
                       std::shared_ptr<CT> ct, std::shared_ptr<AX> aX,
                       std::shared_ptr<GT> gT)
      : a_inv_(a_inv), c_(c), ct_(ct), aX_(aX), gT_(gT) {
    assert(c_->vec_in() == gT_->vec_in());
    assert(a_inv_->vec_in() == c_->vec_out());
    assert(ct_->vec_in() == a_inv_->vec_out());
    assert(ct_->vec_out() == gT_->vec_out());
    assert(aX_->vec_in() == gT_->vec_in());
    assert(aX_->vec_out() == gT_->vec_out());
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    Eigen::VectorXd result;
    std::cout << c_->rows() << " " << c_->cols() << std::endl;
    result = c_->Apply(v);
    std::cout << a_inv_->rows() << std::endl;
    result = a_inv_->Apply(result);
    result = ct_->Apply(result);

    result += aX_->Apply(v);
    result += gT_->Apply(v);
    return result;
  }

  DblVecIn *vec_in() const final { return c_->vec_in(); }
  DblVecOut *vec_out() const final { return ct_->vec_out(); }

 protected:
  std::shared_ptr<AinvY> a_inv_;
  std::shared_ptr<C> c_;
  std::shared_ptr<CT> ct_;
  std::shared_ptr<AX> aX_;
  std::shared_ptr<GT> gT_;
};
}  // namespace spacetime
