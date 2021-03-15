#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <chrono>
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

  double TimeApply() const { return time_apply_.count(); };
  const auto &TimeApplySplit() const { return time_apply_split_; }
  std::string TimePerApply() const {
    int num_apply = num_apply_;
    if (num_apply == 0) num_apply = -1;

    std::stringstream result;
    result << "(" << time_apply_.count() / num_apply;
    for (const auto &time : time_apply_split_)
      result << "," << time.count() / num_apply;
    result << ")";
    return result.str();
  };
  double TimeConstruct() const { return time_construct_.count(); }

  virtual std::string Information() { assert(false); }

 protected:
  // Timing debug information.
  std::chrono::duration<double> time_construct_{0};
  std::chrono::duration<double> time_apply_{0};
  std::array<std::chrono::duration<double>, 4> time_apply_split_{};
  size_t num_apply_ = 0;
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
    // Debug information.
    auto time_start = std::chrono::steady_clock::now();
    num_apply_++;

    Eigen::VectorXd result = bil_form_->ApplyTranspose(v);

    // Store timing results.
    time_apply_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_start);

    return result;
  }

  DblVecIn *vec_in() const final { return bil_form_->vec_out(); }
  DblVecOut *vec_out() const final { return bil_form_->vec_in(); }

  auto Transpose() { return bil_form_; }

 protected:
  std::shared_ptr<BilForm> bil_form_;

  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_;
  using BilinearFormBase<DblVecIn, DblVecOut>::num_apply_;
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
    // Debug information.
    auto time_start = std::chrono::steady_clock::now();
    num_apply_++;

    Eigen::VectorXd result = a_->Apply(v) + b_->Apply(v);

    // Store timing results.
    time_apply_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_start);
    for (size_t i = 0; i < time_apply_split_.size(); i++)
      time_apply_split_[i] = a_->TimeApplySplit()[i] + b_->TimeApplySplit()[i];

    return result;
  }
  DblVecIn *vec_in() const final { return a_->vec_in(); }
  DblVecOut *vec_out() const final { return a_->vec_out(); }
  auto sigma() { return a_->sigma(); }
  auto theta() { return a_->theta(); }

  std::shared_ptr<BilFormA> A() { return a_; }
  std::shared_ptr<BilFormB> B() { return b_; }

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

  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_;
  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_split_;
  using BilinearFormBase<DblVecIn, DblVecOut>::num_apply_;
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
  }

  Eigen::VectorXd Apply(const Eigen::VectorXd &v) final {
    // Debug information.
    auto time_start = std::chrono::steady_clock::now();
    num_apply_++;

    Eigen::VectorXd result;
    result = b_->Apply(v);
    result = a_inv_->Apply(result);
    result = bt_->Apply(result);

    result += g_->Apply(v);

    // Store timing results.
    time_apply_ += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - time_start);
    for (size_t i = 0; i < time_apply_split_.size(); i++)
      time_apply_split_[i] = b_->TimeApplySplit()[i] +
                             a_inv_->TimeApplySplit()[i] +
                             bt_->TimeApplySplit()[i] + g_->TimeApplySplit()[i];

    return result;
  }

  DblVecIn *vec_in() const final { return b_->vec_in(); }
  DblVecOut *vec_out() const final { return bt_->vec_out(); }

 protected:
  std::shared_ptr<Ainv> a_inv_;
  std::shared_ptr<B> b_;
  std::shared_ptr<BT> bt_;
  std::shared_ptr<G> g_;

  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_;
  using BilinearFormBase<DblVecIn, DblVecOut>::time_apply_split_;
  using BilinearFormBase<DblVecIn, DblVecOut>::num_apply_;
};
}  // namespace spacetime
