#pragma once
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <tuple>

// This class represents the adjoint of a bilinear form.
template <typename BilForm>
class TransposeBilinearForm {
 public:
  TransposeBilinearForm(std::shared_ptr<BilForm> bil_form)
      : bil_form_(bil_form) {}

  Eigen::VectorXd Apply() { return bil_form_->ApplyTranspose(); }
  auto Transpose() { return bil_form_; }

  auto vec_in() const { return bil_form_->vec_out(); }
  auto vec_out() const { return bil_form_->vec_in(); }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This class represents the sum of two bilinear forms.
template <typename BilFormA, typename BilFormB>
class SumBilinearForm {
 public:
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

  auto vec_in() const { return a_->vec_in(); }
  auto vec_out() const { return a_->vec_out(); }

 protected:
  std::shared_ptr<BilFormA> a_;
  std::shared_ptr<BilFormB> b_;
};

// This class represents a negative bilinear form (-BilForm).
template <typename BilForm>
class NegativeBilinearForm {
 public:
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

  auto vec_in() const { return bil_form_->vec_in(); }
  auto vec_out() const { return bil_form_->vec_out(); }

 protected:
  std::shared_ptr<BilForm> bil_form_;
};

// This class represents a 2x2 block diagonal bilinear form.
template <typename B00, typename B01, typename B10, typename B11>
class BlockBilinearForm {
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

  std::array<Eigen::VectorXd, 2> Apply() {
    auto v0 = b00_->Apply() + b01_->Apply();
    auto v1 = b10_->Apply() + b11_->Apply();

    b00_->vec_out()->FromVectorContainer(v0);
    b10_->vec_out()->FromVectorContainer(v1);
    return {v0, v1};
  }

 protected:
  std::shared_ptr<B00> b00_;
  std::shared_ptr<B01> b01_;
  std::shared_ptr<B10> b10_;
  std::shared_ptr<B11> b11_;
};

// Helper functions.
template <typename BilFormA, typename BilFormB>
auto operator+(std::shared_ptr<BilFormA> a, std::shared_ptr<BilFormB> b) {
  return std::make_shared<SumBilinearForm<BilFormA, BilFormB>>(a, b);
}

template <typename B00, typename B01, typename B10, typename B11>
auto CreateBlockBilinearForm(std::shared_ptr<B00> b00, std::shared_ptr<B01> b01,
                             std::shared_ptr<B10> b10,
                             std::shared_ptr<B11> b11) {
  return std::make_shared<BlockBilinearForm<B00, B01, B10, B11>>(b00, b01, b10,
                                                                 b11);
}
