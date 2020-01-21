#pragma once
#include <cmath>
#include <utility>
#include <vector>

namespace Time {
template <typename basis>
class SparseVector : public std::vector<std::pair<basis *, double>> {
 public:
  using Super = std::vector<std::pair<basis *, double>>;

  SparseVector() = default;
  SparseVector(const SparseVector<basis> &) = default;
  SparseVector(SparseVector<basis> &&) = default;
  SparseVector(Super &&vec) : Super(std::move(vec)) {}
  SparseVector<basis> &operator=(SparseVector<basis> &&) = default;

  std::vector<basis *> Indices() const {
    std::vector<basis *> result;
    for (auto [phi, _] : *this) {
      result.emplace_back(phi);
    }
    return result;
  }

  void StoreInTree() const {
    for (auto &[phi, coeff] : *this) {
      phi->set_data(const_cast<double *>(&coeff));
    }
  }

  void RemoveFromTree() const {
    for (auto [phi, _] : *this) {
      phi->reset_data();
    }
  }

  void Compress() {
    // Store the data in the time tree, and sum values up.
    SparseVector<basis> result;
    for (auto &[phi, coeff] : *this) {
      if (phi->has_data()) {
        (*phi->template data<double>()) += coeff;
      } else {
        phi->set_data(&coeff);
        result.emplace_back(phi, NAN);
      }
    }

    // Now copy the correct values, and reset the data.
    for (auto &[phi, coeff] : result) {
      coeff = *phi->template data<double>();
      phi->reset_data();
    }

    // Now store the result in our own vector.
    (*this) = std::move(result);
  }
};

}  // namespace Time
