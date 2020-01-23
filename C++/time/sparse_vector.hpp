#pragma once
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "cassert"

namespace Time {
template <typename Basis>
class SparseIndices : public std::vector<Basis *> {
 public:
  void Compress() {
    SparseIndices<Basis> result;

    // Loop over all indices, mark the unseen ones and add to result.
    for (auto phi : *this)
      if (!phi->marked()) {
        result.emplace_back(phi);
        phi->set_marked(true);
      }

    // Unmark.
    for (auto phi : result) phi->set_marked(false);

    // Now store the result in our own vector.
    (*this) = std::move(result);
  }

  bool IsUnique() const {
    bool result = true;

    // Loop over all indices, mark the unseen ones.
    for (auto phi : *this)
      if (phi->marked()) {
        result = false;
        break;
      } else
        phi->set_marked(true);

    for (auto phi : *this) phi->set_marked(false);
    return result;
  }

  SparseIndices<Basis> operator|=(const SparseIndices<Basis> &rhs) {
    this->insert(this->end(), rhs.begin(), rhs.end());
    Compress();
    return *this;
  }
  friend std::ostream &operator<<(std::ostream &os,
                                  const SparseIndices<Basis> &ind) {
    os << "{";
    for (auto phi : ind) os << *phi << " ";
    os << "}";
    return os;
  }
};

template <typename Basis>
class SparseVector : public std::vector<std::pair<Basis *, double>> {
 public:
  using Super = std::vector<std::pair<Basis *, double>>;

  SparseVector() = default;
  SparseVector(const SparseVector<Basis> &) = default;
  SparseVector(SparseVector<Basis> &&) = default;
  SparseVector(Super &&vec) : Super(std::move(vec)) {}
  SparseVector<Basis> &operator=(SparseVector<Basis> &&) = default;
  SparseVector<Basis> &operator=(const SparseVector<Basis> &) = default;

  SparseIndices<Basis> Indices() const {
    SparseIndices<Basis> result;
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
    SparseVector<Basis> result;
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
      assert(coeff != NAN);
    }

    // Now store the result in our own vector.
    (*this) = std::move(result);
  }

  SparseVector<Basis> Restrict(const SparseIndices<Basis> &ind) const {
    SparseVector<Basis> result;
    for (auto phi : ind) phi->set_marked(true);
    for (auto [phi, coeff] : *this)
      if (phi->marked()) result.emplace_back(phi, coeff);
    for (auto phi : ind) phi->set_marked(false);
    return result;
  }

  SparseVector<Basis> operator+=(const SparseVector<Basis> &rhs) {
    this->insert(this->end(), rhs.begin(), rhs.end());
    Compress();
    return *this;
  }

  SparseVector<Basis> operator*=(double alpha) {
    for (auto &[_, coeff] : *this) coeff *= alpha;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const SparseVector<Basis> &vec) {
    os << "{";
    for (auto [phi, coeff] : vec) os << "(" << *phi << ", " << coeff << ")  ";
    os << "}";
    return os;
  }
};

}  // namespace Time
