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
    SparseIndices<Basis> &self = (*this);

    // Loop over all indices, and keep the unseen ones.
    size_t i = 0;
    for (size_t j = 0; j < self.size(); ++j) {
      if (!self[j]->marked()) {
        self[i] = self[j];
        self[i]->set_marked(true);
        i++;
      }
    }

    // Remove extra space.
    this->resize(i);

    // Unmark.
    for (auto phi : self) phi->set_marked(false);
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
    result.reserve(this->size());
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
    SparseVector<Basis> &self = (*this);

    // Store the data in the time tree, and sum values up.
    size_t i = 0;
    for (size_t j = 0; j < self.size(); ++j) {
      auto &[phi, coeff] = self[j];
      if (phi->has_data()) {
        (*phi->template data<double>()) += coeff;
      } else {
        if (i != j) self[i] = self[j];
        auto &coeff = self[i].second;
        phi->set_data(&coeff);
        i++;
      }
    }

    // Remove extra space.
    this->resize(i);

    // Remove data.
    for (auto [phi, _] : self) phi->reset_data();
  }

  SparseVector<Basis> Restrict(const SparseIndices<Basis> &ind) const {
    SparseVector<Basis> result;
    result.reserve(ind.size());
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
