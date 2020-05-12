#pragma once
#include "basis.hpp"
#include "triangulation.hpp"

namespace space {

class InitialTriangulation {
 public:
  datastructures::Tree<Vertex> vertex_tree;
  datastructures::Tree<Element2D> elem_tree;
  datastructures::Tree<HierarchicalBasisFn> hierarch_basis_tree;

  // Convenient pointers.
  Vertex *const vertex_meta_root;
  Element2D *const elem_meta_root;
  HierarchicalBasisFn *const hierarch_basis_meta_root;

  InitialTriangulation(const std::vector<std::array<double, 2>> &vertices,
                       const std::vector<std::array<int, 3>> &elements);

  // Static constructor in case you want the initial triangulation to be
  // refined.
  static InitialTriangulation CreateInitialTriangulation(
      const std::vector<std::array<double, 2>> &vertices,
      const std::vector<std::array<int, 3>> &elements,
      size_t initial_refinement = 0);

  static InitialTriangulation UnitSquare(size_t initial_refinement = 0);
  static InitialTriangulation LShape(size_t initial_refinement = 0);
};
}  // namespace space
