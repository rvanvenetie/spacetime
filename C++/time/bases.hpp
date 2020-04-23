#pragma once
#include "basis.hpp"
#include "haar_basis.hpp"
#include "orthonormal_basis.hpp"
#include "three_point_basis.hpp"

namespace Time {

// Convenience object for constructing all the time trees.
struct Bases {
  // The element tree.
  datastructures::Tree<Element1D> elem_tree;
  Element1D* mother_element;

  // The haar tree.
  datastructures::Tree<DiscConstantScalingFn> disc_cons_tree;
  datastructures::Tree<HaarWaveletFn> haar_tree;

  // The 3pt tree.
  datastructures::Tree<ContLinearScalingFn> cont_lin_tree;
  datastructures::Tree<ThreePointWaveletFn> three_point_tree;

  // Reset the orthonormal tree.
  datastructures::Tree<DiscLinearScalingFn> disc_lin_tree;
  datastructures::Tree<OrthonormalWaveletFn> ortho_tree;

  Bases()
      : elem_tree(),
        mother_element(elem_tree.meta_root->children()[0]),
        disc_cons_tree(mother_element),
        haar_tree(disc_cons_tree.meta_root->children()[0]),
        cont_lin_tree(mother_element),
        three_point_tree(cont_lin_tree.meta_root->children()),
        disc_lin_tree(mother_element),
        ortho_tree(disc_lin_tree.meta_root->children()) {}
};

}  // namespace Time
