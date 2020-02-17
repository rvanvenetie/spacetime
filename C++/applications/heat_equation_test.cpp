#include "heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::GenerateYDelta;
using Time::ortho_tree;
using Time::three_point_tree;

namespace applications {

// Function that can be used to validate whether we have a valid vector.
template <typename DblVec>
auto ValidateVector(const DblVec &vec) {
  for (const auto &node : vec.container()) {
    if (node.is_metaroot() || node.node_1()->on_domain_boundary())
      assert(node.value() == 0);
  }
}

TEST(HeatEquation, SparseMatVec) {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(6);
  ortho_tree.UniformRefine(6);
  three_point_tree.UniformRefine(6);

  for (int level = 1; level < 6; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);

    HeatEquation heat_eq(X_delta);

    // Generate some random rhs.
    for (auto nv : heat_eq.vec_X_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(((double)std::rand()) / RAND_MAX);
    }
    for (auto nv : heat_eq.vec_Y_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(((double)std::rand()) / RAND_MAX);
    }

    // Validate the input.
    ValidateVector(*heat_eq.vec_X_in());
    ValidateVector(*heat_eq.vec_Y_in());

    // Turn this into an eigen-friendly vector.
    auto v_in =
        heat_eq.BlockMat()->ToVector({heat_eq.vec_Y_in()->ToVectorContainer(),
                                      heat_eq.vec_X_in()->ToVectorContainer()});

    // Apply the block matrix :-).
    heat_eq.BlockMat()->Apply();

    // Validate the result.
    ValidateVector(*heat_eq.vec_X_out());
    ValidateVector(*heat_eq.vec_Y_out());

    // Check that the input vector remained untouched.
    auto v_now =
        heat_eq.BlockMat()->ToVector({heat_eq.vec_Y_in()->ToVectorContainer(),
                                      heat_eq.vec_X_in()->ToVectorContainer()});
    ASSERT_TRUE(v_in.isApprox(v_now));

    // Now use Eigen to atually solve something.
    Eigen::MINRES<EigenBilinearForm, Eigen::Lower | Eigen::Upper,
                  Eigen::IdentityPreconditioner>
        minres;
    minres.compute(*heat_eq.BlockMat());
    Eigen::VectorXd x;
    x = minres.solve(v_in);
    std::cout << "MINRES:   #iterations: " << minres.iterations()
              << ", estimated error: " << minres.error() << std::endl;
    ASSERT_NEAR(minres.error(), 0, 1e-14);

    // Store the result, and validate wether it validates.
    size_t i = 0;
    for (auto &node : heat_eq.vec_Y_out()->container()) node.set_value(x(i++));
    for (auto &node : heat_eq.vec_X_out()->container()) node.set_value(x(i++));
    ValidateVector(*heat_eq.vec_X_out());
    ValidateVector(*heat_eq.vec_Y_out());
  }
}
}  // namespace applications
