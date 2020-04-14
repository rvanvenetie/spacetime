#include "heat_equation.hpp"

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
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

    HeatEquation<true> heat_eq(X_delta);

    // Generate some random rhs.
    for (auto nv : heat_eq.vec_X_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_random();
    }
    for (auto nv : heat_eq.vec_Y_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_random();
    }

    // Validate the input.
    ValidateVector(*heat_eq.vec_X_in());
    ValidateVector(*heat_eq.vec_Y_in());

    // Turn this into an eigen-friendly vector.
    auto v_in =
        heat_eq.BlockBF()->ToVector({heat_eq.vec_Y_in()->ToVectorContainer(),
                                     heat_eq.vec_X_in()->ToVectorContainer()});

    // Apply the block matrix :-).
    heat_eq.BlockBF()->Apply();

    // Validate the result.
    ValidateVector(*heat_eq.vec_X_out());
    ValidateVector(*heat_eq.vec_Y_out());

    // Check that the input vector remained untouched.
    auto v_now =
        heat_eq.BlockBF()->ToVector({heat_eq.vec_Y_in()->ToVectorContainer(),
                                     heat_eq.vec_X_in()->ToVectorContainer()});
    ASSERT_TRUE(v_in.isApprox(v_now));

    // Now use Eigen to atually solve something.
    Eigen::MINRES<EigenBilinearForm, Eigen::Lower | Eigen::Upper,
                  Eigen::IdentityPreconditioner>
        minres;
    minres.compute(*heat_eq.BlockBF());
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

TEST(HeatEquation, SchurCG) {
  int max_level = 7;
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  ortho_tree.UniformRefine(max_level);
  three_point_tree.UniformRefine(max_level);

  for (int level = 1; level < max_level; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        three_point_tree.meta_root.get(),
        T.hierarch_basis_tree.meta_root.get());
    X_delta.SparseRefine(level);

    HeatEquation<true> heat_eq(X_delta);

    // Generate some rhs.
    for (auto nv : heat_eq.vec_X_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(1.0);
    }

    // Validate the input.
    ValidateVector(*heat_eq.vec_X_in());

    // Turn this into an eigen-friendly vector.
    auto v_in = heat_eq.vec_X_in()->ToVectorContainer();

    // Apply the block matrix :-).
    heat_eq.SchurBF()->Apply();

    // Validate the result.
    ValidateVector(*heat_eq.vec_X_out());

    // Check that the input vector remained untouched.
    auto v_now = heat_eq.vec_X_in()->ToVectorContainer();
    ASSERT_TRUE(v_in.isApprox(v_now));

    // Now use Eigen to atually solve something.
    Eigen::ConjugateGradient<EigenBilinearForm, Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(*heat_eq.SchurBF());
    Eigen::VectorXd x;
    x = cg.solve(v_in);
    std::cout << "CG:   #iterations: " << cg.iterations()
              << ", estimated error: " << cg.error() << std::endl;
    ASSERT_NEAR(cg.error(), 0, 1e-14);

    // Store the result, and validate wether it validates.
    heat_eq.vec_X_out()->FromVectorContainer(x);
    ValidateVector(*heat_eq.vec_X_out());

    if (level == 5) {
      auto dblnodes = heat_eq.vec_X_out()->Bfs();
      Eigen::VectorXd python_output(dblnodes.size());
      python_output << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          -1.63485745, -1.57125316, 0., 0., 0., 0., 0., 0., 0., 0., -0.62675243,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          -0.07582972, 0., 0., 0., 0., 0., 0., 0., 0., -0.09725856, 0., 0., 0.,
          0., 1.12858449, 1.12858449, 1.12858449, 1.12858449, 1.16929006,
          1.16929006, 1.16929006, 1.16929006, 0., 0., 0., 0., 0., 0., 0., 0.,
          0.15233081, 0., 0., 0., 0., 0., 0., 0., 0., 0.1568453, 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0.16239127, 0., 0., 0., 0., 0., 0.,
          0., 0., 0.12308778, 0., 0., 0., 0., 0.92231708, 0.92231708,
          0.92231708, 0.92231708, 2.21977687, 0., 0., 2.21977687, 0.,
          2.21977687, 0., 0., 0., 2.21977687, 0., 0., 2.21665418, 0., 0.,
          2.21665418, 0., 2.21665418, 0., 0., 0., 2.21665418, 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.04729041, 0., 0., 0., 0., 0., 0., 0., 0.,
          0.06387484, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0.06073723, 0., 0., 0., 0., 0., 0., 0., 0., 0.05891265, 0., 0., 0.,
          0., 0.37926171, 0.37926171, 0.37926171, 0.37926171, 0., 0., 0., 0.,
          0., 0., 0., 0., 0.05899896, 0., 0., 0., 0., 0., 0., 0., 0.,
          0.06048094, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0.06600522, 0., 0., 0., 0., 0., 0., 0., 0., 0.03226911, 0., 0., 0.,
          0., 0.3621864, 0.3621864, 0.3621864, 0.3621864, 0.83205333, 0., 0.,
          0.83205333, 0., 0.83205333, 0., 0., 0., 0.83205333, 0., 0.,
          0.77184211, 1.53163442, 1.53163442, 0.77184211, 0.41810435,
          0.77184211, 0.77184211, 1.53163442, 0.41810435, 0.77184211,
          1.53163442, 0.77184211, 0.41810435, 0.77184211, 0.77184211,
          0.41810435, 0.76334412, 1.51257801, 1.51257801, 0.76334412,
          0.41129631, 0.76334412, 0.76334412, 1.51257801, 0.41129631,
          0.76334412, 1.51257801, 0.76334412, 0.41129631, 0.76334412,
          0.76334412, 0.41129631;

      for (int i = 0; i < dblnodes.size(); i++)
        ASSERT_NEAR(dblnodes[i]->value(), python_output[i], 1e-5);
    }
  }
}

TEST(HeatEquation, SchurPCG) {
  int max_level = 7;
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());

  for (int level = 1; level < max_level; level++) {
    T.hierarch_basis_tree.UniformRefine(level);
    ortho_tree.UniformRefine(level);
    three_point_tree.UniformRefine(level);
    X_delta.SparseRefine(level, {2, 1});
    HeatEquation<true> heat_eq(X_delta);

    // Generate some rhs.
    for (auto nv : heat_eq.vec_X_in()->Bfs()) {
      if (nv->node_1()->on_domain_boundary())
        nv->set_value(0.0);
      else
        nv->set_value(1.0);
    }

    // Turn this into an eigen-friendly vector.
    auto v_in = heat_eq.vec_X_in()->ToVectorContainer();

    // auto precond = Eigen::SparseMatrix<double>(v_in.rows(), v_in.rows());
    // precond.setIdentity();
    auto [result, data] =
        tools::linalg::PCG(*heat_eq.SchurBF(), v_in, *heat_eq.PrecondX(),
                           Eigen::VectorXd::Zero(v_in.rows()), 1000, 1e-5);
    auto [residual, iter] = data;
    std::cout << ortho_tree.Bfs().size() << " " << three_point_tree.Bfs().size()
              << " " << T.hierarch_basis_tree.Bfs().size() << " "
              << X_delta.Bfs().size() << " PCG:   #iterations: " << iter
              << ", estimated error: " << residual << std::endl;
  }
}

}  // namespace applications
