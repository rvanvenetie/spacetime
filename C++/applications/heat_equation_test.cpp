#include "heat_equation.hpp"

#include <unsupported/Eigen/IterativeSolvers>

#include "../space/initial_triangulation.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using spacetime::BilinearFormBase;

namespace applications {

// Function that can be used to validate whether we have a valid vector.
template <typename DblVec>
auto ValidateVector(const DblVec &vec) {
  for (const auto &node : vec.container()) {
    if (node.is_metaroot() || node.node_1()->on_domain_boundary())
      assert(node.value() == 0);
  }
}

TEST(HeatEquation, CompareToPython) {
  for (bool use_cache : {true, false}) {
    int level = 6;
    auto B = Time::Bases();
    auto T = space::InitialTriangulation::UnitSquare();
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());

    T.hierarch_basis_tree.UniformRefine(level);
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    X_delta.SparseRefine(level, {2, 1});
    HeatEquationOptions opts;
    opts.use_cache = use_cache;
    opts.PX_alpha = 1.0;
    HeatEquation heat_eq(X_delta, opts);

    // Set double-tree vectors
    auto X_vec = heat_eq.vec_X();
    auto Y_vec = heat_eq.vec_Y();
    auto X_bfs = X_vec->Bfs();
    auto Y_bfs = Y_vec->Bfs();
    for (size_t i = 0; i < X_bfs.size(); i++)
      if (X_bfs[i]->node_1()->on_domain_boundary())
        X_bfs[i]->set_value(0);
      else
        X_bfs[i]->set_value(i);
    for (size_t i = 0; i < Y_bfs.size(); i++)
      if (Y_bfs[i]->node_1()->on_domain_boundary())
        Y_bfs[i]->set_value(0);
      else
        Y_bfs[i]->set_value(i);

    auto X_in = heat_eq.vec_X()->ToVectorContainer();
    auto Y_in = heat_eq.vec_Y()->ToVectorContainer();

// Load the vectors in Python format.
#include "heat_equation_python.ipp"

    // Create a little compare function.
    auto compare = [&](auto bfs, auto res_py) {
      ASSERT_EQ(bfs.size(), res_py.size());
      for (size_t i = 0; i < bfs.size(); i++) {
        auto [level, index, x, y] = res_py[i].first;
        double value = res_py[i].second;
        ASSERT_EQ(bfs[i]->node_0()->level(), level);
        ASSERT_EQ(bfs[i]->node_0()->index(), index);
        ASSERT_EQ(bfs[i]->node_1()->vertex()->x, x);
        ASSERT_EQ(bfs[i]->node_1()->vertex()->y, y);
        ASSERT_NEAR(bfs[i]->value(), value, 1e-8);
      }
    };

    // Compare the results in C++ to the Python format.

    // For A * v.
    std::cout << "Comparing A" << std::endl;
    Y_vec->FromVectorContainer(heat_eq.A()->Apply(Y_in));
    compare(Y_bfs, A_py);

    // For B * v.
    std::cout << "Comparing B" << std::endl;
    Y_vec->FromVectorContainer(heat_eq.B()->Apply(X_in));
    compare(Y_bfs, B_py);

    // For B.T * v.
    std::cout << "Comparing B.T" << std::endl;
    X_vec->FromVectorContainer(heat_eq.BT()->Apply(Y_in));
    compare(X_bfs, BT_py);

    // For G * v
    std::cout << "Comparing G" << std::endl;
    X_vec->FromVectorContainer(heat_eq.G()->Apply(X_in));
    compare(X_bfs, G_py);

    // For P_Y * v
    std::cout << "Comparing P_Y" << std::endl;
    Y_vec->FromVectorContainer(heat_eq.P_Y()->Apply(Y_in));
    compare(Y_bfs, A_inv_py);

    // For precond_X * v
    std::cout << "Comparing P_X" << std::endl;
    X_vec->FromVectorContainer(heat_eq.P_X()->Apply(X_in));
    compare(X_bfs, precond_X_py);

    // For schur_mat * v
    std::cout << "Comparing schur_mat" << std::endl;
    X_vec->FromVectorContainer(heat_eq.S()->Apply(X_in));
    compare(X_bfs, schur_mat_py);
  }
}

TEST(HeatEquation, SchurCG) {
  int max_level = 7;
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  B.ortho_tree.UniformRefine(max_level);
  B.three_point_tree.UniformRefine(max_level);

  for (int level = 1; level < max_level; level++) {
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(level);

    HeatEquation heat_eq(X_delta);

    // Generate some rhs.
    for (auto nv : heat_eq.vec_X()->Bfs()) {
      if (nv->node_1()->on_domain_boundary()) continue;
      nv->set_value(1.0);
    }

    // Validate the input.
    ValidateVector(*heat_eq.vec_X());

    // Turn this into an eigen-friendly vector.
    auto v_in = heat_eq.vec_X()->ToVectorContainer();

    // Apply the block matrix :-).
    heat_eq.vec_X()->FromVectorContainer(heat_eq.S()->Apply(v_in));

    // Validate the result.
    ValidateVector(*heat_eq.vec_X());

    // Now use Eigen to atually solve something.
    Eigen::ConjugateGradient<BilinearFormBase<HeatEquation::TypeXVector>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::IdentityPreconditioner>
        cg;
    cg.compute(*heat_eq.S());
    Eigen::VectorXd x;
    x = cg.solve(v_in);
    std::cout << "CG:   #iterations: " << cg.iterations()
              << ", estimated error: " << cg.error() << std::endl;
    ASSERT_NEAR(cg.error(), 0, 1e-14);

    // Store the result, and validate wether it validates.
    heat_eq.vec_X()->FromVectorContainer(x);
    ValidateVector(*heat_eq.vec_X());

    if (level == 5) {
      auto dblnodes = heat_eq.vec_X()->Bfs();
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
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  for (int level = 1; level < max_level; level++) {
    T.hierarch_basis_tree.UniformRefine(level);
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    X_delta.SparseRefine(level, {2, 1});
    HeatEquation heat_eq(X_delta);

    // Generate some rhs.
    for (auto nv : heat_eq.vec_X()->Bfs()) {
      if (nv->node_1()->on_domain_boundary())
        nv->set_value(0.0);
      else
        nv->set_value(1.0);
    }

    // Turn this into an eigen-friendly vector.
    auto v_in = heat_eq.vec_X()->ToVectorContainer();

    // auto precond = Eigen::SparseMatrix<double>(v_in.rows(), v_in.rows());
    // precond.setIdentity();
    auto [result, data] =
        tools::linalg::PCG(*heat_eq.S(), v_in, *heat_eq.P_X(),
                           Eigen::VectorXd::Zero(v_in.rows()), 1000, 1e-5);
    std::cout << B.ortho_tree.Bfs().size() << " "
              << B.three_point_tree.Bfs().size() << " "
              << T.hierarch_basis_tree.Bfs().size() << " "
              << X_delta.Bfs().size()
              << " PCG:   #iterations: " << data.iterations
              << ", estimated error: " << data.relative_residual << std::endl;
  }
}

TEST(HeatEquation, LanczosDirectInverse) {
  int max_level = 12;
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  for (int level = 1; level <= max_level; level++) {
    if (level % 2) continue;
    T.hierarch_basis_tree.UniformRefine(level);
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    X_delta.SparseRefine(level, {2, 1});
    HeatEquation heat_eq(X_delta);

    std::cout << "Level " << level << "; #(X_delta, Y_delta) = ("
              << heat_eq.vec_X()->Bfs().size() << ", "
              << heat_eq.vec_Y()->Bfs().size() << ")" << std::endl;

    // Generate some initial Y_delta rhs.
    for (auto nv : heat_eq.vec_Y()->Bfs())
      if (!nv->node_1()->on_domain_boundary())
        nv->set_random();
      else
        nv->set_value(0);
    auto lanczos_Y = tools::linalg::Lanczos(
        *heat_eq.A(), *heat_eq.P_Y(), heat_eq.vec_Y()->ToVectorContainer());
    std::cout << "\tkappa(P_Y * A_s): " << lanczos_Y << std::endl;
    ASSERT_NEAR(lanczos_Y.cond(), 1., 1e-5);

    // Generate some initial X_delta rhs.
    for (auto nv : heat_eq.vec_X()->Bfs())
      if (!nv->node_1()->on_domain_boundary()) nv->set_random();
    auto lanczos_X = tools::linalg::Lanczos(
        *heat_eq.S(), *heat_eq.P_X(), heat_eq.vec_X()->ToVectorContainer());
    std::cout << "\tkappa(P_X * S) :" << lanczos_X << std::endl << std::endl;
    ASSERT_LT(lanczos_Y.cond(), 6);
  }
}

TEST(HeatEquation, LanczosMG) {
  int max_level = 10;
  auto B = Time::Bases();
  auto T = space::InitialTriangulation::UnitSquare();
  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());

  for (int level = 1; level <= max_level; level++) {
    if (level % 2) continue;
    T.hierarch_basis_tree.UniformRefine(level);
    B.ortho_tree.UniformRefine(level);
    B.three_point_tree.UniformRefine(level);
    X_delta.SparseRefine(level, {2, 1});

    HeatEquationOptions heat_eq_opts;
    heat_eq_opts.PX_inv = HeatEquationOptions::SpaceInverse::Multigrid;
    heat_eq_opts.PY_inv = HeatEquationOptions::SpaceInverse::Multigrid;
    HeatEquation heat_eq(X_delta, heat_eq_opts);

    std::cout << "Level " << level << "; #(X_delta, Y_delta) = ("
              << heat_eq.vec_X()->Bfs().size() << ", "
              << heat_eq.vec_Y()->Bfs().size() << ")" << std::endl;

    // Generate some initial Y_delta rhs.
    for (auto nv : heat_eq.vec_Y()->Bfs())
      if (!nv->node_1()->on_domain_boundary())
        nv->set_random();
      else
        nv->set_value(0);
    auto lanczos_Y = tools::linalg::Lanczos(
        *heat_eq.A(), *heat_eq.P_Y(), heat_eq.vec_Y()->ToVectorContainer());
    std::cout << "\tkappa(P_Y * A_s): " << lanczos_Y << std::endl;
    ASSERT_NEAR(lanczos_Y.cond(), 1., 5e-2);

    // Generate some initial X_delta rhs.
    for (auto nv : heat_eq.vec_X()->Bfs())
      if (!nv->node_1()->on_domain_boundary()) nv->set_random();
    auto lanczos_X = tools::linalg::Lanczos(
        *heat_eq.S(), *heat_eq.P_X(), heat_eq.vec_X()->ToVectorContainer());
    std::cout << "\tkappa(P_X * S) :" << lanczos_X << std::endl << std::endl;
    ASSERT_LT(lanczos_Y.cond(), 5.5);
  }
}

}  // namespace applications
