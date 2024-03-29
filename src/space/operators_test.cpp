#include "operators.hpp"

#include "datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "tools/linalg.hpp"

using namespace space;
using namespace datastructures;

constexpr int max_level = 6;

Eigen::VectorXd RandomVector(const TriangulationView &triang,
                             bool dirichlet_boundary = true) {
  Eigen::VectorXd vec(triang.V);
  vec.setRandom();
  if (dirichlet_boundary)
    for (int v = 0; v < triang.V; v++)
      if (triang.OnBoundary(v)) vec[v] = 0.0;
  return vec;
}

TEST(Operator, InverseTimesForwardOpIsIdentity) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (int level = 2; level <= max_level; ++level) {
    auto vertex_view = datastructures::TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    // Now create the corresponding element tree
    TriangulationView triang(vertex_view);
    for (bool dirichlet_boundary : {true, false}) {
      auto forward_op = StiffPlusScaledMassOperator(
          triang, {.dirichlet_boundary = dirichlet_boundary});
      auto backward_op = DirectInverse<StiffPlusScaledMassOperator>(
          triang, {.dirichlet_boundary = dirichlet_boundary});
      for (int i = 0; i < 10; i++) {
        Eigen::VectorXd vec = RandomVector(triang, dirichlet_boundary);
        Eigen::VectorXd vec2 = vec;
        forward_op.Apply(vec2);
        backward_op.Apply(vec2);
        ASSERT_TRUE(vec2.isApprox(vec));

        backward_op.Apply(vec);
        forward_op.Apply(vec);
        ASSERT_TRUE(vec.isApprox(vec2));
      }
    }
  }
}

TEST(ForwardOperator, SingleScaleMatrix) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  // Create a subtree with only vertices lying below the diagonal.
  auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
  vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
    return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
  });

  TriangulationView triang(vertex_subtree);
  MassOperator mass_op_build(triang, {.build_mat = true});
  MassOperator mass_op_nobuild(triang, {.build_mat = false});

  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd vec = RandomVector(triang);
    Eigen::VectorXd vec_copy = vec;
    mass_op_build.ApplySingleScale(vec_copy);
    mass_op_nobuild.ApplySingleScale(vec);
    ASSERT_TRUE(vec.isApprox(vec_copy));
  }
}

TEST(MultiGridOperator, RestrictProlongate) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);

  // Create a subtree with only vertices lying below the diagonal.
  auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
  vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
    return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
  });

  TriangulationView triang(vertex_subtree);
  auto mg_op = MultigridPreconditioner<MassOperator>(triang);
  size_t V = triang.V;

  // Test that prolongate of constant function remains constant.
  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd vec_fine = Eigen::VectorXd::Ones(V);

    Eigen::VectorXd v = Eigen::VectorXd::Zero(V);
    v.head(triang.InitialVertices()) =
        Eigen::VectorXd::Ones(triang.InitialVertices());

    for (size_t vi = triang.InitialVertices(); vi < V; ++vi)
      mg_op.Prolongate(vi, v);

    ASSERT_TRUE(v.isApprox(vec_fine));
  }

  // Check that prolongation & restriction work in bilinear forms.
  auto mat_fine =
      MassOperator(triang, {.dirichlet_boundary = false, .build_mat = true})
          .MatrixSingleScale();
  auto mat_coarse =
      MassOperator(triang.InitialTriangulationView(),
                   {.dirichlet_boundary = false, .build_mat = true})
          .MatrixSingleScale();
  for (int i = 0; i < 10; i++) {
    // Calculate vector on coarsest mesh and apply M_coarse.
    Eigen::VectorXd v(triang.InitialVertices());
    v.setRandom();
    Eigen::VectorXd M_coarse_v = mat_coarse * v;

    // Prolongate, apply on finest mesh, restrict.
    Eigen::VectorXd v_fine(V);
    v_fine.setZero();
    v_fine.head(triang.InitialVertices()) = v;
    Eigen::VectorXd R_M_fine_P_v = v_fine;

    // Prolongate.
    for (size_t vi = triang.InitialVertices(); vi < V; ++vi)
      mg_op.Prolongate(vi, R_M_fine_P_v);

    // Apply mass.
    R_M_fine_P_v = mat_fine * R_M_fine_P_v;

    // Restrict.
    int vi = V - 1;
    for (; vi >= triang.InitialVertices(); --vi)
      mg_op.Restrict(vi, R_M_fine_P_v);

    ASSERT_TRUE(
        R_M_fine_P_v.head(triang.InitialVertices()).isApprox(M_coarse_v));
  }

  // Test that restriction & restrictioninverse is the identity.
  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd vec(triang.V);
    vec.setRandom();
    Eigen::VectorXd vec_copy = vec;

    // Step 1: restrict this vector all the way down.
    int vi = V - 1;
    for (; vi >= triang.InitialVertices(); --vi) mg_op.Restrict(vi, vec);

    // Step 2: prolongate this vector all the way back up
    for (size_t vi = triang.InitialVertices(); vi < V; ++vi)
      mg_op.RestrictInverse(vi, vec);
    ASSERT_TRUE(vec.isApprox(vec_copy));
  }
}

template <typename ForwardOp>
void TestMultigridOperator(bool dirichlet_boundary, size_t time_level = 0) {
  // std::cout << "TestMultigridOperator<" << typeid(ForwardOp).name() << ">("
  //          << dirichlet_boundary << ", " << time_level << ")" << std::endl;

  // Test multigrid preconditioner on coarsest mesh is an exact solve.
  {
    for (size_t initial_ref = 0; initial_ref < 4; initial_ref++) {
      auto T = InitialTriangulation::UnitSquare(initial_ref);
      T.hierarch_basis_tree.UniformRefine(1);

      // Create a subtree with only vertices on coarset mesh.
      auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
      vertex_subtree.UniformRefine(0);

      TriangulationView triang(vertex_subtree);
      auto mg_op = MultigridPreconditioner<ForwardOp>(
          triang,
          {.dirichlet_boundary = dirichlet_boundary, .time_level = time_level});
      auto mass_op = ForwardOp(
          triang,
          {.dirichlet_boundary = dirichlet_boundary, .time_level = time_level});
      size_t V = triang.V;

      for (int i = 0; i < 10; i++) {
        Eigen::VectorXd vec = RandomVector(triang);
        Eigen::VectorXd vec_copy = vec;
        mass_op.Apply(vec);
        mg_op.Apply(vec);
        ASSERT_TRUE(vec.isApprox(vec_copy));
      }
    }
  }

  // Test MG works for a locally refined mesh.
  {
    auto T = InitialTriangulation::UnitSquare();
    // Create a mesh refined along the line x==y.
    size_t ml = 12;
    T.elem_tree.DeepRefine([ml](auto elem) {
      if (elem->level() >= ml) return false;
      for (auto vertex : elem->vertices())
        if (vertex->x == vertex->y) return true;
      return false;
    });

    auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
    vertex_subtree.DeepRefine();
    TriangulationView triang(vertex_subtree);

    auto mass_op = ForwardOp(triang, {.dirichlet_boundary = dirichlet_boundary,
                                      .time_level = time_level,
                                      .alpha = 1.0});
    double prev_cond = 99999999;
    for (size_t cycles = 1; cycles < 5; cycles++) {
      auto mg_op = MultigridPreconditioner<ForwardOp>(
          triang, {.dirichlet_boundary = dirichlet_boundary,
                   .time_level = time_level,
                   .alpha = 1.0,
                   .mg_cycles = cycles});

      // Evaluate condition number.
      tools::linalg::Lanczos lanczos(mass_op, mg_op,
                                     RandomVector(triang, dirichlet_boundary));

      ASSERT_LE(lanczos.cond(), 1.5);
      ASSERT_LE(lanczos.cond(), prev_cond);
      prev_cond = lanczos.cond();
    }
  }

  // Test that the multigridpreconditioner is SPD
  {
    auto T = InitialTriangulation::UnitSquare();
    T.hierarch_basis_tree.UniformRefine(max_level);

    // Create a subtree with only vertices lying below the diagonal.
    auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
    vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
      return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
    });

    TriangulationView triang(vertex_subtree);

    for (size_t cycles = 1; cycles < 5; cycles++) {
      auto mg_op = MultigridPreconditioner<ForwardOp>(
          triang, {.dirichlet_boundary = dirichlet_boundary,
                   .time_level = time_level,
                   .alpha = 1.0,
                   .mg_cycles = cycles});
      auto mg_mat = mg_op.ToMatrix();

      // Verify that the matrix is symmetric.
      ASSERT_TRUE(mg_mat.isApprox(mg_mat.transpose()));

      // Check that al its eigenvalues are real and positive.
      auto eigs = mg_mat.template selfadjointView<Eigen::Upper>().eigenvalues();
      for (size_t i = 0; i < triang.V; i++) {
        if (dirichlet_boundary)
          ASSERT_GE(eigs[i], -1e-12);
        else
          ASSERT_GT(eigs[i], 0);
      }
    }
  }
}

TEST(MultiGridOperator, All) {
  TestMultigridOperator<MassOperator>(/*dirichlet */ false);
  TestMultigridOperator<MassOperator>(/*dirichlet */ true);
  TestMultigridOperator<StiffnessOperator>(/* dirichlet */ true);
  TestMultigridOperator<StiffPlusScaledMassOperator>(/* dirichlet */ true);
  TestMultigridOperator<StiffPlusScaledMassOperator>(/* dirichlet */ true,
                                                     /* time_level */ 10);
}
