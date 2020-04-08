#include "operators.hpp"

#include "datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"

using namespace space;
using namespace datastructures;

constexpr int max_level = 6;

TEST(Operator, InverseTimesForwardOpIsIdentity) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (int level = 2; level <= max_level; ++level) {
    auto vertex_view = datastructures::TreeView<Vertex>(T.vertex_meta_root);
    vertex_view.UniformRefine(level);

    // Now create the corresponding element tree
    TriangulationView triang(vertex_view);
    for (bool dirichlet_boundary : {true, false}) {
      auto forward_op = MassOperator(triang, dirichlet_boundary);
      auto backward_op =
          DirectInverse<MassOperator>(triang, dirichlet_boundary);
      for (int i = 0; i < 10; i++) {
        Eigen::VectorXd vec(triang.vertices().size());
        vec.setRandom();
        if (dirichlet_boundary)
          for (int v = 0; v < triang.vertices().size(); v++)
            if (triang.vertices()[v]->on_domain_boundary) vec[v] = 0.0;
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
  size_t V = triang.vertices().size();

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
  auto mat_fine = MassOperator(triang, false).MatrixSingleScale();
  auto mat_coarse = MassOperator(triang.InitialTriangulationView(), false)
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
    Eigen::VectorXd vec(triang.vertices().size());
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

TEST(MultiGridOperator, CoarsestMesh) {
  for (size_t initial_ref = 0; initial_ref < 4; initial_ref++) {
    auto T = InitialTriangulation::UnitSquare(initial_ref);
    T.hierarch_basis_tree.UniformRefine(1);

    // Create a subtree with only vertices on coarset mesh.
    auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
    vertex_subtree.UniformRefine(0);

    TriangulationView triang(vertex_subtree);
    auto mg_op = MultigridPreconditioner<MassOperator>(triang);
    auto mass_op = MassOperator(triang);
    size_t V = triang.vertices().size();

    for (int i = 0; i < 10; i++) {
      Eigen::VectorXd vec(V);
      vec.setRandom();
      for (int v = 0; v < triang.vertices().size(); v++)
        if (triang.vertices()[v]->on_domain_boundary) vec[v] = 0.0;
      Eigen::VectorXd vec_copy = vec;
      mass_op.Apply(vec);
      mg_op.Apply(vec);
      ASSERT_TRUE(vec.isApprox(vec_copy));
    }
  }
}

TEST(MultiGridOperator, MultilevelMesh) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(5);

  // Create a subtree with only vertices lying below the diagonal.
  auto vertex_subtree = TreeView<Vertex>(T.vertex_meta_root);
  vertex_subtree.DeepRefine(/* call_filter */ [](const auto &vertex) {
    return vertex->level() == 0 || (vertex->x + vertex->y <= 1.0);
  });
  // vertex_subtree.UniformRefine(2);

  TriangulationView triang(vertex_subtree);

  bool dirichlet_boundary = true;
  auto mg_op =
      MultigridPreconditioner<MassOperator>(triang, dirichlet_boundary);
  auto mass_op = MassOperator(triang, dirichlet_boundary);
  size_t V = triang.vertices().size();

  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd vec = Eigen::VectorXd::Ones(V);
    if (dirichlet_boundary)
      for (int v = 0; v < triang.vertices().size(); v++)
        if (triang.vertices()[v]->on_domain_boundary) vec[v] = 0.0;
    Eigen::VectorXd vec_copy = vec;
    mass_op.ApplySingleScale(vec);
    mg_op.ApplySingleScale(vec);
    std::cout << "vec - vec_copy " << vec - vec_copy << std::endl;
    ASSERT_TRUE(vec.isApprox(vec_copy));
  }
}
