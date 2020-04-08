#include "bilinear_form.hpp"

#include <cmath>
#include <map>
#include <set>

#include "datastructures/multi_tree_view.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "operators.hpp"
#include "space/initial_triangulation.hpp"
#include "space/triangulation.hpp"
#include "space/triangulation_view.hpp"
#include "tools/integration.hpp"

using namespace space;
using namespace datastructures;
using ::testing::ElementsAre;
using tools::IntegrationRule;

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

Eigen::VectorXd RandomVector(const TreeVector<HierarchicalBasisFn>& vec) {
  auto nodes = vec.Bfs();
  Eigen::VectorXd result(nodes.size());
  result.setRandom();
  for (int v = 0; v < nodes.size(); v++)
    if (nodes[v]->node()->on_domain_boundary()) result[v] = 0;
  return result;
}

Eigen::MatrixXd MatrixQuad(const TreeVector<HierarchicalBasisFn>& tree_in,
                           const TreeVector<HierarchicalBasisFn>& tree_out,
                           bool deriv) {
  auto functions_in = tree_in.Bfs();
  auto functions_out = tree_out.Bfs();
  Eigen::MatrixXd mat(functions_out.size(), functions_in.size());

  for (size_t i = 0; i < functions_out.size(); ++i)
    for (size_t j = 0; j < functions_in.size(); ++j) {
      double quad = 0;
      auto fn_out = functions_out[i]->node();
      auto fn_in = functions_in[j]->node();
      if (!fn_out->vertex()->on_domain_boundary &&
          !fn_in->vertex()->on_domain_boundary) {
        auto elems_fine = fn_in->level() < fn_out->level() ? fn_out->support()
                                                           : fn_in->support();
        for (auto elem : elems_fine) {
          if (deriv) {
            quad += IntegrationRule</*dim*/ 2, /*degree*/ 0>::Integrate(
                [&](double x, double y) {
                  return fn_out->EvalGrad(x, y).dot(fn_in->EvalGrad(x, y));
                },
                *elem);
          } else {
            quad += IntegrationRule</*dim*/ 2, /*degree*/ 2>::Integrate(
                [&](double x, double y) {
                  return fn_out->Eval(x, y) * fn_in->Eval(x, y);
                },
                *elem);
          }
        }
      }
      mat(i, j) = quad;
    }
  return mat;
}

constexpr int max_level = 3;

TEST(BilinearForm, SymmetricQuadrature) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  auto vec_in = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
  vec_in.DeepRefine();
  auto vec_out = vec_in.DeepCopy();
  auto mass_bil_form = CreateBilinearForm<MassOperator>(vec_in, vec_out);
  auto mass_mat = mass_bil_form.ToMatrix();
  auto mass_quad = MatrixQuad(vec_in, vec_out, /*deriv*/ false);
  ASSERT_TRUE(mass_mat.isApprox(mass_quad));

  // Check that the transpose is correct
  auto mass_tmat = mass_bil_form.Transpose().ToMatrix();
  ASSERT_TRUE(mass_mat.transpose().isApprox(mass_tmat));

  // Check also the apply of a random vector.
  Eigen::VectorXd v = RandomVector(vec_in);
  vec_in.FromVector(v);
  mass_bil_form.Apply();
  ASSERT_TRUE(vec_out.ToVector().isApprox(mass_quad * v));

  auto stiff_bil_form = CreateBilinearForm<StiffnessOperator>(vec_in, vec_out);
  auto stiff_mat = stiff_bil_form.ToMatrix();
  auto stiff_quad = MatrixQuad(vec_in, vec_out, /*deriv*/ true);
  ASSERT_TRUE(stiff_mat.isApprox(stiff_quad));

  // Check that the transpose is correct
  auto stiff_tmat = stiff_bil_form.Transpose().ToMatrix();
  ASSERT_TRUE(stiff_mat.transpose().isApprox(stiff_tmat));

  // Check also the apply of a random vector.
  v = RandomVector(vec_in);
  vec_in.FromVector(v);
  stiff_bil_form.Apply();
  ASSERT_TRUE(vec_out.ToVector().isApprox(stiff_quad * v));
}

TEST(BilinearForm, UnsymmetricQuadrature) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  for (size_t j = 0; j < 20; ++j) {
    auto vec_in = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    auto vec_out = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    vec_in.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    vec_out.DeepRefine(
        /* call_filter */ [](auto&& nv) {
          return nv->level() <= 0 || bsd_rnd() % 3 != 0;
        });
    auto mass_bil_form = CreateBilinearForm<MassOperator>(vec_in, vec_out);
    auto mass_mat = mass_bil_form.ToMatrix();
    auto mass_quad = MatrixQuad(vec_in, vec_out, /*deriv*/ false);
    ASSERT_TRUE(mass_mat.isApprox(mass_quad));

    // Check that the transpose is correct
    auto mass_tmat = mass_bil_form.Transpose().ToMatrix();
    ASSERT_TRUE(mass_mat.transpose().isApprox(mass_tmat));

    // Check also the apply of a random vector.
    Eigen::VectorXd v = RandomVector(vec_in);
    vec_in.FromVector(v);
    mass_bil_form.Apply();
    ASSERT_TRUE(vec_out.ToVector().isApprox(mass_quad * v));

    auto stiff_bil_form =
        CreateBilinearForm<StiffnessOperator>(vec_in, vec_out);
    auto stiff_mat = stiff_bil_form.ToMatrix();
    auto stiff_quad = MatrixQuad(vec_in, vec_out, /*deriv*/ true);
    ASSERT_TRUE(stiff_mat.isApprox(stiff_quad));

    // Check that the transpose is correct
    auto stiff_tmat = stiff_bil_form.Transpose().ToMatrix();
    ASSERT_TRUE(stiff_mat.transpose().isApprox(stiff_tmat));

    // Check also the apply of a random vector.
    v = RandomVector(vec_in);
    vec_in.FromVector(v);
    stiff_bil_form.Apply();
    ASSERT_TRUE(vec_out.ToVector().isApprox(stiff_quad * v));
  }
}
