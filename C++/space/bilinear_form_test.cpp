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

using namespace space;
using namespace datastructures;
using ::testing::ElementsAre;

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

Eigen::MatrixXd MatrixQuad(const TreeVector<HierarchicalBasisFn>& tree_in,
                           const TreeVector<HierarchicalBasisFn>& tree_out) {
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
        // Do midpoint quadrature.
        auto eval = [&](auto pt) {
          return fn_out->eval(pt[0], pt[1]) * fn_in->eval(pt[0], pt[1]);
        };
        auto elems_fine = fn_in->level() < fn_out->level() ? fn_out->support()
                                                           : fn_in->support();
        for (auto elem : elems_fine) {
          Eigen::Vector2d p1, p2, p3, m1, m2, m3;
          p1 << elem->vertices()[0]->x, elem->vertices()[0]->y;
          p2 << elem->vertices()[1]->x, elem->vertices()[1]->y;
          p3 << elem->vertices()[2]->x, elem->vertices()[2]->y;
          m1 = (p1 + p2) / 2.0;
          m2 = (p1 + p3) / 2.0;
          m3 = (p2 + p3) / 2.0;
          quad += elem->area() / 3.0 * (eval(m1) + eval(m2) + eval(m3));
        }
      }
      mat(i, j) = quad;
    }
  return mat;
}

constexpr int max_level = 8;
TEST(BilinearForm, MassSymmetricQuadrature) {
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(max_level);
  auto vec_in = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
  vec_in.DeepRefine();
  auto vec_out = vec_in.DeepCopy();
  auto bil_form = BilinearForm<MassOperator>(vec_in, &vec_out);
  auto mat = bil_form.ToMatrix();
  auto mat_quad = MatrixQuad(vec_in, vec_out);
  ASSERT_TRUE(mat.isApprox(mat_quad));
}

TEST(BilinearForm, MassUnsymmetricQuadrature) {
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
    auto bil_form = BilinearForm<MassOperator>(vec_in, &vec_out);
    auto mat = bil_form.ToMatrix();
    auto mat_quad = MatrixQuad(vec_in, vec_out);
    ASSERT_TRUE(mat.isApprox(mat_quad));
  }
}
