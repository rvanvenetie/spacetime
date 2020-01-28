#include "multi_tree_vector.hpp"

#include <cmath>
#include <cstdlib>
#include <map>
#include <set>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "space/initial_triangulation.hpp"
#include "space/triangulation.hpp"

using namespace space;
using namespace datastructures;
using ::testing::ElementsAre;

constexpr int max_level = 4;

TEST(TreeVector, add) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vec = TreeVector<Vertex>(T.vertex_meta_root);
  vec.DeepRefine();
  auto vec_items = vec.Bfs();

  for (const auto &nv : vec_items) {
    nv->set_value(1.0 + ((double)std::rand()) / RAND_MAX);
  }

  // Make a copy, assert the values are close.
  auto vec_copy = vec.DeepCopy();
  auto vec_copy_items = vec_copy.Bfs();
  ASSERT_EQ(vec_items.size(), vec_copy_items.size());
  for (size_t i = 0; i < vec_items.size(); ++i) {
    ASSERT_DOUBLE_EQ(vec_items[i]->value(), vec_copy_items[i]->value());
  }

  // Add something to vec_copy, assert some things.
  auto vec_values = vec.ToVector();
  vec_copy += vec;
  for (size_t i = 0; i < vec_items.size(); ++i)
    ASSERT_DOUBLE_EQ(2 * vec_items[i]->value(), vec_copy_items[i]->value());

  // Multiply vec, assert some things.
  vec *= 1.5;
  for (size_t i = 0; i < vec_items.size(); ++i)
    ASSERT_DOUBLE_EQ(vec_values[i] * 1.5, vec_items[i]->value());

  // Create a unit vector on a coarser grid.
  auto vec2 = TreeVector<Vertex>(T.vertex_meta_root);
  vec2.UniformRefine(2);
  auto vec2_items = vec2.Bfs();
  for (const auto &nv : vec2_items) nv->set_value(1.0);
  ASSERT_TRUE(vec2.Bfs().size() < vec.Bfs().size());

  // Add vec to it, which is defined on a finer grid.
  vec2 += vec;
  ASSERT_EQ(vec2.Bfs().size(), vec.Bfs().size());
  vec2_items = vec2.Bfs();
  // Assert that `vec2` now indeed has 1+v[node] for node.level <= 1,
  // and v[node] for node.level > 1.
  for (size_t i = 0; i < vec_items.size(); i++)
    ASSERT_DOUBLE_EQ(
        vec2_items[i]->value(),
        vec_items[i]->value() + (vec_items[i]->level() <= 2 ? 1 : 0));
}
TEST(MultiTreeVector, basic) {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_tree.UniformRefine(max_level);
  auto vertices = T.vertex_meta_root->Bfs(/*include_metaroot*/ true);
  auto elements = T.elem_meta_root->Bfs(/*include_metaroot*/ true);

  // Fill sparse vector with random junk.
  auto vec_sparse = TripleTreeVector<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  vec_sparse.SparseRefine(3);
  for (auto &nv : vec_sparse.Bfs())
    nv->set_value((double)std::rand() / RAND_MAX);

  // Fill another uniform vector with random junk.
  auto vec_unif = TripleTreeVector<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);
  vec_unif.UniformRefine({1, 5, 2});
  for (auto &nv : vec_unif.Bfs()) nv->set_value((double)std::rand() / RAND_MAX);

  // Create empty vector.
  auto vec_result = TripleTreeVector<Vertex, Element2D, Vertex>(
      T.vertex_meta_root, T.elem_meta_root, T.vertex_meta_root);

  // vec_result = vec_sparse * 3.14 + vec_unif
  vec_result += vec_sparse;
  vec_result *= 3.14;
  vec_result += vec_unif;

  // Verify that this is correct using a dictionary.
  std::map<TripleTreeVector<Vertex, Element2D, Vertex>::Impl::TupleNodes,
           double>
      vec_dict;
  for (auto nv : vec_sparse.Bfs()) vec_dict[nv->nodes()] = 3.14 * nv->value();
  for (auto nv : vec_unif.Bfs()) vec_dict[nv->nodes()] += nv->value();

  for (auto nv : vec_result.Bfs())
    ASSERT_DOUBLE_EQ(nv->value(), vec_dict[nv->nodes()]);
}
