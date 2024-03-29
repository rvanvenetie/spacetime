#include <cmath>
#include <map>
#include <random>
#include <set>

#include "../space/initial_triangulation.hpp"
#include "bilinear_form.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace spacetime;
using namespace space;
using namespace Time;
using namespace datastructures;

constexpr int level = 10;
constexpr int bilform_iters = 5;
constexpr int inner_iters = 10;
constexpr bool use_cache = true;

int main() {
  auto B = Time::Bases();
  auto T = InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(::level);
  B.ortho_tree.UniformRefine(::level);
  B.three_point_tree.UniformRefine(::level);

  for (size_t j = 0; j < ::bilform_iters; ++j) {
    // Setup random X_delta
    auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
        B.three_point_tree.meta_root(), T.hierarch_basis_tree.meta_root());
    X_delta.SparseRefine(::level, {2, 1});
    auto Y_delta = GenerateYDelta<DoubleTreeView>(X_delta);

    auto vec_X = X_delta.template DeepCopy<
        DoubleTreeVector<ThreePointWaveletFn, HierarchicalBasisFn>>();
    auto vec_Y = Y_delta.template DeepCopy<
        DoubleTreeVector<OrthonormalWaveletFn, HierarchicalBasisFn>>();
    auto bil_form =
        CreateBilinearForm<Time::TransportOperator, space::MassOperator>(
            &vec_X, &vec_Y, /* use_cache */ use_cache);

    // std::cout << "----" << std::endl;
    // std::cout << "X_delta size " << X_delta.Bfs().size() << " sizeof element
    // "
    //          << sizeof(decltype(X_delta)::Impl) << std::endl;
    // std::cout << "Y_delta size " << Y_delta.Bfs().size() << " sizeof element
    // "
    //          << sizeof(decltype(Y_delta)::Impl) << std::endl;
    // std::cout << "Sigma size " << bil_form->sigma()->Bfs().size() <<
    // std::endl; std::cout << "Theta size " << bil_form->theta()->Bfs().size()
    // << std::endl;
    for (size_t k = 0; k < ::inner_iters; k++) {
      for (auto& nv : vec_X.Bfs()) {
        nv->set_value(bsd_rnd());
      }
      bil_form->Apply(vec_X.ToVectorContainer());
    }
  }
  return 0;
}
