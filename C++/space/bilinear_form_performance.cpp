#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "bilinear_form.hpp"
#include "initial_triangulation.hpp"

int bsd_rnd() {
  static unsigned int seed = 0;
  int a = 1103515245;
  int c = 12345;
  unsigned int m = 2147483648;
  return (seed = (a * seed + c) % m);
}

using namespace space;
using namespace datastructures;

constexpr int bilform_iters = 5;

int main() {
  auto T = InitialTriangulation::UnitSquare();

  size_t level = 0;
  while (true) {
    level++;
    T.hierarch_basis_tree.UniformRefine(level);
    std::chrono::duration<double> time_create{0};
    std::chrono::duration<double> time_apply{0};
    size_t dofs = 0;
    for (size_t j = 0; j < ::bilform_iters; ++j) {
      space::OperatorOptions space_opts({.build_mat = false, .mg_cycles = 1});

      // Set random tree.
      auto vec_in = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
      vec_in.UniformRefine(level);
      dofs = vec_in.Bfs().size();
      for (auto v : vec_in.Bfs())
        if (!v->node()->on_domain_boundary())
          v->set_value(static_cast<double>(std::rand()) / RAND_MAX);

      // Create BilForm.
      auto time_compute = std::chrono::steady_clock::now();
      auto bilform =
          CreateBilinearForm<MultigridPreconditioner<space::StiffnessOperator>>(
              vec_in, vec_in, space_opts);
      time_create += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);

      // Apply BilForm.
      time_compute = std::chrono::steady_clock::now();
      bilform.Apply();
      time_apply += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - time_compute);
    }
    std::cout << "\nlevel: " << level << "\tspace-size: " << dofs
              << "\ttime-create: " << time_create.count()
              << "\ttime-apply: " << time_apply.count()
              << "\ttime-create-per-dof: " << time_create.count() / dofs
              << "\ttime-apply-per-dof: " << time_apply.count() / dofs;
  }
  return 0;
}
