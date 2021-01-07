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
    std::chrono::duration<double> time_create_bfs{0};
    std::chrono::duration<double> time_create_tview{0};
    std::chrono::duration<double> time_tview_transform{0};
    std::chrono::duration<double> time_tview_mark{0};
    std::chrono::duration<double> time_tview_bfs{0};
    std::chrono::duration<double> time_create_operator{0};
    std::chrono::duration<double> time_apply{0};

    // Create Tree.
    auto vec_in = TreeVector<HierarchicalBasisFn>(T.hierarch_basis_meta_root);
    vec_in.UniformRefine(level);
    size_t dofs = vec_in.Bfs().size();

    for (size_t j = 0; j < ::bilform_iters; ++j) {
      space::OperatorOptions space_opts({.build_mat = false, .mg_cycles = 1});

      // Set Random values.
      for (auto v : vec_in.Bfs())
        if (!v->node()->on_domain_boundary())
          v->set_value(static_cast<double>(std::rand()) / RAND_MAX);

      // Create BilForm.
      auto time_compute = std::chrono::high_resolution_clock::now();
      auto bilform =
          CreateBilinearForm<StiffnessOperator>(
              vec_in, vec_in, space_opts);
      time_create += std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - time_compute);
      time_create_bfs += bilform.time_bfs_;
      time_create_tview += bilform.time_tview_;
      time_create_operator += bilform.time_operator_;
      time_tview_transform += bilform.TView()->time_transform_;
      time_tview_mark += bilform.TView()->time_mark_;
      time_tview_bfs += bilform.TView()->time_bfs_;

      // Apply BilForm.
      time_compute = std::chrono::high_resolution_clock::now();
      bilform.Apply();
      time_apply += std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - time_compute);
    }
    std::cout << "\nlevel: " << level << "\tspace-size: " << dofs
              << "\ttime-apply-per-dof: " << time_apply.count() / dofs
              << "\ttime-create-per-dof: " << time_create.count() / dofs
              << "\ttime-create-bfs-per-dof: " << time_create_bfs.count() / dofs
              << "\ttime-create-tview-per-dof: "
              << time_create_tview.count() / dofs
              << "\ttime-create-operator-per-dof: "
              << time_create_operator.count() / dofs
              << "\ttime-tview-transform-per-dof: "
              << time_tview_transform.count() / dofs
              << "\ttime-tview-mark-per-dof: " << time_tview_mark.count() / dofs
              << "\ttime-tview-bfs-per-dof: " << time_tview_bfs.count() / dofs
              << std::flush;
  }
  return 0;
}
