#include <chrono>

#include "../space/initial_triangulation.hpp"
#include "../spacetime/linear_form.hpp"
#include "../time/basis.hpp"
#include "../tools/linalg.hpp"
#include "adaptive_heat_equation.hpp"

using applications::AdaptiveHeatEquation;
using datastructures::DoubleTreeVector;
using datastructures::DoubleTreeView;
using space::HierarchicalBasisFn;
using spacetime::CreateQuadratureLinearForm;
using spacetime::CreateSumLinearForm;
using spacetime::CreateZeroEvalLinearForm;
using spacetime::GenerateYDelta;
using Time::ortho_tree;
using Time::OrthonormalWaveletFn;
using Time::three_point_tree;
using Time::ThreePointWaveletFn;

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>

unsigned long long getmem() {
  task_t task = MACH_PORT_NULL;
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

  assert(KERN_SUCCESS == task_info(mach_task_self(), TASK_BASIC_INFO,
                                   (task_info_t)&t_info, &t_info_count));
  return t_info.resident_size / 1024;
}
#else
/*
 * Measures the current (and peak) resident and virtual memories
 * usage of your linux C process, in kB
 */
int getmem() {
  int peakRealMem;
  // stores each word in status file
  char buffer[1024] = "";

  // linux file contains this-process info
  FILE* file = fopen("/proc/self/status", "r");

  // read the entire file
  while (fscanf(file, " %1023s", buffer) == 1) {
    if (strcmp(buffer, "VmHWM:") == 0) {
      fscanf(file, " %d", peakRealMem);
    }
  }
  fclose(file);
  return peakRealMem;
}
#endif

auto SmoothProblem() {
  // Solution u = (1 + t^2) x (1-x) y (1-y).
  auto time_g1 = [](double t) { return -2 * (1 + t * t); };
  auto space_g1 = [](double x, double y) { return (x - 1) * x + (y - 1) * y; };
  auto time_g2 = [](double t) { return 2 * t; };
  auto space_g2 = [](double x, double y) { return (x - 1) * x * (y - 1) * y; };
  auto u0 = [](double x, double y) { return (1 - x) * x * (1 - y) * y; };

  return std::make_pair(
      CreateSumLinearForm<OrthonormalWaveletFn>(
          CreateQuadratureLinearForm<OrthonormalWaveletFn, 2, 2>(time_g1,
                                                                 space_g1),
          CreateQuadratureLinearForm<OrthonormalWaveletFn, 1, 4>(time_g2,
                                                                 space_g2)),
      CreateZeroEvalLinearForm<ThreePointWaveletFn, 4>(u0));
}

int main() {
  auto T = space::InitialTriangulation::UnitSquare();
  T.hierarch_basis_tree.UniformRefine(1);
  ortho_tree.UniformRefine(1);
  three_point_tree.UniformRefine(1);

  auto X_delta = DoubleTreeView<ThreePointWaveletFn, HierarchicalBasisFn>(
      three_point_tree.meta_root.get(), T.hierarch_basis_tree.meta_root.get());
  X_delta.SparseRefine(1);

  auto [g_lf, u0_lf] = SmoothProblem();

  AdaptiveHeatEquation heat_eq(std::move(X_delta), std::move(g_lf),
                               std::move(u0_lf));

  while (true) {
    auto start = std::chrono::steady_clock::now();
    auto solution = heat_eq.Solve(heat_eq.vec_Xd_out()->ToVectorContainer());
    auto [residual, residual_norm] = heat_eq.Estimate(/*mean_zero*/ false);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "XDelta-size: " << solution->container().size()
              << " residual-norm: " << residual_norm
              << " total-memory-kB: " << getmem()
              << " solve-estimate-time: " << elapsed_seconds.count()
              << std::endl;
    auto marked_nodes = heat_eq.Mark();
    heat_eq.Refine(marked_nodes);
  }

  return 0;
}
