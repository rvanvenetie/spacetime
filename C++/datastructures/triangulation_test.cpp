#include "triangulation.hpp"

int main() {
  auto T = InitialTriangulation::unit_square();
  auto elems = T.elem_meta_root->Bfs();

  std::cout << "Elements in BFS before refine" << std::endl;
  for (auto elem : elems) {
    std::cout << *elem << ", ";
  }
  std::cout << std::endl;

  T.elem_meta_root->UniformRefine(2);
  elems = T.elem_meta_root->Bfs();
  assert(elems.size() == 2 + 4 + 8);
  std::cout << "Elements in BFS after refine" << std::endl;
  for (auto elem : elems) {
    std::cout << *elem << ", ";
  }
  std::cout << std::endl;

  return 0;
}
