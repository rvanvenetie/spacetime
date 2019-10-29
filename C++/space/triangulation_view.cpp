#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

using namespace space;

int main() {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_meta_root->UniformRefine(1);

  datastructures::NodeView<Vertex> root_view(T.vertex_meta_root);
  std::shared_ptr<Vertex> child = T.vertex_meta_root->children()[0];
  auto child_view = std::make_shared<datastructures::NodeView<Vertex>>(child);

  root_view.children(0).push_back(child_view);

  int x = child_view->level();

  std::cout << x << std::endl;
  root_view.Refine<0>();
  return 0;
}
