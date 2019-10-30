#include "datastructures/multi_tree_view.hpp"
#include "triangulation.hpp"

using namespace space;

int main() {
  auto T = InitialTriangulation::UnitSquare();
  T.elem_meta_root->UniformRefine(3);

  auto root_view =
      std::make_shared<datastructures::NodeView<Vertex>>(T.vertex_meta_root);

  std::cout << root_view->level() << std::endl;
  std::cout << root_view->is_metaroot() << std::endl;

  std::cout << root_view->children(0).size() << std::endl;
  root_view->Refine<0>();

  std::cout << root_view->children(0).size() << std::endl;

  auto bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine<0>();
  }

  std::cout << " ---- after uniform refinemenet --- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine<0>();
  }

  std::cout << " ---- after uniform refinemenet --- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine<0>();
  }

  auto bla_view =
      std::make_shared<datastructures::NodeView<Vertex>>(T.vertex_meta_root);
  bla_view->DeepRefine();
  root_view->Union(bla_view);

  std::cout << " --- after union with a deep refine -- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine<0>();
  }
  //
  //  int x = child_view->level();
  //
  //  std::cout << x << std::endl;
  //  root_view.Refine<0>();
  return 0;
}
