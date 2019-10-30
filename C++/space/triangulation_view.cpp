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

  std::cout << root_view->children().size() << std::endl;
  root_view->Refine();

  std::cout << root_view->children().size() << std::endl;

  auto bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine();
  }

  std::cout << " ---- after uniform refinemenet --- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine();
  }

  std::cout << " ---- after uniform refinemenet --- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
    nv->Refine();
  }

  auto bla_view =
      std::make_shared<datastructures::NodeView<Vertex>>(T.vertex_meta_root);
  bla_view->DeepRefine();
  root_view->Union(bla_view);

  std::cout << " --- after union with a deep refine -- " << std::endl;
  bfs = root_view->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
  }

  auto root_cpy = root_view->DeepCopy();
  std::cout << " --- copied -- " << std::endl;
  bfs = root_cpy->Bfs();
  for (auto nv : bfs) {
    std::cout << *nv << std::endl;
  }

  //
  //  int x = child_view->level();
  //
  //  std::cout << x << std::endl;
  //  root_view.Refine();
  return 0;
}
