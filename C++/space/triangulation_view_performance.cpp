
#include <cmath>
#include <map>
#include <random>
#include <set>

#include "datastructures/multi_tree_view.hpp"
#include "space/triangulation.hpp"
#include "space/triangulation_view.hpp"

class mRND {
 public:
  void seed(unsigned int s) { _seed = s; }

 protected:
  mRND() : _seed(0), _a(0), _c(0), _m(2147483648) {}
  int rnd() { return (_seed = (_a * _seed + _c) % _m); }

  int _a, _c;
  unsigned int _m, _seed;
};
class BSD_RND : public mRND {
 public:
  BSD_RND() {
    _a = 1103515245;
    _c = 12345;
  }
  int rnd() { return mRND::rnd(); }
};

using namespace space;
using namespace datastructures;

constexpr int level = 10;
constexpr int iters = 10000;

int main() {
  BSD_RND bsd_rnd;

  auto T = InitialTriangulation::UnitSquare();
  T.elem_meta_root->UniformRefine(level);

  for (size_t i = 0; i < iters; ++i) {
    // Create a random subtree
    auto vertex_subtree = NodeView<Vertex>::CreateRoot(T.vertex_meta_root);
    vertex_subtree->DeepRefine(
        /* call_filter */ [&bsd_rnd](auto &&vertex) {
          return vertex->level() <= 0 || bsd_rnd.rnd() % 3 != 0;
        });

    auto T_view = TriangulationView(vertex_subtree);
    // std::cout << T_view.history_.size() << std::endl;
  }
}
