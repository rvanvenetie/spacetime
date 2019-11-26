from ..datastructures.tree_view import TreeView
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView

seed = 0


def bsd_rnd():
    global seed
    seed = (1103515245 * seed + 12345) & 0x7fffffff
    return seed


def test_python(level, iters):
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(level)

    for _ in range(iters):
        vertex_subtree = TreeView.from_metaroot(T.vertex_meta_root)
        vertex_subtree.deep_refine(call_filter=lambda vertex: vertex.level <= 0
                                   or (bsd_rnd() % 3) != 0)
        TriangulationView(vertex_subtree)


def test_cppyy(level, iters):
    import cppyy
    cppyy.cppdef('#pragma cling optimize 3')
    cppyy.include('C++/space/triangulation_view.hpp')
    cppyy.cppdef('extern template datastructures::NodeView<space::Vertex>;')
    cppyy.load_library('C++/build/space/libtriangulation.dylib')
    cppyy.load_library('C++/build/space/libtriangulation_view_lib.dylib')
    T = cppyy.gbl.space.InitialTriangulation.UnitSquare()
    T.elem_meta_root.UniformRefine(level)

    for _ in range(iters):
        vertex_subtree = cppyy.gbl.datastructures.NodeView[
            cppyy.gbl.space.Vertex].CreateRoot(T.vertex_meta_root)
        vertex_subtree.DeepRefine()
        cppyy.gbl.space.TriangulationView(vertex_subtree)


if __name__ == "__main__":
    test_python(level=10, iters=1000)
    #test_cppyy(level=10, iters=10000)
