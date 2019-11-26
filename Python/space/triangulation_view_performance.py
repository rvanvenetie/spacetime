from ..datastructures.tree_view import TreeView
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView


def bsd_rand(seed):
    def rand():
        nonlocal seed
        seed = (1103515245 * seed + 12345) & 0x7fffffff
        return seed

    return rand


def test_python(level, iters):
    bsd_rnd = bsd_rand(0)
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(level)

    for _ in range(iters):
        vertex_subtree = TreeView.from_metaroot(T.vertex_meta_root)
        vertex_subtree.deep_refine(call_filter=lambda vertex: vertex.level <= 0
                                   or (bsd_rnd() % 3) != 0)
        T_view = TriangulationView(vertex_subtree)


def test_cppyy(level, iters):
    import cppyy
    cppyy.cppdef('#pragma cling optimize 3')
    cppyy.include('C++/space/triangulation_view.hpp')
    cppyy.cppdef('extern template datastructures::NodeView<space::Vertex>;')
    cppyy.load_library('C++/build/space/libtriangulation.dylib')
    cppyy.load_library('C++/build/space/libtriangulation_view_lib.dylib')
    bsd_rnd = bsd_rand(0)
    T = cppyy.gbl.space.InitialTriangulation.UnitSquare()
    T.elem_meta_root.UniformRefine(level)

    for _ in range(iters):
        vertex_subtree = cppyy.gbl.datastructures.NodeView[
            cppyy.gbl.space.Vertex].CreateRoot(T.vertex_meta_root)
        vertex_subtree.DeepRefine()
        T_view = cppyy.gbl.space.TriangulationView(vertex_subtree)


if __name__ == "__main__":
    test_python(level=10, iters=500)
    #test_cppyy(level=10, iters=10000)
