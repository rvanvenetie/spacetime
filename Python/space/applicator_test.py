import random

import numpy as np

from ..datastructures.tree_vector import MetaRootVector
from .applicator import Applicator
from .operators import MassOperator, StiffnessOperator
from .triangulation import InitialTriangulation
from .triangulation_view import TriangulationView


def test_operators():
    T = InitialTriangulation.unit_square()
    T.elem_meta_root.uniform_refine(8)
    for op in [MassOperator(), StiffnessOperator()]:
        applicator = Applicator(op)
        for ml in range(5):
            # Create some (sub)vector on the vertices.
            vec_in = MetaRootVector(T.vertex_meta_root)
            vec_in.uniform_refine(ml)
            assert len(vec_in.bfs()) < len(T.vertex_meta_root.bfs())
            for vertex in vec_in.bfs():
                vertex.value = random.random()

            vec_out = MetaRootVector(T.vertex_meta_root)
            vec_out.uniform_refine(max_level=ml)

            applicator.apply(vec_in, vec_out)

            # Compare.
            T_view = TriangulationView(vec_in)
            assert vec_in.bfs() == T_view.vertices
            vec_out_np = op.apply(vec_in.to_array())
            assert np.allclose(vec_out.to_array(), vec_out_np)
