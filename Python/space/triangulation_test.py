import numpy as np

from triangulation import Triangulation


def test_on_domain_bdr():
    triangulation = Triangulation.unit_square()
    assert all([v.on_domain_boundary for v in triangulation.vertices])
    triangulation.refine(triangulation.elements[0])
    for _ in range(100):
        triangulation.refine(triangulation.elements[np.random.randint(
            len(triangulation.vertices))])
    for vert in triangulation.vertices:
        assert vert.on_domain_boundary == (vert.x == 0 or vert.x == 1
                                           or vert.y == 0 or vert.y == 1)
