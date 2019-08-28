from fractions import Fraction

from .interval import Interval
from .triangulation import Triangulation


def test_triangulation_bisect():
    triang = Triangulation()
    assert triang.mother_element.interval == Interval(0, 1)

    triang.bisect(triang.mother_element)
    assert len(triang) == 3
    assert (1, 0) in triang.elements and (1, 1) in triang.elements

    triang.bisect(triang.get_element((1, 1)))
    assert (2, 2) in triang.elements and (2, 3) in triang.elements

    assert triang.get_element((2, 3)).interval == Interval(Fraction(3, 4), 1)


def test_triangulation_get_element():
    triang = Triangulation()

    elem = triang.get_element((2, 3), ensure_existence=True)
    assert set(triang) == {(0, 0), (1, 0), (1, 1), (2, 2), (2, 3)}

    elem = triang.get_element((2, 1), ensure_existence=True)
    assert set(triang) == {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2),
                           (2, 3)}
