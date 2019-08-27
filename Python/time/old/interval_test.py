from fractions import Fraction

from .interval import Interval, IntervalSet


def F(x, y):
    return Fraction(x, y)


def test_interval_intersect():
    assert not Interval(0, F(1, 2)).intersects(Interval(F(3, 4), 1))
    assert not Interval(F(3, 4), 1).intersects(Interval(0, F(1, 2)))
    assert not Interval(F(1, 2), 1).intersects(
        Interval(0, F(1, 2)), closed=False)
    assert Interval(F(1, 2), 1).intersects(Interval(0, F(1, 2)), closed=True)
    assert not Interval(0, F(1, 4)).intersects(Interval(F(1, 2), 1))


def test_interval_merge():
    ivs = IntervalSet([
        Interval(0, 1),
        Interval(0, F(1, 2)),
        Interval(0, 1),
        Interval(0, F(1, 2))
    ])
    assert ivs.covers(Interval(0, 1))
