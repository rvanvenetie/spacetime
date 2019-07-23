from interval import Interval, IntervalSet


def test_interval_intersect():
    assert not Interval(0, 0.5).intersects(Interval(0.75, 1.0))
    assert not Interval(0.75, 1.0).intersects(Interval(0, 0.5))


def test_interval_merge():
    ivs = IntervalSet(
        [Interval(0, 1),
         Interval(0, 0.5),
         Interval(0, 1),
         Interval(0, 0.5)])
    assert ivs.covers(Interval(0, 1))
