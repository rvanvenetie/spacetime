from fractions import Fraction


class Interval(object):
    """ Represents an open or closed interval (a,b). """
    __slots__ = ('a','b')

    def __init__(self, a, b):
        assert isinstance(a, int) or isinstance(a, Fraction)
        assert isinstance(b, int) or isinstance(b, Fraction)
        self.a = Fraction(a)
        self.b = Fraction(b)

    @property
    def mid(self):
        return (self.a + self.b) / 2

    def intersects(self, interval, closed=False):
        if not closed:
            if interval.a >= self.b or self.a >= interval.b: return False
            return True
        else:
            if interval.a > self.b or self.a > interval.b: return False
            return True

    def intersection(self, interval):
        if interval.a >= self.b or self.a >= interval.b: return None
        return Interval(max(self.a, interval.a), min(self.b, interval.b))

    def contains(self, interval):
        return self.a <= interval.a and interval.b <= self.b

    def __repr__(self):
        return r"Interval(%s,%s)" % (self.a, self.b)


class IntervalSet(object):
    """ Represents a collection of intervals with fast merging of overlap.

    """

    def __init__(self, intervals):
        """ This runs in O(N log N) time.

        Current complexity: O(N log N) because of a sort step
        Goal complexity: O(N)
        TODO: if we assume to receive a list of intervals that is already sorted
        in order of increasing left-hand side, then this sort step is
        unnecessary. If we assume to get a number of these interval-lists, where
        every list itself is already sorted, we can do a merge-sort-like thing
        to sort the entire list in linear time. On the other hand: sorting in
        Python is highly optimized and it may actually be faster to just use
        the sort option.
        """
        sorted_intervals = sorted(intervals, key=lambda interval: interval.a)
        stack = []
        if len(sorted_intervals):
            stack.append(Interval(sorted_intervals[0].a, sorted_intervals[0].b))
            for interval in sorted_intervals:
                if interval.intersects(stack[-1], closed=True):
                    stack[-1] = Interval(stack[-1].a,
                                         max(stack[-1].b, interval.b))
                else:
                    stack.append(Interval(interval.a, interval.b))
            self.intervals = stack
        else:
            self.intervals = []

    def __iter__(self):
        return iter(self.intervals)

    def covers(self, interval):
        """ Test if the interval set contains the given interval completely."""
        return any([iv.contains(interval) for iv in self.intervals])

    def intersects(self, interval):
        return any([iv.intersects(interval) for iv in self.intervals])
