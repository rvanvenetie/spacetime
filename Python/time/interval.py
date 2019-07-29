class Interval(object):
    """ Represents an open or closed interval (a,b). """

    def __init__(self, a, b=None):
        if not b:
            # Copy constructor.
            assert isinstance(a, Interval)
            self.a = a.a
            self.b = a.b
            self.mid = a.mid
        else:
            assert a < b
            self.a = a
            self.b = b
            self.mid = (a + b) / 2

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
    """ Represents a collection of intervals with fast merging of overlap. """

    def __init__(self, intervals):
        """ This runs in O(#intervals) time. """
        sorted_intervals = sorted(intervals, key=lambda interval: interval.a)
        stack = []
        if len(sorted_intervals):
            stack.append(Interval(sorted_intervals[0]))
            for interval in sorted_intervals:
                if interval.intersects(stack[-1], closed=True):
                    stack[-1].b = max(stack[-1].b, interval.b)
                else:
                    stack.append(Interval(interval))
                    stack[-1].mid = (stack[-1].a + stack[-1].b) / 2
            self.intervals = stack
        else:
            self.intervals = []

    def covers(self, interval):
        """ Test if the interval set contains the given interval completely."""
        return any([iv.contains(interval) for iv in self.intervals])

    def intersects(self, interval):
        return any([iv.intersects(interval) for iv in self.intervals])
