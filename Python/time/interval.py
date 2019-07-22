class Interval(object):
    """ Represents an open or closed interval (a,b). """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def intersects(self, interval, closed=False):
        if not closed:
            return self.b < interval.a or self.a < interval.b
        else:
            return self.b <= interval.a or self.a <= interval.b

    def contains(self, interval):
        return self.a <= interval.a and interval.b <= self.b

    def intersection(self, interval):
        return Interval(max(self.a, interval.a), min(self.b, interval.b))


class IntervalSet(object):
    """ Represents a collection of intervals with fast merging of overlap. """

    def __init__(self, intervals):
        """ This runs in O(#intervals) time. """
        sorted_intervals = sorted(intervals, key=lambda interval: interval.a)
        stack = []
        if len(sorted_intervals):
            stack.append(sorted_intervals[0])
            for interval in sorted_intervals:
                if interval.intersects(stack[-1], closed=True):
                    stack[-1].b = max(stack[-1].b, interval.b)
                else:
                    stack.append(interval)
            self.intervals = stack
        else:
            self.intervals = []

    def covers(self, interval):
        """ Test if the interval set contains the given interval completely."""
        return any([iv.contains(interval) for iv in self.intervals])
