from interval import Interval
from fractions import Fraction
from collections import Iterable

class Element:
    """ Represents an interval part of a locally refined triangulation.

    Represents the interval [2^(-l)*node_index, 2^(-l)*(node_index+1)].
    """
    def __init__(self, level, node_index, parent):
        self.level = level
        self.node_index = node_index
        self.parent = parent
        self.children = []
        h = Fraction(1, 2**level)
        self.interval = Interval(h * node_index, h * (node_index + 1))

        # TODO: this should be moved to a different class
        self.Lambda_in = []
        self.Lambda_out = []
        self.Pi_in = []
        self.Pi_out = []

    def is_leaf(self):
        return not len(self.children)

    @property
    def key(self):
        return (self.level, self.node_index)

class Triangulation:
    """ Represents a locally refined triangulation of the interval [0,1]. """

    def __init__(self):
        # Store all elements inside a hashmap
        self.elements = {}

        # Also store (references) on a per level basis
        self.per_level = [[] for _ in range(1024)]

        # Create the initial element of [0,1]
        self.mother_element = Element(0, 0, None)
        self._add_element(self.mother_element)

    def _add_element(self, elem):
        assert elem.key not in self.elements

        self.elements[elem.key] = elem
        self.per_level[elem.level].append(elem)

    def get_element(self, key, ensure_existence=True):
        """ Retrieves the element indexed by `key`.

        Retrieves the element object associated with interval:
            [2^-l * node_index, 2^-l * (node_index + 1)].

        If ensure_existence is set, this will create all the parents
        necessary for this interval to exist.

        TODO: This function should be removed eventually!
        """
        if key in self.elements: return self.elements[key]
        if not ensure_existence: assert False

        l, n = key
        assert l >= 0 and 0 <= n < 2**l

        parent = self.get_element((l-1, n // 2), True)
        self.bisect(parent)
        return self.elements[key]

    def children(self, elem, ensure_existence=True):
        """ Returns the children associated to `elem`.

        If elem is an iterable, it will return a iterable to all of its children.

        If ensure_existence is set, this will bisect the given element if it
        doesn't have children yet.
        """
        if isinstance(elem, Iterable):
            return (child for e in elem for child in self.children(e, ensure_existence))

        if elem.children: return elem.children
        if not ensure_existence: assert False
        self.bisect(elem)
        return elem.children

    def bisect(self, elem):
        """ Bisects `elem`.

        If this element was already bisected, then this function has no effect.
        """
        if not elem.is_leaf(): return
        child_left = Element(elem.level + 1, elem.node_index * 2, elem)
        child_right = Element(elem.level + 1, elem.node_index * 2 + 1, elem)
        self._add_element(child_left)
        self._add_element(child_right)
        elem.children = [child_left, child_right]

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)
