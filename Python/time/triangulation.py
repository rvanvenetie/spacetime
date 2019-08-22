from interval import Interval
from fractions import Fraction
from collections import Iterable

class Triangulation:
    """ Represents a locally refined triangulation of the interval [0,1]. """

    class Element:
        """ Represents an interval part of a locally refined triangulation.

        Represents the interval [2^(-l)*node_index, 2^(-l)*(node_index+1)].
        """
        def __init__(self, level, node_index, parent):
            self.level = level
            self.node_index = node_index
            self.parent = parent
            self.children = []

        @property
        def interval(self):
            h = Fraction(1, 2**self.level)
            return Interval(h * self.node_index, h * (self.node_index + 1))

        def is_leaf(self):
            return not len(self.children)

        @property
        def key(self):
            return (self.level, self.node_index)

    def __init__(self, element_class=Element):
        """
        Initializes the triangulation object.

        In case you would like to use another element class, you can pass
        it as an argument.
        """
        self.element_class = element_class

        # Store all elements inside a hashmap
        self.elements = {}

        # Also store (references) on a per level basis
        self.per_level = [[] for _ in range(1024)]

        # Create the initial element of [0,1]
        self.mother_element = self.element_class(0, 0, None)
        self._add_element(self.mother_element)

    def _add_element(self, elem):
        assert elem.key not in self.elements

        self.elements[elem.key] = elem
        self.per_level[elem.level].append(elem)

    def get_element(self, key, ensure_existence=True):
        """ Retrieves the element indexed by `key`.

        Retrieves the element object associated with interval:
            [2^-l * node_index, 2^-l * (node_index + 1)].

        If key is an iterable, it will return all the corresponding elements.
        If ensure_existence is set, this will create all the parents
        necessary for this interval to exist.

        TODO: This function should be removed eventually!
        """
        if isinstance(key, list):
            return [self.get_element(k, ensure_existence) for k in key]
        if key in self.elements: return self.elements[key]
        if not ensure_existence: assert False

        l, n = key
        assert l >= 0 and 0 <= n < 2**l

        parent = self.get_element((l-1, n // 2), True)
        self.bisect(parent)
        return self.elements[key]

    def children(self, elem):
        """ Returns the children associated to `elem`.

        If elem is an iterable, it will return a iterable to all of its children.
        """
        if isinstance(elem, Iterable):
            return [child for e in elem for child in self.children(e)]
        return elem.children


    def bisect(self, elem):
        """ Bisects `elem`.

        If this element was already bisected, then this function has no effect.
        """
        if not elem.is_leaf(): return
        child_left = self.element_class(elem.level + 1, elem.node_index * 2, elem)
        child_right = self.element_class(elem.level + 1, elem.node_index * 2 + 1, elem)
        self._add_element(child_left)
        self._add_element(child_right)
        elem.children = [child_left, child_right]

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)
