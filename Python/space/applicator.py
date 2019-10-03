from .triangulation_view import TriangulationView


class Applicator:
    """ Class that can apply operators on the hierarchical basis. """
    def __init__(self, singlescale_operator):
        """ Initialize the applicator.

        Arguments:
            singlescale_operator: an instance of operators.Operator.
        """
        self.operator = singlescale_operator

    def apply(self, vec_in, vec_out):
        """ Apply the multiscale operator.  """
        assert len(vec_in.bfs()) == len(vec_out.bfs())
        assert all(n1.node is n2.node
                   for n1, n2 in zip(vec_in.bfs(), vec_out.bfs()))

        self.operator.triang = TriangulationView(vec_in)

        np_vec_in = vec_in.to_array()
        np_vec_out = self.operator.apply(np_vec_in)
        vec_out.from_array(np_vec_out)

        return vec_out
