import numpy as np
import numpy.random as rnd


def nail_on_a_wall(dim, padding, hasNail):
    """
    :param dim: a tuple indicates the x, y dimension of the generated matrix
    :param padding: an integer indicates the padding outside the wall
    :return: a 2D numpy array of dimension according to dim
    """
    assert 2*padding[0] + 7 < dim[0]
    assert 2*padding[1] + 7 < dim[1]
    puzzle = np.zeros(dim)
    puzzle[padding:-padding - 1][padding:-padding - 1] = 1
    puzzle[padding + 1:-padding - 2][padding + 1:-padding - 2] = 0
    if hasNail:
        pass

    return puzzle, hasNail