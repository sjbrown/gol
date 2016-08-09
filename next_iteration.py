#! /usr/bin/env python
# encoding: utf-8

import numpy
import doctest
from rules_cache import rules

def zeros_and_ones_to_int(group):
    """
    >>> zeros_and_ones_to_int([0,0,0,0,0,0])
    0
    >>> zeros_and_ones_to_int([1,0,1,0,0,0])
    5
    >>> zeros_and_ones_to_int([1,1,1,1,1,1])
    63
    """
    return sum((x*2**i for (i,x) in enumerate(group)))

def surrounding_3x3_group(M, row_index, col_index):
    """
    Given the index into the array "M",
    return the 3x3 group of cells centered on that index

    >>> a = numpy.zeros([5,20])
    >>> a[0,0] = 1
    >>> a[4,0] = 5
    >>> surrounding_3x3_group(a,0,0)
    array([[ 0.,  5.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  0.]])
    """
    rows = [row_index-1, row_index, row_index+1]
    cols = [col_index-1, col_index, col_index+1]
    return M.take(rows, mode='wrap', axis=0).take(cols, mode='wrap', axis=1)

def surrounding_3x3_group_wrap(M, row_index, col_index):
    result = surrounding_3x3_group(M, row_index, col_index)
    return result.flatten()

def surrounding_3x3_group_nowrap(M, row_index, col_index):
    """
    If invoked in "nowrap" mode, consider anything beyond the boundaries
    to always be dead cells.

    >>> a = numpy.zeros([5,20])
    >>> a[0,0] = 1
    >>> a[4,0] = 5
    >>> surrounding_3x3_group_nowrap(a,0,0)
    array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.])
    """
    result = surrounding_3x3_group(M, row_index, col_index)
    if row_index < 1:
        result[0] = [0,0,0]
    elif row_index > M.shape[0]-1:
        result[-1] = [0,0,0]
    if col_index < 1:
        result[:,0] = [0,0,0]
    elif col_index > M.shape[1]-1:
        result[:,-1] = [0,0,0]
    return result.flatten()

def visit_each(input_matrix, grouper_fn):
    M = numpy.array(input_matrix)

    if M.shape[0] < 3 or M.shape[1] < 3:
        raise Exception('Matrix must be at least (3,3) in shape')

    result_matrix = numpy.zeros(M.shape)

    m_iter = numpy.nditer(M, flags=['multi_index'])

    for item in m_iter:
        i, j = m_iter.multi_index
        group = grouper_fn(M, i, j)
        g_int = zeros_and_ones_to_int(group)
        result_matrix[i,j] = rules[g_int]

    return result_matrix

def calc_nowrap(input_matrix):
    return visit_each(input_matrix, surrounding_3x3_group_nowrap)

def calc_wrap(input_matrix):
    return visit_each(input_matrix, surrounding_3x3_group_wrap)


def test():
    def matrix_equal(a,b):
        # make it jive with numpy
        return (a == b).all()

    test_matrix = [
     [0,0,0,0],
     [0,1,1,0],
     [0,0,1,0],
    ]

    print calc_wrap(test_matrix)
    assert matrix_equal(calc_wrap(test_matrix), [
    [0,  1,  1,  0],
    [0,  1,  1,  0],
    [0,  1,  1,  0]
    ])


    glider_matrix = numpy.array([
     [0,0,0,0,0,0],
     [0,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,1,0,1,0,0],
     [0,0,1,1,0,0],
     [0,0,0,0,0,0],
     [0,0,0,0,0,0],
    ])
    print numpy.array(glider_matrix)


    print calc_wrap(glider_matrix)
    assert matrix_equal(calc_wrap(glider_matrix), [
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  1,  0,  0,  0],
    [0,  0,  0,  1,  1,  0],
    [0,  0,  1,  1,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    ])

    print calc_wrap(calc_wrap(glider_matrix))
    assert matrix_equal(calc_wrap(calc_wrap(glider_matrix)), [
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  1,  0,  0],
    [0,  0,  0,  0,  1,  0],
    [0,  0,  1,  1,  1,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    ])

    print calc_wrap(calc_wrap(calc_wrap(glider_matrix)))
    assert matrix_equal(calc_wrap(calc_wrap(calc_wrap(glider_matrix))), [
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  1,  0,  1,  0],
    [0,  0,  0,  1,  1,  0],
    [0,  0,  0,  1,  0,  0],
    [0,  0,  0,  0,  0,  0],
    ])

    print calc_wrap(calc_wrap(calc_wrap(calc_wrap(glider_matrix))))
    assert matrix_equal(calc_wrap(calc_wrap(calc_wrap(calc_wrap(glider_matrix)))), [
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  0,  0],
    [0,  0,  0,  0,  1,  0],
    [0,  0,  1,  0,  1,  0],
    [0,  0,  0,  1,  1,  0],
    [0,  0,  0,  0,  0,  0],
    ])

if __name__ == '__main__':
    doctest.testmod()
    test()
