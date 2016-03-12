


def vl_binsum(accumulator, values, indexes, dim=-1):
    '''
    accumulator = vl_binsum(accumulator, values, indedex)
    adds the elements of the array values to the elements of the array
    accumulator indexed by indexes. vlaues and indexes must have the
    same dimensions, and the elements of indexes must be valid indexes
    for the array accumulator.
    an application is the caculation of a histogram accumulator, where indexed
    are the bin occurences and values are the occurence weights

    vl_binsum(..., DIM) operates only along the specified dimension DIM.
    In this case, accumulator, values and indexes are array of the same
    dimensions, except for the dimension DIM of ACCUMULATOR, which may differ
    and indexes is an array of subscropts of the DIM-th dimension of 
    accumulator. A typical application is the calculation of multiple 
    histograms. where each histogram is a 1-dimensional slice of the array
    ACCUMULATOR along the dimension DIM


    Example:
      vl_binsum([0,0], 1, 2) = [0,1]
      vl_binsum([1,7],-1, 1) = [0,7]
      vl_binsum(eye(3), [1,1,1], [1,2,3], 1) = 2*eye(3)
    
    '''
    
    for i in indexes:
        accumulator[i] = accumulator[i] + values
    
    return accumulator 
