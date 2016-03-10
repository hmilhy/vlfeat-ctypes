


def vl_binsum(accumulator, values, indexes,dim=0):

    for i_s in indexes:
        for i in i_s:
            accumulator[i] = accumulator[i] + 1
    
    return accumulator 
