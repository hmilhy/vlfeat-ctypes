# from dsift import vl_dsift
# from imsmooth import vl_imsmooth
# from phow import vl_phow

# Not doing actual imports here until the functions are called, because
# otherwise "python -m vlfeat.download" doesn't work (since it first loads
# this __init__, which loads things that nead the actual lib, which crashes...).
#
# TODO: do lazy-import in a way that doesn't break docstrings and such

def vl_dsift(*args, **kwargs):
    from .dsift import vl_dsift as f
    return f(*args, **kwargs)

def vl_imsmooth(*args, **kwargs):
    from .imsmooth import vl_imsmooth as f
    return f(*args, **kwargs)

def vl_phow(*args, **kwargs):
    from .phow import vl_phow as f
    return f(*args, **kwargs)

###########################################################
# kmeans
def vl_kmeans(*args, **kwargs):
    from kmeans.kmeans import vl_kmeans as f
    return f(*args, **kwargs)

def vl_ikmeans(*args, **kwargs):
    from kmeans.ikmeans import vl_ikmeans as f
    return f(*args, **kwargs)

def vl_ikmeanspush(*args, **kwargs):
    from kmeans.ikmeanspush import vl_ikmeanspush as f
    return f(*args,**kwargs)

def vl_ikmeanshist(*args, **kwargs):
    from kmeans.ikmeanshist import vl_ikmeanshist as f
    return f(*args, **kwargs)

def vl_hikmeans(*args, **kwargs):
    from kmeans.hikmeans import vl_hikmeans as f
    return f(*args, **kwargs)

def vl_hikmeanspush(*args, **kwargs):
    from kmeans.hikmeanspush import vl_hikmeanspush as f
    return f(*args, **kwargs)

def vl_hikmeanshist(*args, **kwargs):
    from kmeans.hikmeanshist import vl_hikmeanshist as f
    return f(*args, **kwargs)

###########################################################
#
