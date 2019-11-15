from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform, normal
import numpy as np
from math import isnan
import pandas as pd
from scipy.spatial import Delaunay


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def sample_from_hull(hull, xmin, xmax, dims):

    in_hull_bool = False

    for _ in range(10000):
        sample = uniform(xmin, xmax, dims).reshape(1, -1)
        if in_hull(sample, hull):
            in_hull_bool = True
            break

    if in_hull_bool == False:
        raise ValueError("Sample from in hull failed")
    return sample


def hopkins_statistic(
    X, m_prop_n=0.1, n_neighbors=1, distribution="uniform", flip=False
):
    """ Computes hopkins statistic over a distribution X
    based upon:
    - https://matevzkunaver.wordpress.com/2017/06/20/hopkins-test-for-cluster-tendency/
    - https://github.com/rflachlan/Luscinia/wiki/Hopkins-statistic
    - https://en.wikipedia.org/wiki/Hopkins_statistic
    - https://pypi.org/project/pyclustertend/
    - 
    X: The original dataset
    m: number of samples in Y
    n: number of samples in X
    dims: number of dimensions in x
    nbrs: nearest neighbor for each value of X
    ujd: the distance of y_i from its nearest neighbor in X
    wjd: the distance of x_i from its nearest neighbor in X
    
    """
    # convert to pandas dataframe
    if type(X) == np.ndarray:
        X = pd.DataFrame(X)

    # print('X shape: {}'.format(X.shape))

    # the nearest neigbor(s) for each element in X
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    # statistics over X (for sampling)
    xmin = np.min(X, axis=0)
    xmax = np.max(X, axis=0)
    loc = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    # print(xmin, xmax, loc, scale)
    # number of dimensions
    dims = X.shape[1]
    # the number of examples
    n = len(X)

    # choose how many samples over X
    m = int(np.ceil(m_prop_n * n))

    # sample from X m times
    rand_X = sample(range(0, n, 1), m)
    # print("rand_X", rand_X)

    if distribution == "uniform_convex_hull":
        hull = Delaunay(X)

    def sample_dist():
        if distribution == "uniform":
            # print(np.shape(uniform(xmin, xmax, dims).reshape(1, -1)))
            return uniform(xmin, xmax, dims).reshape(1, -1)
        elif distribution == "normal":
            return normal(loc, scale, dims).reshape(1, -1)
        elif distribution == "uniform_convex_hull":
            return sample_from_hull(hull, xmin, xmax, dims)
        else:
            raise ValueError(
                'distribution must be "uniform", "uniform_convex_hull", or "normal"'
            )

    ujd = []  # distance of y_i from its nearest neighbor in X
    wjd = []  # the distance of x_i from its nearest neighbor in X

    # for each sample from X
    for j in range(0, m):
        # sample Y (since its with replacement, repeat each time)

        Y = sample_dist()
        # get distance from Y to nearest neighbors in X
        u_dist, _ = nbrs.kneighbors(Y, n_neighbors + 1, return_distance=True)
        ujd.append(np.mean(u_dist[0][1:]))

        # get the distance from X sample to the nearest neighbors in X
        w_dist, _ = nbrs.kneighbors(
            X.iloc[rand_X[j]].values.reshape(1, -1),
            n_neighbors + 1,
            return_distance=True,
        )
        wjd.append(np.mean(w_dist[0][1:]))

    # distances of sampled / distances of sampled + distances of true distribution
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if flip:
        H = sum(wjd) / (sum(ujd) + sum(wjd))

    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H
