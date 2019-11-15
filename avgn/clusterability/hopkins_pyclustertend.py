from sklearn.neighbors import BallTree
import numpy as np
from random import sample
import pandas as pd
import matplotlib.pyplot as plt


def hopkins(D, sampling_size, flip=False):
    """Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express no clusterability and a score tending to 0 express a high cluster tendency.
    Parameters
    ----------
    D : numpy array 
        The input dataset
    sampling_size : int
        The sampling size which is used to evaluate the number of DataFrame.
    Returns
    ---------------------
    score : float
        The hopkins score of the dataset (between 0 and 1)
    Examples
    --------
    >>> from sklearn import datasets
    >>> from pyclustertend import hopkins
    >>> X = datasets.load_iris().data
    >>> hopkins(X,150)
    0.16
    """

    if type(D) == np.ndarray:
        D = pd.DataFrame(D)

    # Sample n observations from D : P

    if sampling_size > D.shape[0]:
        raise Exception(
            "The number of sample of sample is superieur than the shape of D"
        )

    P = D.sample(n=sampling_size)

    # Get the distance to their neirest neighbors in D : X

    tree = BallTree(D, leaf_size=2)
    dist, _ = tree.query(P, k=2)
    X = dist[:, 1]

    # Randomly simulate n points with the same variation as in D : Q.

    max_D = D.max()
    min_D = D.min()

    matrix = np.column_stack(
        (
            np.random.uniform(min_D[0], max_D[0], sampling_size),
            np.random.uniform(min_D[1], max_D[1], sampling_size),
        )
    )

    if len(max_D) >= 2:
        for i in range(2, len(max_D)):
            matrix = np.column_stack(
                (matrix, np.random.uniform(min_D[i], max_D[i], sampling_size))
            )
    Q = pd.DataFrame(matrix)

    # Get the distance to their neirest neighbors in D : Y

    tree = BallTree(D, leaf_size=2)
    dist, _ = tree.query(Q, k=1)
    Y = dist

    # return the hopkins score

    x = sum(X)
    y = sum(Y)

    if x + y == 0:
        raise Exception("The denominator of the hopkins statistics is null")
    if flip:
        return y / (x + y)[0]
    return x / (x + y)[0]

