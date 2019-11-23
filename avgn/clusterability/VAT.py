""" This code is modified from pyclustertend
https://github.com/lachhebo/pyclustertend/blob/master/LICENSE
"""
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


def ordered_dissimilarity_matrix(X):
    """The ordered dissimilarity matrix is used by visual assesement of tendency. It is a just a a reordering 
    of the dissimilarity matrix.
    Parameters
    ----------
    X : matrix
        numpy array
    Return
    -------
    ODM : matrix
        the ordered dissimalarity matrix .
    """

    # Step 1 :

    I = []

    R = pairwise_distances(X)
    P = np.zeros(R.shape[0], dtype="int")

    argmax = np.argmax(R)

    j = argmax % R.shape[1]
    i = argmax // R.shape[1]

    P[0] = i
    I.append(i)

    K = np.linspace(0, R.shape[0] - 1, R.shape[0], dtype="int")
    J = np.delete(K, i)

    # Step 2 :
    # for each row

    total_ticks = np.sum(
        [i * j for i, j in zip(range(1, R.shape[0] + 1), range(R.shape[0])[::-1])]
    )
    pbar = tqdm(total=total_ticks, desc="candidates")
    for r in tqdm(range(1, R.shape[0]), desc="row"):

        p, q = (-1, -1)

        mini = np.max(R)

        for candidate_p in I:
            for candidate_j in J:
                if R[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = R[p, q]

            pbar.update(len(J))
        P[r] = q
        I.append(q)

        ind_q = np.where(np.array(J) == q)[0][0]
        J = np.delete(J, ind_q)

    # Step 3

    ODM = np.zeros(R.shape)

    for i in range(ODM.shape[0]):
        for j in range(ODM.shape[1]):
            ODM[i, j] = R[P[i], P[j]]

    # Step 4 :

    return ODM, P


def ivat_ordered_dissimilarity_matrix(D):
    """The ordered dissimilarity matrix is used by ivat. It is a just a a reordering 
    of the dissimilarity matrix.
    Parameters
    ----------
    X : matrix
        numpy array
    Return
    -------
    D_prim : matrix
        the ordered dissimalarity matrix .
    """

    D_prim = np.zeros((D.shape[0], D.shape[0]))

    for r in range(1, D.shape[0]):
        # Step 1 : find j for which D[r,j] is minimum and j in [1:r-1]

        j = np.argmin(D[r, 0:r])

        # Step 2 :

        D_prim[r, j] = D[r, j]

        # Step 3 : pour c : 1,r-1 avec c !=j
        c_tab = np.array(range(0, r))
        c_tab = c_tab[c_tab != j]

        for c in c_tab:
            D_prim[r, c] = max(D[r, j], D_prim[j, c])
            D_prim[c, r] = D_prim[r, c]

    return D_prim
