import numpy as np
from sklearn.neighbors import kneighbors_graph


def theta_mle(X, M):
    """Use reconstructed data to estimate the theta parameters used by the prediction
    algorithm.

    Args:
        X: Sparse data matrix
        M: Completed data matrix

    Returns:
        Theta estimate (float)
    """

    mask = (X != 0).astype(float)
    return np.sum(mask) / (2 * np.square(np.linalg.norm(mask * (X - M))))


def finite_difference_matrix(T):
    "Construct a (T x T) forward difference matrix"

    return np.diag(np.pad(-np.ones(T - 1), (0, 1), "constant")) + np.diag(
        np.ones(T - 1), 1
    )


def laplacian_kernel_matrix(T, gamma=1.0):
    "Construct a (T x T) matrix for convolutional regularization"

    def kernel(x):
        return np.exp(-1.0 * gamma * np.abs(x))

    return np.array(
        [kernel(np.arange(T) - i) for i in np.arange(T)]
    )


def basis_baseline_value(shape, min_value):
    "Shifting the basic profiles so that min(V) >= min_value."
    return np.ones(shape) * min_value


def weighted_adjacency_matrix(X, k=5):
    "Build a k-NN similarity graph based on the rows in data matrix X."
    # Create the distance graph
    D = kneighbors_graph(
        X,
        n_neighbors=k,
        mode="distance",     # Distance between neighbours
        metric="minkowski",  
        p=2,                 # Euclidean distance
        include_self=False
    )
    D = D.tocoo()

    # Apply similarity kernel
    sigma = np.median(D.data)
    D.data = np.exp(-(D.data**2)/(2*sigma**2))
    S = D.tocsr()
    S = S.maximum(S.T)
    # Larger S_ij indicates stronger connection
    return S.todense()


def chebyshev_polynomials(L, order):
    "Chebyshev polynomials for Tensorflow tensors."
    terms = []
    for k in range(order):
        if k == 0:
            terms.append(np.eye(L.shape[0], dtype=np.float32))
        elif k == 1:
            terms.append(L)
        else:
            terms.append(2 * np.matmul(L, terms[k-1]) - terms[k-2])
    return terms
