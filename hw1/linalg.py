import numpy as np
from numpy import linalg as LA
import math

def dot_product(a, b):
    sum=0
    if a.shape[1]!=b.shape[0]:
        return None
    else:
        # for i in range(a.shape[0]):
        #     for x in range(a.shape[1]):
        #         for l in range(b.shape[1]):
        #             sum+=a[i][x]*b[x][l]

        out=np.dot(a,b)
    return out

def complicated_matrix_function(M, c, b):

    a=np.array(c)
#dimension of each element should be >1
    if M.ndim <=1:
        M=1
    if a.ndim <=1:
        a=1
    if b.ndim <=1:
        b=1
    ab=dot_product(a,b)

    if ab.shape[0]==1 and ab.shape[1]==1:
        mt=dot_product(M,a.transpose())
        if mt.ndim <=1:
            return None
        sum=mt*(int(ab))
    else:
        sum=dot_product(ab,mt)
    
    out=sum
    return out


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, m)

    Returns:
        w: numpy array of shape (m,) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w=None
    v = None
    ### YOUR CODE HERE
    pass
    w,v = np.linalg.eig(M)
    ### END YOUR CODE
    return w, v


def euclidean_distance_native(u, v):
    """Computes the Euclidean distance between two vectors, represented as Python
    lists.

    Args:
        u (List[float]): A vector, represented as a list of floats.
        v (List[float]): A vector, represented as a list of floats.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    sum=0
    # First, run some checks:
    assert isinstance(u, list)
    assert isinstance(v, list)
    assert len(u) == len(v)

    for i in range(len(u)):
        diff=u[i]-v[i]
        sum+=pow(diff,2)
    # Compute the distance!
    # Notes:
    #  1) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    ### YOUR CODE HERE
    pass
    return math.sqrt(sum)


def euclidean_distance_numpy(u, v):
    """Computes the Euclidean distance between two vectors, represented as NumPy
    arrays.

    Args:
        u (np.ndarray): A vector, represented as a NumPy array.
        v (np.ndarray): A vector, represented as a NumPy array.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape

    dist=np.linalg.norm(u-v)
    return dist


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    ### YOUR CODE HERE
    eigenvalues,eigenvectors=eigen_decomp(M)
    fin=np.argsort(np.abs(eigenvalues))[::-1][:k]

    k_eigenvalues = eigenvalues[fin]
    k_eigenvectors = eigenvectors[:, fin]
    kk_eigenvectors=k_eigenvectors.transpose()

    return k_eigenvalues, kk_eigenvectors
