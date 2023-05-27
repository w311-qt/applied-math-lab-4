import scipy.sparse as scs
import random
import numpy as np


def generate_diagonal_dominance_matrix(k):
    A_k = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            if i != j:
                A_k[i, j] = -1
            else:
                A_k[i, j] = k + 1
    return A_k


def generate_random_matrix(k):
    A = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            A[i, j] = random.randint(-10, 10)
    return A


def generate_vector(A):
    n = A.indptr.size
    x = []
    for i in range(1, n):
        x.append(i)
    return np.dot(A.toarray(), x)


def empty_matrix(n, m, format):
    Matrix = scs.__dict__[format + "_matrix"]
    return Matrix((n, m))


def identity_matrix(n, format):
    return scs.identity(n, format=format)


def generate_gilbert(k):
    A = scs.csr_matrix(([], [], [0] * (k + 1)), shape=(k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            A[i, j] = 1 / (i + j + 1)
    return A

def inverse_matrix(L, U):
    n = len(L.indptr) - 1
    e_matrix = identity_matrix(n, "csr")
    y = np.array([])
    result_matrix = [np.array([])]
    for k in range(0, n):
        e = e_matrix.getrow(k).toarray()[0]
        temp = np.array([])
        for i in range(0, n):
            sum = 0
            for p in range(0, i):
                sum += L[i, p] * temp[p]
            yi = e[i] - sum
            temp = np.append(temp, yi)
        y = np.append(y, temp)

    y = y.reshape(n, n)

    for k in range(0, n):
        yi = y[k]
        x = np.zeros(n)
        for i in range(0, n):
            sum = 0
            for k in range(0, i):
                sum += U[n - i - 1, n - k - 1] * x[n - k - 1]
            x[n - i - 1] = 1 / U[n - i - 1, n - i - 1] * (yi[n - i - 1] - sum)
        result_matrix = np.append(result_matrix, x)
    return result_matrix.reshape(n, n).transpose()