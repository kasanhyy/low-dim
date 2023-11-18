import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from pmm import pmm_data
from numpy.linalg import norm

from plot import computeW1


def round01(v):
    for i, c in enumerate(v):
        if c > 1:
            v[i] = 1
        elif c < 0:
            v[i] = 0

def matrixLap(d):
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            temp = np.random.laplace(scale=1)
            A[i, j] += temp
            A[j, i] += temp
    return A


def proj_pmm_data(X, y, eps=1, d2=-1, scale=1, isNotShift=False):
    n, d = X.shape

    row = 0

    Xbar = np.mean(X, axis=0)
    M = X.T @ X / n - np.outer(Xbar, Xbar)
    #####################
    sigma = 12 * d**2 / (eps * n)  ### See if we can remove the d**2
    A = matrixLap(d) * sigma
    M2 = M + A
    vecs, sig_vals, _ = np.linalg.svd(M2, hermitian=True)
    # every col of vecs is eigen vector

    # # get sorted eig
    # eigen_zip = list(zip(e_vals, e_vecs))
    # sorted_eigen_zip = sorted(eigen_zip, reverse=True)
    # e_vals, e_vecs = zip(*sorted_eigen_zip)
    # e_vals = np.array(e_vals)
    # e_vecs = np.array(e_vecs)

    if d2 == -1:
        d2 = 10

    if isNotShift:
        proj_basis = np.array(vecs[:, :d2])
        R = np.sqrt(d)
        proj_X = X @ proj_basis
        proj_X = (proj_X + R) / (2 * R)
        new_proj_X = pmm_data(proj_X, eps=eps * 3 / 4, scale=scale)
        syn = new_proj_X @ proj_basis.T
        for v in syn:
            round01(v)
        return syn

    proj_basis = np.array(vecs[:, :d2])

    # new_proj = [0]*10
    # for i in range(10):
    #     new_proj[i] = np.array([X[j] for j in range(n) if y[j]==i]) @ proj_basis @ proj_basis.T
    # return new_proj

    ###################
    shift_noise = np.random.laplace(scale=4 / (eps * n), size=d)
    shift_X = X - Xbar - shift_noise
    R = np.sqrt(d) + np.linalg.norm(Xbar + shift_noise)

    proj_X = shift_X @ proj_basis
    proj_X = (proj_X + R) / (2 * R)
    #####################

    classified_proj_X = [0] * 10
    new_proj_X = [0] * 10
    syn = [0] * 10
    for i in range(10):
        classified_proj_X[i] = np.array([proj_X[i] for j in range(n) if y[j] == i])
        new_proj_X[i] = pmm_data(classified_proj_X[i], eps=eps / 4, scale=scale)
        # new_proj_X[i] = classified_proj_X[i]
        new_proj_X[i] = (2 * R) * new_proj_X[i] - R
        syn[i] = new_proj_X[i] @ proj_basis.T

    ###################
    for i in range(10):
        shift_noise_2 = np.random.laplace(scale=4 / (eps * n), size=d)
        syn[i] = syn[i] + Xbar + shift_noise_2

        for v in syn[i]:
            round01(v)
    return syn
