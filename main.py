import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs, make_regression
from pmm import pmm_data
from psmm import psmm_data
from plot import *
from mpl_toolkits.mplot3d import Axes3D
import ot.plot
from scipy.stats import linregress


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dim = 5
    method = pmm_data
    n_list = np.array([250, 500, 1000, 2000, 4000, 8000])
    eps_list = np.array([1, 2, 4, 8, 16])
    # plotW1(dim, eps_list=eps_list, n_default=400, method=method, datatype="blobs")
    plotW1(dim, n_list=n_list, method=method, datatype="uniform", plot_data_pts=True)
