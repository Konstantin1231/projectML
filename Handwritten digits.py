import scipy.stats as st
import scipy.io
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tools import *
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture

np.random.seed(101)

# Import Matlab data and array shaping
mat = scipy.io.loadmat("digitFive1000.mat")
truth = np.array(mat["cluster_assignment____local"]-1).reshape((500,))
similarity = np.array(mat["similarity____local"])

# Cluster and get rate
y = normalised_L_rw(5, similarity)
print(best_rate(truth, y, 5))
print(rate(truth, y))

