import numpy as np
from scipy.spatial import distance
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.stats import  norm
from sklearn.cluster import KMeans

def normalised_L_sym(k,W):
    d=W.sum(axis=1)
    d=d**(-1/2)
    D=np.eye(len(d))
    np.fill_diagonal(D,d)
    L=np.eye(len(d)) -  D@W@D#laplacian matrix
    U , sigma, Vh =linalg.svd(L)
    U=U[:,len(d)-k:]
    y_pred=KMeans(n_clusters=k).fit_predict(normalize(U))
    return y_pred


def normalised_L_rw(k,W):
    d = W.sum(axis=1)
    d = d ** (-1)
    D = np.eye(len(d))
    np.fill_diagonal(D, d)
    L = np.eye(len(d)) - D @ W   # laplacian matrix
    U, sigma, Vh = linalg.svd(L)
    U = U[:, len(d) - k:]
    y_pred = KMeans(n_clusters=k).fit_predict(U)
    return y_pred

def unnormalised(k,W):
    d = W.sum(axis=1)
    D = np.eye(len(d))
    np.fill_diagonal(D, d)
    L=D-W
    U, sigma, Vh = linalg.svd(L)
    U = U[:, len(d) - k:]
    y_pred = KMeans(n_clusters=k).fit_predict(U)
    return y_pred

def rate(y_test, y_pred):
    rate=np.array([ (y_test[i]-y_pred[i]==0) for i in range(len(y_test))])
    return np.mean(rate)

