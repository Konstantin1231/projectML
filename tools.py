import itertools

import numpy as np
from scipy.spatial import distance
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.stats import  norm
from sklearn.cluster import KMeans



#computing labels using different algorithme of spectral clustering
def normalised_L_sym(k,W):
    d=W.sum(axis=1)
    d=d**(-1/2)
    D=np.eye(len(d))
    np.fill_diagonal(D,d)
    L=np.eye(len(d)) -  D@W@D#laplacian matrix
    U , sigma, Vh =linalg.svd(L)
    U=U[:,len(d)-k:]
    y_pred=KMeans(n_clusters=k, random_state=3).fit_predict(normalize(U))
    return y_pred


def normalised_L_rw(k,W):
    d = W.sum(axis=1)
    d = d ** (-1)
    D = np.eye(len(d))
    np.fill_diagonal(D, d)
    L = np.eye(len(d)) - D @ W   # laplacian matrix
    U, sigma, Vh = linalg.svd(L)
    U = U[:, len(d) - k:]
    y_pred = KMeans(n_clusters=k,random_state=3).fit_predict(U)
    return y_pred

def unnormalised(k,W):
    d = W.sum(axis=1)
    D = np.eye(len(d))
    np.fill_diagonal(D, d)
    L=D-W
    U, sigma, Vh = linalg.svd(L)
    U = U[:, len(d) - k:]
    y_pred = KMeans(n_clusters=k,random_state=3).fit_predict(U)
    return y_pred




#diferent similarity graphs
def k_nearest(W,k):
    n=len(W[:,0])
    for i in range(n):
        I = np.argpartition( W[i,:],n-k-1)[0:n-k-1]
        W[i,I]=0
    for i in range(n):
        for j in range(n):
           W[i,j]=np.maximum(W[i,j],W[j,i])
    return W

def mutal_k_nearest(W,k):
    n = len(W[:, 0])
    for i in range(n):
        I = np.argpartition( W[i,:],n-k-1)[0:n-k-1]
        W[i,I]=0
    for i in range(n):
        for j in range(n):
            W[i, j] = np.minimum(W[i, j], W[j, i])
    return W


def epsilon_neighbor(W,epsilon):
    n=len(W[0,:])
    for i in range(n):
        for j in range(n):
           W[i,j]= W[i,j] * (W[i,j]>epsilon)
    return W



#calculation of rate and correcting labels
def rate(y_test, y_pred):
    x=np.array([ (y_test[i]-y_pred[i]==0) for i in range(len(y_test))])
    return np.mean(x)


def correcting_labels(y_pred,y_test):
    Ind_0=np.where(y_pred==0)
    Ind_1=np.where(y_pred==1)
    Ind_2=np.where(y_pred==2)


    #keep Ind_0
    y1=y_pred
    y1[Ind_1]=2
    y1[Ind_2]=1

    #keep Ind 1
    y2=y_pred
    y2[Ind_0]=2
    y2[Ind_2]=0

    #keep Ind 2
    y3=y_pred
    y3[Ind_0] = 1
    y3[Ind_1] = 0

    #shif by 1
    y4=y_pred
    y4[Ind_0]=2
    y4[Ind_1]=0
    y4[Ind_2]=1

    #shift by 2
    y5 = y_pred
    y5[Ind_0] = 1
    y5[Ind_1] = 2
    y5[Ind_2] = 0

    Y=np.array([y_pred,y1,y2,y3,y4,y5])
    rates_=np.array([ rate(y_test ,Y[i,:]) for i in range(6) ])

    k=np.where(rates_==np.amax(rates_))
    return Y[k,:]


def best_rate(truth, y, k):
    max_rate = rate(truth, y)

    # Split into label-belonging matrix
    label_belong = np.zeros((k, len(y)))
    for i in range(k):
        label_belong[i, :] = (y == i)

    # Test all label-permutations
    labels = np.cumsum(np.ones(k))-1
    for perm in itertools.permutations(labels):
        test = np.zeros(len(y))
        for i in range(k):
            test += perm[i] * label_belong[i, :]
        max_rate = max(max_rate, rate(truth, test))

    return max_rate










