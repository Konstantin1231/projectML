#main file
from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tools import *
from scipy.spatial import distance


fig=plt.figure()
np.random.seed(100)
#generating data (2D)
N1, N2 , N3 = 200 , 100 , 50 # number of points for different classes

ax1=fig.add_subplot(121)
X1= np.array([1 +1.5 *norm.rvs(size=N1), 1 + 0.9*norm.rvs(size=N1)]) # gaussian class 1
ax1.scatter(X1[0],X1[1])


x2=np.linspace(-4,4,N2)
epsilon2= 0.2 * norm.rvs(size=N2) # noise
X2=np.array([x2,4 + np.sin(x2) + epsilon2]) # second class
ax1.scatter(X2[0],X2[1])


x3=np.linspace(1.5,6,N3)
epsilon3= 0.2 * norm.rvs(size=N3) # noise
X3=np.array([x3, 7 + -np.log(x3) + epsilon3]) # class
ax1.scatter(X3[0],X3[1])


#unsupervised data

X=np.append(X1,X2,axis=1)
X=np.append(X,X3, axis=1) # unsupervised data
X=np.transpose(X)

Y_test=np.zeros(N1)
Y_test=np.append(Y_test,np.ones(N2))
Y_test=np.append(Y_test,2*np.ones(N3)) # corresponding labels

ax2=fig.add_subplot(122)
ax2.scatter(X[:,0],X[:,1])
plt.show()







# looking on clustering by k-means methode
fig=plt.figure()
ax1=fig.add_subplot(111)

random_state=170
Y=KMeans(n_clusters=3, random_state=random_state).fit_predict(X)# k-means algoritme

ax1.scatter(X[:,0],X[:,1], c=Y)
plt.title("K-means")
plt.show()



#spectral clustering

#one connected graph (very sensetive in the choice of lambda_)
lambda_=2.6
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #3.5 is optimal value

#normalised decomposition of L_sym
y=normalised_L_sym(3,W)

index0=np.where(y==1)#just to have same colors
index1=np.where(y==0)
y[index0]=0
y[index1]=1

plt.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym ' + ' rate=' + str(rate(Y_test,y))
plt.title(string)
plt.show()


#normalised decomposition of L_rw
y=normalised_L_rw(3,W)

index0=np.where(y==1)#just to have same colors
index1=np.where(y==2)
index2=np.where(y==0)
y[index0]=0
y[index1]=1
y[index2]=2

plt.scatter(X[:,0],X[:,1],c=y)
string='normalised L_rw ' + ' rate=' + str(rate(Y_test,y))
plt.title(string)
plt.show()


#normalised decomposition of L_rw
y=unnormalised(3,W)
plt.scatter(X[:,0],X[:,1],c=y)
string='unnormalised ' + ' rate=' + str(rate(Y_test,y))
plt.title(string )
plt.show()
