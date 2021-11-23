#main file
from scipy.stats import uniform
from scipy.stats import norm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tools import *
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture


fig=plt.figure(figsize=(10,5))
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
ax1.set_title('actual 3-classes')


#unsupervised data

X=np.append(X1,X2,axis=1)
X=np.append(X,X3, axis=1) # unsupervised data
X=np.transpose(X)

Y_test=np.zeros(N1)
Y_test=np.append(Y_test,np.ones(N2))
Y_test=np.append(Y_test,2*np.ones(N3)) # corresponding labels

ax2=fig.add_subplot(122)
ax2.scatter(X[:,0],X[:,1])
ax2.set_title('unsupervised data points')
plt.show() #figure1







# looking on clustering by k-means and mixture gaussian methode
fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)


Y=KMeans(n_clusters=3).fit_predict(X)# k-means algoritme
ax1.scatter(X[:,0],X[:,1], c=Y)
ax1.set_title("k-means")
Y=GaussianMixture(n_components=3).fit_predict(X)#gaussian mixture
ax2.scatter(X[:,0],X[:,1], c=Y)
ax2.set_title("Gaussian mixture")

plt.show() #figure2



#spectral clustering

#one connected graph (very sensetive in the choice of lambda_)
lambda_=3.4
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #3.5 is optimal value
fig=plt.figure(figsize=(20, 5))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
#normalised decomposition of L_sym
y=normalised_L_sym(3,W)
ax1.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym ' + 'rate=' + str(rate(Y_test,y))
ax1.set_title(string)


#normalised decomposition of L_rw
y=normalised_L_rw(3,W)
ax2.scatter(X[:,0],X[:,1],c=y)
string='normalised L_rw ' + 'rate=' + str(rate(Y_test,y))
ax2.set_title(string)



#normalised decomposition of L_rw
y=unnormalised(3,W)
ax3.scatter(X[:,0],X[:,1],c=y)
string='unnormalised ' + 'rate=' + str(rate(Y_test,y))
ax3.set_title(string )
plt.show() #figure 3


"""""#look on how change the hyperparameter effect spectral clustering for one connected graph (reason to use normalized)

lambda_=1.5
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #2.6 is optimal value
y=normalised_L_sym(3,W)
I1=np.where(y==2)
I2=np.where(y==1)
y[I1]=1
y[I2]=2

plt.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym  '+ '  hyperparameter='+str(3) + '  rate=' + str(rate(Y_test,y))
plt.title(string)
plt.show() #figure 4


#compare normalised with unnrmalised vs hyperparameter
fig=plt.figure(figsize=(10,5))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)

l=np.linspace(1.5,7,50)
rate_vect=np.array([ rate(Y_test,normalised_L_sym(3,np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' ))) ) for lambda_ in l])
ax1.plot(l,rate_vect)
ax1.set_title('rate of normalised vs hyperparameter')

rate_vect=np.array([ rate(Y_test,unnormalised(3,np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' ))) ) for lambda_ in l])
ax2.plot(l,rate_vect)
ax2.set_title('rate of unnormalised vs hyperparameter')
plt.show() #figure 5"""





#k-nearest methode (sparse simitry graph)


#mutual-k-nerest(k=10)
np.random.seed(100)
lambda_=3.5
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #3.5 is optimal value
k=10

W=mutal_k_nearest(W,k) # mutual k-nereast similarity  graph

fig=plt.figure(figsize=(20, 5))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
#normalised decomposition of L_sym
y=normalised_L_sym(3,W)
ax1.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym ' + 'rate=' + str(rate(Y_test,y))
ax1.set_title(string)

#normalised decomposition of L_rw
y=normalised_L_rw(3,W)
ax2.scatter(X[:,0],X[:,1],c=y)
string='normalised L_rw ' + 'rate=' + str(rate(Y_test,y))
ax2.set_title(string)


#unormalised decomposition of L
y=unnormalised(3,W)
ax3.scatter(X[:,0],X[:,1],c=y)
string='unnormalised ' + 'rate=' + str(rate(Y_test,y))
ax3.set_title(string )
plt.show()

#just k-nearest(k=10)
np.random.seed(100)
lambda_=3.5
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #3.5 is optimal value
k=10

W=k_nearest(W,k) # k-nereast similarity  graph

fig=plt.figure(figsize=(20, 5))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
#normalised decomposition of L_sym
y=normalised_L_sym(3,W)
ax1.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym ' + 'rate=' + str(rate(Y_test,y))
ax1.set_title(string)

#normalised decomposition of L_rw
y=normalised_L_rw(3,W)
ax2.scatter(X[:,0],X[:,1],c=y)
string='normalised L_rw ' + 'rate=' + str(rate(Y_test,y))
ax2.set_title(string)


#unormalised decomposition of L
y=unnormalised(3,W)
ax3.scatter(X[:,0],X[:,1],c=y)
string='unnormalised ' + 'rate=' + str(rate(Y_test,y))
ax3.set_title(string )
plt.show()



#epsilon-neighbor-hood
lambda_=3.5
W=np.exp(-lambda_ * distance.cdist(X,X,metric='sqeuclidean' )) # define weihghts by one connected graph #3.5 is optimal value
epsilon=np.exp(-lambda_ *8)#define epsilon
W=epsilon_neighbor(W,epsilon)

fig=plt.figure(figsize=(20, 5))
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
#normalised decomposition of L_sym
y=normalised_L_sym(3,W)
ax1.scatter(X[:,0],X[:,1],c=y)
string='normalised L_sym ' + 'rate=' + str(rate(Y_test,y))
ax1.set_title(string)

#normalised decomposition of L_rw
y=normalised_L_rw(3,W)
ax2.scatter(X[:,0],X[:,1],c=y)
string='normalised L_rw ' + 'rate=' + str(rate(Y_test,y))
ax2.set_title(string)


#unormalised decomposition of L
y=unnormalised(3,W)
ax3.scatter(X[:,0],X[:,1],c=y)
string='unnormalised ' + 'rate=' + str(rate(Y_test,y))
ax3.set_title(string )
plt.show()



