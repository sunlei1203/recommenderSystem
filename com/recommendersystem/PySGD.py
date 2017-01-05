import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
#from sklearn import cross_validatio as cv
header = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('MyProject/ml-100k/u.data',sep='\t',names=header)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = '+str(n_users)+' | number of movies = '+str(n_items)
train_pre = pd.read_csv('MyProject/ml-100k/ua.base',sep='\t',names=header)
test_pre = pd.read_csv('MyProject/ml-100k/ua.test',sep='\t',names=header)
train_data = pd.DataFrame(train_pre)
test_data = pd.DataFrame(test_pre)

# Create training and test matrix
R = np.zeros((n_users,n_items))
#print "R: "+str(R)
for line in train_data.itertuples():
    R[line[1]-1,line[2]-1] = line[3]
print "R: "+str(R)
T = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    T[line[1]-1,line[2]-1] = line[3]    

# Index matrix for training data
I = R.copy()
I[I>0] =1
I[I==0]=0
# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

#predict the unknown ratings
def prediction(P,Q):
    return np.dot(P.T,Q)

lmbda = 0.1 # Regularisation weight
k = 2  # Dimensionality of the latent feature space
m,n = R.shape  # Number of users and items
n_epochs = 10  # Number of epochs
gamma = 0.01 # Learning rate
P = 3 * np.random.rand(k,m) # Latent user feature matrix
#print "P: "+str(P)
#print "P2: "+str(P[1,:])
Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
#print "Q: "+str(Q)

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R>0]))

train_errors = []
test_errors = []

#Only consider non-zero matrix 
users,items = R.nonzero()
print "users :"
print users
print "items: "
print items
print "zip: "
print zip(users,items)
for epoch in xrange(n_epochs):
    for u,i in zip(users,items):
        e = R[u,i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
        P[:,u] += gamma * (e * Q[:,i] - lmbda * P[:,u])  # Update latent user feature matrix
        Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
        train_rmse = rmse(I,R,Q,P) # Calculate root mean squared error from train dataset
        test_rmse = rmse(I2,T,Q,P)# Calculate root mean squared error from test dataset
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        
# Check performance by plotting train and test errors
print "train_errors\n"+str(train_errors)
print "test_errors\n"+str(test_errors)
# Calculate prediction matrix R_hat (low-rank approximation for R)
R = pd.DataFrame(R)
R_hat = pd.DataFrame(prediction(P,Q))
print str(prediction(P,Q))
# Compare true ratings of user 17 with predictions
#ratings = pd.DataFrame(data=R.loc[16,R.loc[16,:] > 0]).head(n=5)
#ratings['Prediction'] = R_hat.loc[16,R.loc[16,:] > 0]
#ratings.columns = ['Actual Rating', 'Predicted Rating']
#ratings
#print ratings
        