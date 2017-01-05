import numpy as np
import pandas as pd
header = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('MyProject/ml-100k/u.data',sep='\t',names=header)
n_users = df.user_id.unique().shape[0]  #the num of all users
n_items = df.item_id.unique().shape[0]  # the num of all items
print 'Number of users = '+str(n_users)+' | number of movies = '+str(n_items)
train_pre = pd.read_csv('MyProject/ml-100k/ua.base',sep='\t',names=header)
test_pre = pd.read_csv('MyProject/ml-100k/ua.test',sep='\t',names=header)
#train_pre = pd.read_csv('F:\PracticeMakesPerfect/MyJava/GUI/netHomeWork2/PythonTest/ml-100k/train.txt',sep='\t',names=header)
#test_pre = pd.read_csv('F:\PracticeMakesPerfect/MyJava/GUI/netHomeWork2/PythonTest/ml-100k/test.txt',sep='\t',names=header)
train_data = pd.DataFrame(train_pre)
test_data = pd.DataFrame(test_pre)
#print 'the train_data is: '+'\n'+str(train_data)
# there is no differentce between test_pre and test_data
#print 'the test_pre is: '+str(test_pre)
#print 'the test_data is: '+'\n'+str(test_data)
R = np.zeros((n_users,n_items))
for line in train_data.itertuples():
    R[line[1]-1,line[2]-1] = line[3]
#print 'the train matrix R is: '+'\n'+str(R)
T = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    T[line[1]-1, line[2]-1] = line[3]   #build matrix,the matrix label start with 0 while the users and items begin with 1
    #print str(line[1])+','+str(line[2])+','+str(line[3])
#print 'the test matrix T is: '+'\n'+str(T)

# Index matrix for training data
I = R.copy()
I[I>0] = 1
I[I<0] = 0
# Index matrix for test data
I2 = T.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0
#print 'the index matrix I2 is :'+'\n'+str(I2)

# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I*(R-np.dot(P.T,Q)))**2)/len(R[R>0]))

lmbda = 0.1 # Regularisation weight
k = 20 # Dimensionality of latent feature space
m, n = R.shape # Number of users and items
#print str(m)+','+str(n)
n_epochs = 15 # Number of epochs

P = 3 * np.random.rand(k,m)  # build random matrix P of k rows and m columns
Q = 3 * np.random.rand(k,n)
#print 'the random matrix P is :'+str(P)
#print 'the random matrix Q is :'+str(Q)

Q[0,:] = R[R!=0].mean(axis=0)
#print str(Q[0,:])
E = np.eye(k)  
#print str(E)

train_errors = []
test_errors = []

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix Q and estimate P
    for i,Ii in enumerate(I):
        nui = np.count_nonzero(Ii)
        #print 'i: '+str(i)+','+'Ii : '+str(Ii)+'nui: '+str(nui) 
        if(nui==0):nui = 1
        Ai = np.dot(Q,np.dot(np.diag(Ii),Q.T))+lmbda * nui * E
        Vi = np.dot(Q,np.dot(np.diag(Ii),R[i].T))
        #print 'Ai: '+str(Ai)+'Vi: '+str(Vi)
        P[:,i] = np.linalg.solve(Ai,Vi)
    for j,Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij)
        #print 'j: '+str(j)+','+'Ij: '+str(Ij)+'nmj: '+str(nmj) 
        if(nmj==0): nmj=1
        Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
        Vj = np.dot(P, np.dot(np.diag(Ij), R[:,j]))
        #print 'Aj: '+str(Aj)+'Vj: '+str(Vj)
        Q[:,j] = np.linalg.solve(Aj,Vj)
    train_rmse = rmse(I,R,Q,P)
    test_rmse = rmse(I2,T,Q,P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print "[Epoch %d/%d] train error: %f, test error: %f" %(epoch+1, n_epochs, train_rmse, test_rmse)
print "Algorithm converged"
    
    
