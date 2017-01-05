import numpy as np
import pandas as pd

#predict the unknown ratings
def prediction(P,Q):
    return np.dot(P.T,Q)
# Calculate the RMSE
def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R>0]))
def writeTofile(train_errors,test_errors):
    train_file1 = open('MyProject/dataSet/RKMF_data/1_train_output.txt',"w")
    test_file2 = open('MyProject/dataSet/RKMF_data/1_test_output.txt',"w")
    str1 = ""
    str2 = ""
    for item in train_errors:
        str1 += str(item)+"\n"
    for item in test_errors:
        str2 += str(item)+"\n"
    train_file1.write(str1)
    test_file2.write(str2)
    train_file1.close()
    test_file2.close()
def Statictrain():
    header = ['user_id','item_id','rating','timestamp']
    df = pd.read_csv('MyProject/dataSet/RKMF_data/static.txt',sep='\t',names=header)
    n_users = df.user_id.unique().shape[0]
    n_items = 1682
#     n_items = df.item_id.unique().shape[0]
    print 'Number of users = '+str(n_users)+' | number of movies = '+str(n_items)
    train_pre = pd.read_csv('MyProject/dataSet/RKMF_data/train.txt',sep='\t',names=header)
    test_pre = pd.read_csv('MyProject/dataSet/RKMF_data/test.txt',sep='\t',names=header)
    train_data = pd.DataFrame(train_pre)
    test_data = pd.DataFrame(test_pre)
    # Create training and test matrix
    R = np.zeros((n_users,n_items))
#     print "R: "+str(R)
    for line in train_data.itertuples():
        #print line,line[1],line[2],line[3]
        R[line[1]-1,line[2]-1] = line[3]
#     print "R: "+str(R)
    T = np.zeros((n_users,n_items))
    for line in test_data.itertuples():
        T[line[1]-1,line[2]-1] = line[3]    
    print "R and T both are already initialed values"
     # Index matrix for training data
    I = R.copy()
    I[I>0] =1
    I[I==0]=0
     # Index matrix for test data
    I2 = T.copy()
    I2[I2 > 0] = 1
    I2[I2 == 0] = 0
    lmbda = 0.1 # Regularisation weight
    k = 10  # Dimensionality of the latent feature space
    m,n = R.shape  # Number of users and items
    n_epochs = 10  # Number of epochs
    gamma = 0.01 # Learning rate
    P = 3 * np.random.rand(k,m) # Latent user feature matrix
    Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
    print "P and Q both are already initialed values"
    train_errors = []
    test_errors = []
     #Only consider non-zero matrix 
    users,items = R.nonzero()
    for epoch in xrange(n_epochs):
        print "The beginning of the "+str(epoch)+" times iterations!!"
        for u,i in zip(users,items):
            print "Starting to process user "+str(u)+"'s rating to item "+str(i)+"!!!"
            e = R[u,i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
            P[:,u] += gamma * (e * Q[:,i] - lmbda * P[:,u])  # Update latent user feature matrix
            Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix
            print "err, P,Q are already updated for this sample !!!"
            train_rmse = rmse(I,R,Q,P) # Calculate root mean squared error from train dataset
            print "rmse of train data set has calculated!!  "
            test_rmse = rmse(I2,T,Q,P)# Calculate root mean squared error from test dataset
            print "rmse of test data set has calculated!!  "
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
        
     # Check performance by plotting train and test errors
#     print "train_errors\n"+str(train_errors)
#     print "test_errors\n"+str(test_errors)
    print "training is over,start writing !!!"
    writeTofile(train_errors,test_errors)
     # Calculate prediction matrix R_hat (low-rank approximation for R)
    R = pd.DataFrame(R)
    print "Start to calculate predict Matrix S"
    S = pd.DataFrame(prediction(P,Q))
    #print str(prediction(P,Q))
    return (S,P.T,Q)                #2016-12-9   21:24  P.T
if __name__ == "__main__":
    (S,W,H)=Statictrain()
        