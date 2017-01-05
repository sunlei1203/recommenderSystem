'''
Created on 2016/12/9
Description: this is an incremental recommender algorithm
@author: sunlei
'''
from StaticTrain import *
from updaterating import *
def predict(P,Q):
    return np.dot(P,Q)
def preprocessing(m,M):
    header = ['user_id','item_id','rating','timestamp']
    df = pd.read_csv('MyProject/dataSet/RKMF_data/rest_new.txt',sep='\t',names=header)
    n_users = df.user_id.unique().shape[0]
    #n_items = df.item_id.unique().shape[0]
    n_items = 1682
    #print 'Number of new users = '+str(n_users)+' | number of movies = '+str(n_items)
    new_user = pd.DataFrame(df)
    # Create training and test matrix
    R = np.zeros((n_users,n_items))
    #print "R: "+str(R)
    for line in new_user.itertuples():
        R[line[1]-M-1,line[2]-1] = line[3]
    #usersnum,itemsnum = R.nonzero()
    newusers_dict_train={}
    newusers_dict_test={}
#     for u,i in zip(usersnum,itemsnum):
#         newusers_dict_train.setdefault(u,[])
#         newusers_dict_train[u].append((i,R[u][i]))
    user = 0
    V=0      #the rating number of test data set
    while user<len(R):
        n = np.count_nonzero(R[user])
        m1 = min(m,n/2)
        i = 0
        j = 0
        while j<len(R[user]) and i<m1:
            if R[user][j]!=0:
                newusers_dict_train.setdefault(user,[])
                newusers_dict_train[user].append((j,R[user][j])) 
                i+=1
            j+=1
        
        while j<len(R[user]) and i<n:   #       i<n
            if R[user][j]!=0:
                newusers_dict_test.setdefault(user,[])
                newusers_dict_test[user].append((j,R[user][j]))
                V+=1
                i+=1
            j+=1
        user+=1
    return R,newusers_dict_train,newusers_dict_test,V
def combineBoth(S,R):
    return np.row_stack((S,R))

def error_on_each_newuser(i,u,newusers_dict_test,W,H):
    get_i_real_ratings = newusers_dict_test.get(i)
    #real_ratings = user
    error_sum1 = 0
    for tuple1 in get_i_real_ratings:
        (col,rating) = tuple1
        predict_rating = predict(W[i+u],H[:,col])
        error_sum1 += pow((rating - predict_rating),2)
    return error_sum1
    
def rmse(user_error_sum,V):
    return np.sqrt(user_error_sum/V)
def process_each_newuser(S,R,W,H,K,u,m,newusers_dict,newusers_dict_test,V):
    one_user_rmse = {}
    all_user_rmse = {}
    for epoch in xrange(m):
        i=0
        error_sum=0
#         print W[i+u:]
#         (column,rating) = newusers_dict[i][epoch]
#         print H[:,column]
        while i < len(R):
            #user = R[i]
            if epoch > len(newusers_dict[i]):
                i+=1
                continue;
            (column,rating) = newusers_dict[i][epoch]
            err = rating-predict(W[i+u],H[:,column])
            j=0
            while j<K:
                W[i+u][j] = W[i+u][j]+alpha*(err*H[j][column]-lmbda*W[i+u][j])
                j+=1
            one_user_error = error_on_each_newuser(i,u,newusers_dict_test,W,H)
            sqrt_one_user = rmse(one_user_error,len(newusers_dict_test[i]))
            one_user_rmse.setdefault(i,[])
            one_user_rmse[i].append((epoch,sqrt_one_user))
            error_sum+=one_user_error
            i+=1
        all_new_rmse = rmse(error_sum,V) 
        all_user_rmse[epoch] =  all_new_rmse
    return one_user_rmse,all_user_rmse
#         print  "the "+str(epoch)+" iteration rmse : "+str(all_new_rmse)
#     adduser(S,W,H,u,alpha,lmbda)                # u is the linenumber  of the first new user
def writeFile(one_user,all_user):
    newfile1 = open("MyProject/dataSet/RKMF_data/each_user_output1.txt","w")
    newfile2 = open("MyProject/dataSet/RKMF_data/each_epoch_output2.txt","w")
    str_write1 = ""
    str_write2 = ""
    for i in one_user.keys():
        list_epoch = one_user.get(i)
        for tuple1 in list_epoch:
            (epoch1,rmse) = tuple1
            str_write1 += str(epoch1)+" "+str(rmse)+"\n"
    newfile1.write(str_write1)
    for j in all_user.keys():
        str_write2 += str(j)+" "+str(all_user.get(j))+"\n"
    newfile2.write(str_write2)
    newfile1.close() 
    newfile2.close() 
if __name__ == "__main__":
    (S,W,H)=Statictrain()   #???type(S)   
    u = len(S)  # start linenumber of the first new user
    m=50
    alpha = 0.0002
    lmbda = 0.02
    M = len(S)
    N = len(S[0])
    K = 10
    R,newusers_dict_train,newusers_dict_test,V = preprocessing(m,M)
    S = combineBoth(S,R)       #trained Matrix S combined untrained new user Matrix
#     print newusers_dict_test
    P = np.random.rand(len(R),K)
    W = np.row_stack((W,P)) 
#     print W
    one_user,all_user = process_each_newuser(S,R,W,H,K,u,m,newusers_dict_train,newusers_dict_test,V)
    writeFile(one_user,all_user)
    