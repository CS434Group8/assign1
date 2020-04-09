import csv
import sys
import numpy as np
import math
import matplotlib.pyplot as plt 

if(len(sys.argv)!=3):
    print("You must provide 3 arguments")
    quit()

#global variable
train_csv_name=sys.argv[1]
test_csv_name=sys.argv[2]
train_X,train_Y=[],[]
test_X,test_Y=[],[]



def readfile():
    with open(train_csv_name) as csvfile:
        train_csv = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in train_csv:
            arr=row[0].split(',')
            arr=list(map(float,arr)) #convert string arr to float arr
            
            features=arr[:-1]
            features.insert(0,1)  #find all features and add 1 at the first index
            train_X.append(features) #add X into dataset
            train_Y.append(arr[-1]) #add Y into dataset
            
    with open(test_csv_name) as csvfile:
        test_csv = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in test_csv:
            arr=row[0].split(',')
            arr=list(map(float,arr)) #convert string arr to float arr
            
            features=arr[:-1]
            features.insert(0,1)  #find all features and add 1 at the first index
            
            test_X.append(features) #add X into dataset
            test_Y.append(arr[-1]) #add Y into dataset



def part1():
    
    #do the following formula 
    train_X_transpose=np.transpose(train_X)
    result=np.matmul(train_X_transpose, train_X)
    result=np.linalg.inv(result)
    result=np.matmul(result,train_X_transpose)
    result=np.matmul(result,train_Y)

    return result    

def cal_ASE(w,X,Y):
    predictY=np.matmul(X,w)
    errorSum=0
    
    for i in range(0,len(predictY)):
        errorSum=errorSum+math.pow(predictY[i]-Y[i],2)
    return errorSum/(len(predictY))

def part2():
    train_dat_ASE=cal_ASE(w,train_X,train_Y)    
    test_dat_ASE=cal_ASE(w,test_X,test_Y)  
    # print("the learned weight vector is:")
    # print(w)
    # print('\n')
    # print("ASE over the training data is: ",train_dat_ASE)  
    # print("ASE over the testing data is: ",test_dat_ASE)
    return train_dat_ASE,test_dat_ASE

def generateRandomFeature():
    train_size=len(train_X)
    test_size=len(test_X)
    
    train_extra_features=[]
    test_extra_features=[]
    
    for i in range(0,20):
        feature1=np.random.standard_normal(train_size)
        feature2=np.random.standard_normal(test_size)
        train_extra_features.append(feature1)
        test_extra_features.append(feature2)
        
    return train_extra_features,test_extra_features

# main function
readfile()
x_val=[]
y_train_val=[]
y_test_val=[]

# print("feature number: ",len(train_X[0])) 
w=part1()
train_ASE,test_ASE=part2()

x_val.append(len(train_X[0])-14)
y_train_val.append(train_ASE)
y_test_val.append(test_ASE)

# print('\n')
train_extra_features,test_extra_features=generateRandomFeature()

print("running")

for i in range(0,10):
    start=i*2
    for j in range(0,len(train_X)):
        train_X[j].append(train_extra_features[start][j])
        train_X[j].append(train_extra_features[start+1][j])
    for j in range(0,len(test_X)):
        test_X[j].append(test_extra_features[start][j])
        test_X[j].append(test_extra_features[start+1][j])
    # print("feature number: ",len(train_X[0])) 
    w=part1()
    train_ASE,test_ASE=part2()
    x_val.append(len(train_X[0])-14)
    y_train_val.append(train_ASE)
    y_test_val.append(test_ASE)
    # print('\n')

# print(x_val)
# print(y_train_val)
# print(y_test_val)


plt.figure()
plt.plot(x_val, y_train_val) 
  
# naming the x axis 
plt.xlabel('#of features') 
# naming the y axis 
plt.ylabel('train ASE') 
plt.ylim(20,25) 
plt.savefig('x-train_ASE.png')


plt.figure()
plt.plot(x_val, y_test_val) 

# naming the x axis 
plt.xlabel('#of features') 
# naming the y axis 
plt.ylabel('test ASE') 
plt.ylim(20,27) 

plt.savefig('x-test_ASE.png')


print("Please checkout x-train_ASE.png and x-test_ASE.png")