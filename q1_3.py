import csv
import sys
import numpy as np
import math

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
            #not adding 1 in this part
            # features.insert(0,1)  #find all features and add 1 at the first index 
            
            train_X.append(features) #add X into dataset
            train_Y.append(arr[-1]) #add Y into dataset
            
    with open(test_csv_name) as csvfile:
        test_csv = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in test_csv:
            arr=row[0].split(',')
            arr=list(map(float,arr)) #convert string arr to float arr
            
            features=arr[:-1]
            
            #not adding 1 in this part
            # features.insert(0,1)  #find all features and add 1 at the first index
            
            test_X.append(features) #add X into dataset
            test_Y.append(arr[-1]) #add Y into dataset



def part1():
    
    #do the following formula 
    train_X_transpose=np.transpose(train_X)
    result=np.matmul(train_X_transpose, train_X)
    result=np.linalg.inv(result)
    result=np.matmul(result,train_X_transpose)
    result=np.matmul(result,train_Y)
    
    print("Part1\nW is:")
    print(result)
    return result    

def cal_ASE(w,X,Y):
    predictY=np.matmul(X,w)
    errorSum=0
    
    for i in range(0,len(predictY)):
        errorSum=errorSum+math.pow(predictY[i]-Y[i],2)
    return errorSum/(len(predictY))

def part3():
    train_dat_ASE=cal_ASE(w,train_X,train_Y)    
    test_dat_ASE=cal_ASE(w,test_X,test_Y)  
    print("\nPart3\n")
    print("the learned weight vector is:")
    print(w)
    print("ASE over the training data is: ",train_dat_ASE)  
    print("ASE over the testing data is: ",test_dat_ASE)

#main function
readfile()
w=part1()
part3()

print("Influence Explanation:\n If you remove all 1 from the dummy data, there are only 13 features in the weight vector. Instead of Y=b+w1X1+w2X2+...+w13X13, it will miss b in the function. As a result, we are missing bias and the ASE will be higher since it is less accurate")

