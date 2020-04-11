import csv
import sys
import numpy as np
import math
import copy
import matplotlib.pyplot as plt 

if(len(sys.argv)!=4):
    print("You must provide 4 arguments")
    quit()

#global variable
train_csv_name=sys.argv[1]
test_csv_name=sys.argv[2]

lambdas=list(map(float,sys.argv[3].split(',')))
print(lambdas)

learning_rate=0.0001
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
            for i in range(0,len(features)):
                features[i]=features[i]/255
                
            train_X.append(features) #add X into dataset
            train_Y.append(arr[-1]) #add Y into dataset

            
    with open(test_csv_name) as csvfile:
        test_csv = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in test_csv:
            arr=row[0].split(',')
            arr=list(map(float,arr)) #convert string arr to float arr
            
            features=arr[:-1]
            features.insert(0,1)  #find all features and add 1 at the first index
            for i in range(0,len(features)):
                features[i]=features[i]/255
            
            test_X.append(features) #add X into dataset
            test_Y.append(arr[-1]) #add Y into dataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradientDescent(w,lambd):
    newW=[]
    w_transpose=np.transpose(w)
    
    sumli=[0]*257
    for i in range(0,len(train_X)):
        decreaseRate=np.matmul(w_transpose,train_X[i])
        decreaseRate=sigmoid(decreaseRate)
        decreaseRate=(decreaseRate-train_Y[i])
        li=decreaseRate*np.array(train_X[i])
        sumli+=li
    
    newW=w-learning_rate*(sumli-lambd*np.array(w))
    newW[0]=newW[0]-learning_rate*sumli[0]
    return newW


   
def calAccuracy(data_x,data_y):
    accuracy=0
    w_transpose=np.transpose(w)
    for i in range(0,len(data_x)):
        predict_x=np.matmul(w_transpose,data_x[i])
        # print(predict_x)
        if(predict_x>=0 and data_y[i]==1):
            accuracy+=1
        if(predict_x<0 and data_y[i]==0):
            accuracy+=1
    accuracy=accuracy/len(data_x)
    accuracy=round(accuracy,3)
    return accuracy
    
#main function
readfile()
w=[0]*257


x=[]
y1=[]
y2=[]

for lambd in lambdas:
    print("lambda: ",lambd)
    for iter in range(0,10):
        print("doing "+str(iter)+" iteration")
        w=gradientDescent(w,lambd)

    A1=calAccuracy(train_X,train_Y)
    A2=calAccuracy(test_X,test_Y)
    x.append(lambd)
    y1.append(A1)
    y2.append(A2)
            
print(x)
print(y1)
print(y2)



plt.figure()
plt.plot(x, y1) 
  
# naming the x axis 
plt.xlabel('lambdas') 
# naming the y axis 
plt.ylabel('Train Accuracy') 
plt.ylim(0,1) 
plt.savefig('Train_Accuracy_lambdas.png')


plt.figure()
plt.plot(x, y2) 
# naming the x axis 
plt.xlabel('lambdas') 
# naming the y axis 
plt.ylabel('Test Accuracy') 
plt.ylim(0,1) 

plt.savefig('Test_Accuracy_lambdas.png')


print("Please checkout Train_Accuracy_lambdas.png and Test_Accuracy_lambdas.png")