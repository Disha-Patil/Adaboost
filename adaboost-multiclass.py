
# coding: utf-8

# In[23]:

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# In[24]:

wine=pd.read_csv("wine.scale",header=None,sep='\s+')
#print(wine)
for i in range(9):
    wine[i+1] = wine[i+1].str[2:]
ints=[10, 11, 12, 13]
for idx, val in enumerate(ints):
    wine[val] = wine[val].str[3:]

#for i in range(14):
    #print(wine[10])
#    wine[i] = wine[i].astype(float)

wine = wine.convert_objects(convert_numeric=True)
#wine


# In[25]:

X = wine
X=X.drop(X.columns[[0]], axis=1)
y=wine[0]
X=X.fillna(X.median())
X


# In[68]:

###split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) 

###build linear SVM
svc_linear = svm.SVC(kernel='linear', C=1)
#svc_multiclass=OneVsRestClassifier(LinearSVC(random_state=0))
#svc_multiclass.fit(X_train, y_train).predict(X_train)
#len(y_test)


# In[101]:

###set max iterations T
T=10

#initialize sample weights
dm=np.repeat(1/len(X_train), len(X_train), axis=0)
#svc_linear.fit(X_train, y_train,sample_weight=dm)

#predict with initialized weights
predicted= svc_linear.fit(X_train, y_train,sample_weight=dm).predict(X_test)

#print((predicted))
#print (dm)
#X
#np.where(predicted==2)[0].tolist()


# In[102]:

###aplha parameter from adaboost algorith 
alpha=[]

###list to store predictions
pred=[]

###list to store test vector predictions
testpred=[]

###define number of classes
classes=3

###initialize the cost vector for class imbalance
#cost=np.random.uniform(0,1,3).tolist()
#cost=[]
#for i in range(classes):
#    cost.append(np.random.uniform(0,1,1).tolist())
    

###run adabosst multiclass with cost vector to each class
for t in range(T):
    #print(t)
    
    #fit a linear svm with weights dm
    svc_linear.fit(X_train, y_train,sample_weight=dm)
    
    #predict train values
    predicted_train=(svc_linear.predict(X_train))
    
    #predict test values
    predicted_test=(svc_linear.predict(X_test))
    
    #store predicted test values
    testpred.append(predicted_test)
    
    #store predicted train values
    pred.append(predicted_train)
    
    #initially for each iteration error==0
    err=0
    
    #for each example in test predictions check misclassification
    for i in range(len(X_test)):
        if (testpred[t][i] != y_test.values[i]):
            y=y_train.values[i]
            err+=dm[i] #* cost[y-1]
    #calculate alpha wth err value and classes
    alpha.append(np.log((1-err)/err) + np.log(classes-1))
    
    #calculate Z value for normaliztion
    Z=[]
    #print(err)
    for i in range(len(X_train)):
        Z.append(dm[i] * np.exp((-alpha[t])*y_train.values[i] * predicted_train[i]))
    
    #update the weights
    dm=[Z[i] / sum(Z) for i in range(len(X_train))]

#initialize the adaboost ensemble otput for each class
ada_class=[]

#run for loop for number of classes
for k in range(classes):
    
    #initialize predictions for each test vector
    ada_class_pred=[]
    
    #for each test vector 
    for i in range(len(testpred[t])):
        
        #initialize the adaboost output as zero for each class
        ada_predicts=0
        #calculate the adaboost predicted value across t generations.this is a scalar
        for t in range(10):
            vec=testpred[t]
            al=alpha[t]
            if (vec[i]==k+1):
                ada_predicts+=(al *1)
            else:
                ada_predicts+=(al *0)              
        ada_class_pred.append(ada_predicts)
    #predictions of each test vector across classes
    ada_class.append(ada_class_pred)


# In[103]:

#ada_class
#use vstack for better handling
adaclass_array=np.vstack(ada_class)
#initialize final pprediction
y_preds=[]
#for each test vector
for i in range(len(y_test)):
    #see for which class is adaboost prediction the maximum...and return that class
    y_preds.append(np.where(adaclass_array[:,i]==np.max(adaclass_array[:,i]))[0].tolist()[0]  + 1) #plus one because outputs as 1,2,3 needed
#y_preds


# In[104]:

y_preds


# In[ ]:



