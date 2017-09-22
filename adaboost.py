
# coding: utf-8

# In[1]:

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


hearttrain=pd.read_csv("Heart.csv")
#print(type(hearttrain))

X = hearttrain
X=X.drop(X.columns[[0, 14]], axis=1)
y=hearttrain['AHD']

#X=list(X)
#y=list(y)

y=y.replace(['Yes', 'No'], [1, -1])

#X=pd.factorize(X["ChestPain"])

lb_make = LabelEncoder()
X["ChestPain"] = lb_make.fit_transform(X["ChestPain"])
#X["Thal"] = lb_make.fit_transform(X["Thal"])


temp = pd.get_dummies(pd.Series(X["Thal"]))
X=pd.concat([X,temp],axis=1)
X=X.drop(["Thal"],axis=1)

X=X.fillna(X.median())

#print(X.head())
#ect=[]
#for column in vect:
#    temp=pd.get_dummies(pd.Series(df[column])
#    df=pd.concat([df,temp],axis=1)
#    df=df.drop([column],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0) 


svc_linear = svm.SVC(kernel='linear', C=1)



# In[23]:

T=10
dm=np.repeat(1/len(X_train), len(X_train), axis=0)
svc_linear.fit(X_train, y_train,sample_weight=dm)
predicted= svc_linear.predict(X_test)

#print((predicted))
#print (dm)
#X
len(X_train)


# In[38]:

alpha=[]
pred=[]
testpred=[]
for t in range(10):
    svc_linear.fit(X_train, y_train,sample_weight=dm)
    predicted_train=(svc_linear.predict(X_train))
    predicted_test=(svc_linear.predict(X_test))
    testpred.append(predicted_test)
    pred.append(predicted)
    err=0
    for i in range(len(X_train)):
        if (pred[t][i] == y_train.values[i]):
            err+=dm[i]
    alpha.append(np.divide(np.log((1-err)/err),2))
    Z=[]
    for i in range(len(X_train)):
        Z.append(dm[i] * np.exp((-alpha[t])*y_train.values[i] * predicted[i]))
    
    dm=[Z[i] / sum(Z) for i in range(len(X_train))]

ada_predicts=0
for t in range(10):
    ada_predicts+=(alpha[t] * testpred[t] ) 
    
y_preds=[]
for i in range(len(y_test)):
    if (ada_predicts[i] > 0):
        y_preds.append(1)
    else:
        y_preds.append(0)

y_preds


# In[ ]:



