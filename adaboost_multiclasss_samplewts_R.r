
wine=read.csv("wine.csv",sep=",")
wine[,2]

X=wine[,c(-1,-2)]
y=wine[,2]
for(i in 1:ncol(X)){
  X[is.na(X[,i]), i] <- mean(X[,i], na.rm = TRUE)
}


library(RSNNS)

#split train test
split.wine<-splitForTrainingAndTest(X,y,ratio=0.10)
# $inputsTrain $targetsTrain $inputsTest $targetsTest
split.wine$targetsTest

library("e1071")

svm_model <- svm(split.wine$inputsTrain,split.wine$targetsTrain,kernel="radial",gamma=0.03,type="C-classification")
summary(svm_model)
predicted <- predict(svm_model,split.wine$inputsTest)
#?svm

table(predicted,split.wine$targetsTest)
#predicted

###set max iterations T
T=5
#initialize sample weights
dm=replicate(nrow(split.wine$inputsTrain),1/nrow(split.wine$inputsTrain))
length(dm)

library("wSVM")
#svm.wt=wsvm(split.wine$inputsTrain, split.wine$targetsTrain, c.n=dm, kernel = list(type = 'rbf', par = NULL))

library(gmum.r)
library(SparseM)
library(Matrix)
library(ggplot2)
        
svm.wt <- SVM(split.wine$inputsTrain,split.wine$targetsTrain, core="svmlight", kernel="rbf", C=1.0, 
                        gamma=0.5, example.weights=dm)

#summary(svm.wt)
predicted <- predict(svm.wt,split.wine$inputsTrain)
as.numeric(predicted)

###aplha parameter from adaboost algorith 
alpha=c()

###list to store predictions
pred=list()

###list to store test vector predictions
testpred=list()

###define number of classes
classes=3
dm


###run adabosst multiclass with cost vector to each class
for (t in 1:T){
svm.wt <- SVM(split.wine$inputsTrain,split.wine$targetsTrain, core="svmlight", kernel="rbf", C=1.0, 
                        gamma=0.5, example.weights=dm)

#summary(svm.wt)
#predict test values
predicted_test <- predict(svm.wt,split.wine$inputsTest)
#predict train values
predicted_train<- predict(svm.wt,split.wine$inputsTrain)    

    
#store predicted test values
testpred[[t]]<-(predicted_test)
    
#store predicted train values
pred[[t]]<-(predicted_train)
    
#initially for each iteration error==0
err=0
    
    #for each example in train predictions check misclassification
    for (i in (1:nrow(split.wine$inputsTrain))){
        if (pred[[t]][i] != split.wine$targetsTrain[i]){
            y=split.wine$targetsTrain[i]
            err=err + dm[i] #* cost[y-1]
            }
        }
    #calculate alpha wth err value and classes
    alpha[t]<-(log((1-err)/err) + log(classes-1))
    
    #calculate Z value for normaliztion
    Z=c()
    #print(err)
    for (i in (1:nrow(split.wine$inputsTrain))){
        #print(dm)
        #print(as.numeric( predicted_train[i]))
        Z[i]<-(dm[i] * exp((-(alpha[t]))*(split.wine$targetsTrain[i]) *as.numeric( predicted_train[i])))
        #print(Z)
    }
    #update the weights
    
    for (i in (1:nrow(split.wine$inputsTrain))){
        dm[i]=Z[i] / sum(Z)
        #print(dm)
        }

        
}   


#initialize the adaboost ensemble otput for each class
ada_class=c()
ada_class_pred=list()
#run for loop for number of classes
for (k in (1:classes)){
    
    #initialize predictions for each test vector
    
    
    #for each test vector 
    #for (i in 1:nrow(testpred[t]) ){
        
        #initialize the adaboost output as zero for each class
        ada_predicts=replicate(length(testpred[[1]]),0)
        #calculate the adaboost predicted value across t generations.this is a scalar
        for (t in 1:T){
            vec=testpred[[t]]
            al=alpha[t]
            for (i in 1:length(vec)){
            if (vec[i]==k){
                ada_predicts[i]= ada_predicts[i]+(al *1)
                }
            else{
                ada_predicts[i]=ada_predicts[i]+(al *0)
                }
            }
        }
        
    ada_class_pred[[k]]<-(ada_predicts)
        
    #}
    #predictions of each test vector across classes
    #ada_class.append(ada_class_pred)
}

cl=c()
for (i in 1:length(testpred[[1]]))
{
m = as.numeric(do.call( rbind, x)[,1])
cl[i]<-which(m==max(m))
}
cl

acc = length(which(split.wine$targetsTest == cl)) / length(cl)
acc

#test
#x = list(list(1,2), list(3,4), list(5,6))
#a=do.call( rbind, x)[,1]
#as.numeric(a)


