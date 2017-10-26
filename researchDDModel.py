

cd C:\Users\Linsen\Desktop\researchDD_model



import pandas as pd
import numpy as np


#detFiles shows under each traffic scenario, which links are sensored links (sensors are deployed in the link)
detFiles=['DetOutput_7.txt','DetOutput_8.txt','DetOutput_9.txt','DetOutput_95.txt','DetOutput_OD2_19.txt',
         'DetOutput_11.txt','DetOutput_12.txt']


#linkFiles shows the link statistics, where each file corresponds to the 
linkFiles=['LinkOutput_7.txt','LinkOutput_8.txt','LinkOutput_9.txt','LinkOutput_95.txt','LinkOutput_OD2_19.txt',
          'LinkOutput_11.txt','LinkOutput_12.txt']


#get statistics like the sec IDs (i.e., id of roads) with sensors, and their information, name the column name
dataDet=pd.read_csv(detFiles[0],header=None,sep=" ")
dataDet.columns=["detId","secId","count","occupany","density","speed","repId"]#format of detector data files
sec=np.unique(dataDet['secId']) 
nbRep=len(np.unique(dataDet["repId"]))
trainX=np.zeros((nbRep*len(detFiles),len(sec)))
dataLink=pd.read_csv(linkFiles[0],header=None,sep=" ")
dataLink.columns=["secId","flow","TT","density","delay","speed","repId"]#format of link files




detAll=[] #get the ID of all the links with detected (sec ID) 
for sec_i in sec:
    detAll.extend(np.where(dataLink["secId"]==sec_i)[0])
detAll=sorted(detAll)



for i in range(len(detFiles)):
    dataDet=pd.read_csv(detFiles[i],header=None,sep=" ")
    dataDet.columns=["detId","secId","count","occupany","density","speed","repId"]
    dataLink=pd.read_csv(linkFiles[i],header=None,sep=" ")
    dataLink.columns=["secId","flow","TT","density","delay","speed","repId"]
    allTT=dataLink.iloc[detAll]["TT"]
    trainX[i*nbRep:(i+1)*nbRep,:]=np.reshape(allTT,(nbRep,len(sec)))#training data, which include only the travel time data TT
    



trainY=np.ones((nbRep*len(detFiles),1))
for i in range(len(detFiles)):
    trainY[i*nbRep:(i+1)*nbRep,:]=i+1#training data Y, which is the class ID of the existing scenarios
nbGroup=len(detFiles)



#now code LDA/PCA
def LDAfeatures(trainX,nbGroup):
    nonzeroCol=np.sum(trainX,0)!=0
    trainXnew=trainX[:,nonzeroCol]

    nbGroup=len(detFiles)
    nk=int(len(trainXnew)/nbGroup)
    nbFeature=trainXnew.shape[1]
    sampleMean=np.mean(trainXnew,axis=0)
    SW=np.zeros((nbFeature))
    SB=np.zeros((nbFeature))


    for i in range(nbGroup):
    #print (' ')
        groupData=trainXnew[i*int(nk):(i+1)*int(nk,):]
        meanData=np.mean(groupData,axis=0)
        covData=np.cov(np.transpose(groupData))
        SW=SW+(nk-1)*covData
        SB=SB+nk*(np.dot(np.reshape(meanData-sampleMean,(nbFeature,1)),np.reshape(meanData-sampleMean,(1,nbFeature))))
    C=np.dot(np.linalg.inv(SW),SB)#matrix of our interest, SW^-1*SB is the matrix for eigen decomposition 
    W, v=np.linalg.eig(C)#w is the eigenvalues, v is eigen vectors
    XProj=np.zeros((nbGroup-1,nbFeature))
    for i in range(nbGroup-1):
        XProj[i,:]=v[:,i]
    XProjFull=np.zeros((nbGroup-1,trainX.shape[1]))
    XProjFull[:,nonzeroCol]=XProj
    CCTrainLDA=np.transpose(np.dot(XProj,np.transpose(trainXnew)))
    return CCTrainLDA
    



CCTrainLDA=LDAfeatures(trainX,nbGroup)




def PCAfeatures(trainX,nbGroup):
    nonzeroCol=np.sum(trainX,0)!=0
    trainXnew=trainX[:,nonzeroCol]
    nbFeature=trainXnew.shape[1]
    C=np.cov(np.transpose(trainXnew))#in PCA, decompose the the covariance matrix
    W, v=np.linalg.eig(C)#w is the eigenvalues, v is a set of eigenvectors
    XProj=np.zeros((nbGroup,nbFeature))
    for i in range(nbGroup):
        XProj[i,:]=v[:,i]
    XProjFull=np.zeros((nbGroup,trainX.shape[1]))
    XProjFull[:,nonzeroCol]=XProj#pick up npGroup of eigenvectors
    CCTrainPCA=np.transpose(np.dot(XProj,np.transpose(trainXnew)))
    return CCTrainPCA




CCTrainPCA=PCAfeatures(trainX,nbGroup)




M=32 #rounding and scaling factor for normalization



#CC_PS_LDA, normalization function
def CC_PS(CCTrain,M):
    bmin=CCTrain.min(axis=0)
    bmax=CCTrain.max(axis=0)
    PS=np.zeros(CCTrain.shape)
    for i in range(CCTrain.shape[1]):
        PS_temp=np.round((CCTrain[:,i]-bmin[i])*32/(bmax[i]-bmin[i]))
        PS_temp=[min(max(i,0),M) for i in PS_temp]
        PS[:,i]=PS_temp
    return PS,bmin,bmax




PSLDA,bminLDA,bmaxLDA=CC_PS(CCTrainLDA,M)


PSPCA,bminPCA,bmaxPCA=CC_PS(CCTrainPCA,M)


#use random forest, tree and knn for this classification problem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold




#use GridSearchCV to find the best set of model parameters
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV


#split training and testing set
X_train, X_test, y_train, y_test = train_test_split(PSLDA, trainY, test_size=0.2, random_state=0)
y_train=np.reshape(y_train,(1,len(y_train)))[0]
y_test=np.reshape(y_test,(1,len(y_test)))[0]




from sklearn.model_selection import cross_val_score
from sklearn import tree
#k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier



#decision tree algorithm
def treeAlgo(X_train,y_train):
    parameters = {'max_depth':range(3,8),'min_samples_leaf':range(10,30,5)}#tune maximum depth and the minimum number of leaves
    clfTree=GridSearchCV(tree.DecisionTreeClassifier(),parameters)
    #clfTree=tree.DecisionTreeClassifier()
    clfTree=clfTree.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfTree



clfTree=treeAlgo(X_train,y_train) #decision tree model


#decision tree prediction accuarcy on test data
sum(clfTree.predict(X_test)==y_test)/len(X_test)

#random forest algorithm
def RDForestAlgo(X_train,y_train):
    parameters = {'n_estimators':range(10,200,10),'min_samples_leaf':range(10,30,5)} #tune the number of trees and the minimum leaves
    clfRDForest=GridSearchCV(RandomForestClassifier(),parameters)
    clfRDForest=clfRDForest.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfRDForest

#random forest classification model
clfRDForest=RDForestAlgo(X_train,y_train)


#prediction accuary on test data 
sum(clfRDForest.predict(X_test)==y_test)/len(X_test)



#k nearest neighbor
def KNNAlgo(X_train,y_train):
    parameters = {'n_neighbors':range(3,10)}
    clfKNN=GridSearchCV(KNeighborsClassifier(),parameters)
    clfKNN=clfKNN.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfKNN


#k nearest neighbor
clfKNNAlgo=KNNAlgo(X_train,y_train)



sum(clfKNN.predict(X_test)==y_test)/len(X_test)


#model ensamble

#Now ensemble the three methods
treePredict=clfTree.predict(X_test)
RDPredict=clfRDForest.predict(X_test)
KNNPredict=clfKNN.predict(X_test)


from collections import Counter
Ypred=[]
for i in range(len(X_test)):
    Y_ensemble=Counter([treePredict[i],RDPredict[i],KNNPredict[i]])
    if Y_ensemble.most_common(1)[0][1]==1:
        most_common=RDPredict[i]
    else:
        most_common=Y_ensemble.most_common(1)[0][0]
    Ypred.append(most_common)
Ypred=[int(i) for i in Ypred]

#Final prediction of test data



