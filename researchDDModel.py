
# coding: utf-8

# In[1]:

cd C:\Users\Linsen\Desktop\researchDD_model


# In[2]:

import pandas as pd
import numpy as np


# In[3]:

detFiles=['DetOutput_7.txt','DetOutput_8.txt','DetOutput_9.txt','DetOutput_95.txt','DetOutput_OD2_19.txt',
         'DetOutput_11.txt','DetOutput_12.txt']


# In[4]:

linkFiles=['LinkOutput_7.txt','LinkOutput_8.txt','LinkOutput_9.txt','LinkOutput_95.txt','LinkOutput_OD2_19.txt',
          'LinkOutput_11.txt','LinkOutput_12.txt']


# In[5]:

dataDet=pd.read_csv(detFiles[0],header=None,sep=" ")
dataDet.columns=["detId","secId","count","occupany","density","speed","repId"]
sec=np.unique(dataDet['secId']) 
nbRep=len(np.unique(dataDet["repId"]))
trainX=np.zeros((nbRep*len(detFiles),len(sec)))
dataLink=pd.read_csv(linkFiles[0],header=None,sep=" ")
dataLink.columns=["secId","flow","TT","density","delay","speed","repId"]


# In[6]:

detAll=[]
for sec_i in sec:
    detAll.extend(np.where(dataLink["secId"]==sec_i)[0])
detAll=sorted(detAll)


# In[7]:

for i in range(len(detFiles)):
    dataDet=pd.read_csv(detFiles[i],header=None,sep=" ")
    dataDet.columns=["detId","secId","count","occupany","density","speed","repId"]
    dataLink=pd.read_csv(linkFiles[i],header=None,sep=" ")
    dataLink.columns=["secId","flow","TT","density","delay","speed","repId"]
    allTT=dataLink.iloc[detAll]["TT"]
    trainX[i*nbRep:(i+1)*nbRep,:]=np.reshape(allTT,(nbRep,len(sec)))
    


# In[8]:

trainY=np.ones((nbRep*len(detFiles),1))
for i in range(len(detFiles)):
    trainY[i*nbRep:(i+1)*nbRep,:]=i+1
nbGroup=len(detFiles)


# In[9]:

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
    #print ('after all')
        groupData=trainXnew[i*int(nk):(i+1)*int(nk,):]
        meanData=np.mean(groupData,axis=0)
        covData=np.cov(np.transpose(groupData))
        SW=SW+(nk-1)*covData
        SB=SB+nk*(np.dot(np.reshape(meanData-sampleMean,(nbFeature,1)),np.reshape(meanData-sampleMean,(1,nbFeature))))
    C=np.dot(np.linalg.inv(SW),SB)#matrix of our interest
    W, v=np.linalg.eig(C)#w is the eigenvalues, v is eigen vectors
    XProj=np.zeros((nbGroup-1,nbFeature))
    for i in range(nbGroup-1):
        XProj[i,:]=v[:,i]
    XProjFull=np.zeros((nbGroup-1,trainX.shape[1]))
    XProjFull[:,nonzeroCol]=XProj
    CCTrainLDA=np.transpose(np.dot(XProj,np.transpose(trainXnew)))
    return CCTrainLDA
    


# In[10]:

CCTrainLDA=LDAfeatures(trainX,nbGroup)


# In[11]:

def PCAfeatures(trainX,nbGroup):
    nonzeroCol=np.sum(trainX,0)!=0
    trainXnew=trainX[:,nonzeroCol]
    nbFeature=trainXnew.shape[1]
    C=np.cov(np.transpose(trainXnew))
    W, v=np.linalg.eig(C)#w is the eigenvalues, v is eigen vectors
    XProj=np.zeros((nbGroup,nbFeature))
    for i in range(nbGroup):
        XProj[i,:]=v[:,i]
    XProjFull=np.zeros((nbGroup,trainX.shape[1]))
    XProjFull[:,nonzeroCol]=XProj
    CCTrainPCA=np.transpose(np.dot(XProj,np.transpose(trainXnew)))
    return CCTrainPCA


# In[12]:

CCTrainPCA=PCAfeatures(trainX,nbGroup)


# In[13]:

M=32 #rounding and scaling factor


# In[14]:

#CC_PS_LDA
def CC_PS(CCTrain,M):
    bmin=CCTrain.min(axis=0)
    bmax=CCTrain.max(axis=0)
    PS=np.zeros(CCTrain.shape)
    for i in range(CCTrain.shape[1]):
        PS_temp=np.round((CCTrain[:,i]-bmin[i])*32/(bmax[i]-bmin[i]))
        PS_temp=[min(max(i,0),M) for i in PS_temp]
        PS[:,i]=PS_temp
    return PS,bmin,bmax


# In[15]:

PSLDA,bminLDA,bmaxLDA=CC_PS(CCTrainLDA,M)


# In[16]:

PSPCA,bminPCA,bmaxPCA=CC_PS(CCTrainPCA,M)


# In[17]:

#use random forest, tree and knn for this classification problem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# In[ ]:




# In[18]:

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV


# In[19]:

X_train, X_test, y_train, y_test = train_test_split(PSLDA, trainY, test_size=0.2, random_state=0)
y_train=np.reshape(y_train,(1,len(y_train)))[0]
y_test=np.reshape(y_test,(1,len(y_test)))[0]


# In[20]:

from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import confusion_matrix
#k nearest neighbor
from sklearn.neighbors import KNeighborsClassifier


# In[21]:

def treeAlgo(X_train,y_train):
    parameters = {'max_depth':range(3,8),'min_samples_leaf':range(10,30,5)}
    clfTree=GridSearchCV(tree.DecisionTreeClassifier(),parameters)
    #clfTree=tree.DecisionTreeClassifier()
    clfTree=clfTree.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfTree


# In[22]:

clfTree=treeAlgo(X_train,y_train)


# In[23]:

#decision tree prediction accuarcy
sum(clfTree.predict(X_test)==y_test)/len(X_test)


# In[24]:

def RDForestAlgo(X_train,y_train):
    parameters = {'n_estimators':range(10,200,10),'min_samples_leaf':range(10,30,5)}
    clfRDForest=GridSearchCV(RandomForestClassifier(),parameters)
    clfRDForest=clfRDForest.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfRDForest


# In[25]:

clfRDForest=RDForestAlgo(X_train,y_train)


# In[34]:

sum(clfRDForest.predict(X_test)==y_test)/len(X_test)


# In[36]:

#k nearest neighbor
def KNNAlgo(X_train,y_train):
    parameters = {'n_neighbors':range(3,10)}
    clfKNN=GridSearchCV(KNeighborsClassifier(),parameters)
    clfKNN=clfKNN.fit(X_train,np.reshape(y_train,(1,len(y_train)))[0])
    return clfKNN


# In[37]:

clfKNNAlgo=KNNAlgo(X_train,y_train)


# In[39]:

sum(clfKNN.predict(X_test)==y_test)/len(X_test)


# In[40]:

#model ensamble
clfKNN.predict(X_test)


# In[41]:

clfRDForest.predict(X_test)


# In[43]:

#Now ensemble the three methods
treePredict=clfTree.predict(X_test)
RDPredict=clfRDForest.predict(X_test)
KNNPredict=clfKNN.predict(X_test)


# In[44]:

from collections import Counter
Ypred=[]
for i in range(len(X_test)):
#for i in range(1):
    Y_ensemble=Counter([treePredict[i],RDPredict[i],KNNPredict[i]])
    if Y_ensemble.most_common(1)[0][1]==1:
        most_common=RDPredict[i]
    else:
        most_common=Y_ensemble.most_common(1)[0][0]
    Ypred.append(most_common)
Ypred=[int(i) for i in Ypred]


# In[ ]:




# In[ ]:



