import numpy as np
import math
import seaborn as sb
import pandas as pd

def gnb(x,mu,lam):#gaussian naive bayes function
    a = 1/(math.sqrt(2*math.pi)*lam)
    b = -(((x-mu)**2)/(2*(lam**2)))
    return a*(np.exp(b))

def classify(p1, p0, instance, muPos, muNeg, lamPos, lamNeg):
    epsilon = 0.00000000000000000000001
    
    pos = math.log10(p1)
    pos += np.sum(np.log10(gnb(instance,muPos,lamPos)+epsilon))
        
    neg = math.log10(p0)
    neg += np.sum(np.log10(gnb(instance,muNeg,lamNeg)+epsilon))
        
    if(pos > neg):
        return 1
    else:
        return 0

dataA = np.genfromtxt(fname="instanceA.data",delimiter=",")
test = np.genfromtxt(fname="instanceB.data",delimiter=",")

rows = dataA.shape[0]
columns = dataA.shape[1]
noTarget = columns - 1 # column count excluding target column

priorA = dataA[:,-1]
ones = np.count_nonzero(priorA)

p1 = ones/(len(priorA))#P(1) (prob. of spam)
p0 = (len(priorA)-ones)/len(priorA)#P(0) (prob. of not spam)

probMatrix = np.empty((rows, noTarget))
dataApos = dataA[:907, :]#devide spam and not spam
dataAneg = dataA[908:, :]

epsilon = 0.0001
###################################
meanApos = np.empty((noTarget))#average for positive
for i in range(noTarget):
    meanApos[i] = np.mean(dataApos[:,i]) 


stdApos = np.empty((noTarget))#standard deviation for positive
for i in range(noTarget):
    stdApos[i] = np.std(dataApos[:,i])
    
##################################  
meanAneg = np.empty((noTarget))#average for negative
for i in range(noTarget):
    meanAneg[i] = np.mean(dataAneg[:,i]) 


stdAneg = np.empty((noTarget))#standard deviation for negative
for i in range(noTarget):
    stdAneg[i] = np.std(dataAneg[:,i])
       
###################################
confusionMatrix = np.zeros(shape=(2,2))#holds confusion matrix

target = test[:,-1]#seperate target column from atributes
test = test[:,:-1]

right = 0
total = 0
for i in range(rows):
    answer = classify(p1,p0,test[i,:],meanApos, meanAneg, stdApos,stdAneg)
    if(answer == target[i]):#keep track of correct classifications
        right += 1
    confusionMatrix[int(target[i]),answer] += 1#update confusion matrix
    total += 1
    
    
accuracy = right/total
print("Accuracy: ",accuracy)
confMat = pd.DataFrame(confusionMatrix, range(2), range(2))#converts to data frame for confusion matrix
sb.heatmap(confMat, annot=True, fmt='g', cmap='Blues')#generates confusion matrix
###################################



















