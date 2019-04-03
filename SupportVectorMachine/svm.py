import numpy as np
import sklearn.svm as sci
import sklearn.preprocessing as prep
import sklearn.metrics as metrics
import matplotlib.pyplot as plot

####### Experiment 1 below #########################

data = np.genfromtxt(fname="spambase.data",delimiter=",")

np.random.shuffle(data)

testData = data[0:2300, :]
data = data[2301:4601, :]

target = data[:,-1]
testTarget = testData[:,-1]#vectors with target classifications ('rows' long)

data = data[:,:-1]
testData = testData[:,:-1]

rows = data.shape[0] 
columns = data.shape[1]



testRows = data.shape[0] 
testColumns = data.shape[1]
 
data = prep.scale(data)
testData = prep.scale(testData)



svm = sci.SVC(kernel='linear')
svm.fit(data,target)
result = svm.predict(testData)
accuracy = metrics.accuracy_score(testTarget,result)
print("Accuracy: ", accuracy)
w = svm.coef_
funcScore = svm.decision_function(testData)
falsePos, truePos, threshold  = metrics.roc_curve(testTarget,funcScore)
recall = metrics.recall_score(testTarget,result)
print("Recall: ", recall)
precision = metrics.precision_score(testTarget,result)
print("Precision: ", precision)
plot.figure(1)
plot.plot(falsePos, truePos)


######## Experiment 2 below #################

w = np.abs(w)
orderedW = np.argsort(w)
orderedW = np.flip(orderedW)

print()
print("Most important feature: ", orderedW[0,0])
print("2nd most important feature: ", orderedW[0,1])
print("3rd most important feature: ", orderedW[0,2])
print("4th most important feature: ", orderedW[0,3])
print("5th most important feature: ", orderedW[0,4])


XaxisM = []
YaxisAcc = []
svm2 = sci.SVC(kernel='linear')
m = 1


newData = data[:,orderedW[0,0]]
newData = np.reshape(newData,(rows,1))
newData = np.insert(newData,-1,data[:,orderedW[0,1]],axis=1)


newTestData = testData[:,orderedW[0,0]]
newTestData = np.reshape(newTestData,(rows,1))
newTestData = np.insert(newTestData,-1,testData[:,orderedW[0,1]],axis=1)


while(m < columns):
    XaxisM.append(m)
    newData = np.insert(newData,-1,data[:,orderedW[0,m]],axis=1)
    newTestData = np.insert(newTestData,-1,testData[:,orderedW[0,m]],axis=1)
    svm2.fit(newData,target)
    newResult = svm2.predict(newTestData)
    newAcc = metrics.accuracy_score(testTarget, newResult)
    YaxisAcc.append(newAcc)
    m+=1
plot.figure(2)
plot.plot(XaxisM,YaxisAcc)
    

######## Experiment 3 below ##############
'''NOTE: experiment 3 is the same, only the 
indexes listed in orderedW are shuffled so that
indices of features would be in random order so that
drawing from it pulls an arbitrary feature rather than
the most/least important'''

np.random.shuffle(orderedW[0,:])
XaxisM = []
YaxisAcc = []
svm3 = sci.SVC(kernel='linear')
m = 1


newData = data[:,orderedW[0,0]]
newData = np.reshape(newData,(rows,1))
newData = np.insert(newData,-1,data[:,orderedW[0,1]],axis=1)


newTestData = testData[:,orderedW[0,0]]
newTestData = np.reshape(newTestData,(rows,1))
newTestData = np.insert(newTestData,-1,testData[:,orderedW[0,1]],axis=1)


while(m < columns):
    XaxisM.append(m)
    newData = np.insert(newData,-1,data[:,orderedW[0,m]],axis=1)
    newTestData = np.insert(newTestData,-1,testData[:,orderedW[0,m]],axis=1)
    svm3.fit(newData,target)
    newResult = svm3.predict(newTestData)
    newAcc = metrics.accuracy_score(testTarget, newResult)
    YaxisAcc.append(newAcc)
    m+=1
plot.figure(3)
plot.plot(XaxisM,YaxisAcc)
















