import numpy as np
import pandas as pd
#import random
import timeit
import matplotlib.pyplot as plot
import seaborn as sb

start = timeit.default_timer()

#Preprocessing data read in#
########################################
'''
array "data" will represent all read in data.
This will be represented as a
60,000 by 785 matrix, with the first
column being all 1's for out biases,
and the remaining 784 as a pixel each.

The targets have been preprocessed out of
the first column and replaced with ones
'''
data = np.genfromtxt(fname="mnist_train.csv",delimiter=",")
testData = np.genfromtxt(fname="mnist_test.csv",delimiter=",")
np.random.shuffle(data)#shuffles order of images
np.random.shuffle(testData)

epochs = 10
N = .01 #learning rate

rows = data.shape[0] #60,000
columns = data.shape[1] #784

testRows = testData.shape[0]
testColumns = testData.shape[1]


targets = np.copy(data[:, 0])

data *= (1/255)#normalizing pixel values
data[:,0] = 1#setting bias inputs

testTargets = np.copy(testData[:, 0])
testData *= (1/255)
testData[:,0] = 1


#######################
'''Creates a 785x10 matrix, 10 representing 
each perceptron, with 785 weights each, to be
dotted with the data matrix, resulting in a 
60,000 x 10 matrix, representing the network
result for each image'''
perceptronCount = 10
network = np.random.uniform(-.05, .05, (columns,perceptronCount))

###################
'''the vector of targets is now compared to 
the total result of the network for each image,
updating weights when the incorrect answer is
returned'''
epochCount = 1#counter for epochs run
graphX = []#used to plot epochs (x axis)
graphY = []#used to plot accuracy (y axis)
graphY2 = []#same but used for test data plotting
testvar = 0

for x in range(epochs):
    totalCount = 0
    goodCount = 0
    badCount = 0
    i = 0
    while i < targets.size:
        currentImage = data[i]#take next image from data
        np.reshape(currentImage, (1,columns))#reshape vector for dot product
        output = np.dot(currentImage,network)#take dot product
        
        index = 0
        
        for x in output:#normalize outputs to 0 and 1
            if x < 0:
                output[index] = 0
            else:
                output[index] = 1
            index += 1
        if(epochCount != 1):#to measure initial accuracy
            k = 0#represents the perceptron being evaluated
            for h in output:#compare to target, h being single perceptron output
                if (((targets[i] == k) and h != 1) or ((targets[i] != k) and h != 0)):
                    if targets[i] == k:#determines what values will be used for t
                        desired = 1
                    else:
                        desired = 0
                    
                    deltaW = (N*(desired-h))*data[i]#calculate change in weight
                    network[:,k] += deltaW        #Update weights
                k+=1
            
        if(np.argmax(output) == targets[i]):#for computing accuracy
            goodCount += 1
        else: 
            badCount += 1
            
        totalCount += 1    
        i+=1

    accuracy = (goodCount/totalCount)
    
    graphX.append(epochCount)#begins plotting points
    graphY.append(accuracy)
    print("Epoch", epochCount, "Accuracy: ", accuracy)
    epochCount +=1
    
    

################################################
    '''use test data on trained network
        essentially just a copy paste from
        above without updating weights, everything
        else is exactly the same but with different
        variable names to avoid conflicts'''
        
    testTotalCount = 0
    testGoodCount = 0
    testBadCount = 0
    confusionMatrix = np.zeros(shape=(10,10))
    i = 0
    while i < testTargets.size:
        currentImage = testData[i]
        np.reshape(currentImage, (1,columns))
        output = np.dot(currentImage,network)
        
        index = 0   
        for x in output:#normalize outputs to 0 and 1
            if x < 0:
                output[index] = 0
            else:
                output[index] = 1
            index += 1
            
            
        if(np.argmax(output) == testTargets[i]):
            testGoodCount += 1
            
        else: 
            testBadCount += 1
            
        testTotalCount += 1   
    
        confusionMatrix[int(testTargets[i]), int(np.argmax(output))] += 1
        i+=1
    
    
    accuracy = (testGoodCount/testTotalCount)
    graphY2.append(accuracy)
    print("Test Set Accuracy: ", accuracy)

################################################
end = timeit.default_timer()
print("Run time (seconds): ",end-start)#calculates total run time
plot.plot(graphX,graphY)#plots graph
plot.plot(graphX,graphY2)
plot.show()#displays graph
confMat = pd.DataFrame(confusionMatrix, range(10), range(10))#converts to data frame for confusion matrix
sb.heatmap(confMat, annot=True, fmt='g', cmap='Blues')#generates confusion matrix


