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
the first column and replaced with ones,
with all of the data being shuffled
'''
data = np.genfromtxt(fname="mnist_train.csv",delimiter=",")
testData = np.genfromtxt(fname="mnist_test.csv",delimiter=",")
np.random.shuffle(data)#shuffles order of images
np.random.shuffle(testData)


epochs = 50
N = 0.1 #learning rate
momentum = 0.9#momentum value for back propegation
rows = data.shape[0] #60,000
columns = data.shape[1] #785

testRows = testData.shape[0]
testColumns = testData.shape[1]


targets = np.copy(data[:, 0])

data *= (1/255)#normalizing pixel values
data[:,0] = 1#setting bias inputs

testTargets = np.copy(testData[:, 0])
testData *= (1/255)
testData[:,0] = 1


#######################
'''Initialisation of the hidden and output layers
of the network'''
perceptronCount = 10
hiddenCount = 10
hiddenLayer = np.random.uniform(-.05, .05, (columns,hiddenCount))
outputLayer = np.random.uniform(-.05, .05, (hiddenCount+1,perceptronCount))

###################
'''More initialisation'''
epochCount = 1#counter for epochs run
graphX = []#used to plot epochs (x axis)
graphY = []#used to plot accuracy (y axis)
graphY2 = []#same but used for test data plotting
testvar = 0

    
for x in range(epochs):
    previousOuterDelta = np.zeros(( perceptronCount, hiddenCount+1), dtype='double')#h+1 x 10
    previousHiddenDelta = np.zeros((hiddenCount+1, columns), dtype='double')#785 x h
    totalCount = 0
    goodCount = 0
    

    i = 0
    while i < targets.size:
        currentImage = data[i]#take next image from data
        np.reshape(currentImage, (1,columns))#reshape vector for dot product
        
        hiddenOutput = np.dot(currentImage,hiddenLayer)#take dot product
                                                #to get h values for output layer
        

        #apply sigmoid function to each element in output
        #then forward propogate to output layer
        ###        
        hiddenOutput = (1/(1+(np.exp(np.outer(-1,hiddenOutput)))))              
        hiddenOutput = np.insert(hiddenOutput, 0, 1)#adds bias to hidden layer  
        
        ###
        finalOutput = np.dot(hiddenOutput, outputLayer)      
        finalOutput = (1/(1+(np.exp(np.outer(-1,finalOutput)))))  
        
        ###
        index = 0
        networkAnswer = np.argmax(finalOutput)
        #sigma <- x(1-x)(t-x)
        if(networkAnswer == targets[i]):#computes accuracy
            goodCount+=1
        totalCount+=1
        
        if(epochCount != 1):#if statement used to skip training on first epoch
            outLayerError = np.zeros(perceptronCount, dtype='double')#holds sigma values for 10 neurons
            index = 0#Beginning of first step of back propegation
            for p in outLayerError:#calculate and store error values for output layer
                if(targets[i] == index):#t = .9
                    outLayerError[index] = finalOutput[0,index]*(1-finalOutput[0,index])*(.9 - finalOutput[0,index])
                else:#t = .1
                    outLayerError[index] = finalOutput[0,index]*(1-finalOutput[0,index])*(.1 - finalOutput[0,index])
                index += 1
                    
            
            hiddenLayerError = np.zeros(hiddenCount+1, dtype='double')#holds sigma values for hidden units
                        
            index = 0    
            for l in hiddenLayerError:#back propegation for hidden layer
                hiddenLayerError[index] = hiddenOutput[index]*(1-hiddenOutput[index])*(np.dot(outLayerError, outputLayer[index, :]))
                index += 1
            
            ###update weights for output layer
            outLayerError = np.reshape(outLayerError,(outLayerError.size,1))
            hiddenOutput = np.reshape(hiddenOutput,(1,hiddenCount+1))
            
            currentOutDelta = (N * np.outer(outLayerError,hiddenOutput)) + (momentum*previousOuterDelta)
                    
            ###update weights for hidden layer
            
            hiddenLayerError = np.reshape(hiddenLayerError,(1, hiddenLayerError.size))
            currentImage = np.reshape(currentImage, (currentImage.size, 1))
            currentHiddenDelta = (N * np.outer(hiddenLayerError,currentImage)) + (momentum*previousHiddenDelta)
            
            '''
            Add delta weights to the originals and update
            the variable storing the last iterations weight changes
            '''
            previousOuterDelta = currentOutDelta
            previousHiddenDelta = currentHiddenDelta
                        
            currentHiddenDelta = np.delete(currentHiddenDelta, 0, 0)
                        
            outputLayer += np.transpose(currentOutDelta)
            hiddenLayer += np.transpose(currentHiddenDelta)
                   
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
    
    confusionMatrix = np.zeros(shape=(10,10))
    i = 0
    while i < testTargets.size:
        currentImage = testData[i]
        np.reshape(currentImage, (1,columns))
        
        hiddenOutput = np.dot(currentImage,hiddenLayer)
        hiddenOutput = (1/(1+(np.exp(np.outer(-1,hiddenOutput)))))
        
        hiddenOutput = np.insert(hiddenOutput, 0, 1)
        
        finalOutput = np.dot(hiddenOutput, outputLayer)
        finalOutput = (1/(1+(np.exp(np.outer(-1,finalOutput)))))

        networkAnswer = np.argmax(finalOutput)
        
            
        if(np.argmax(finalOutput) == testTargets[i]):
            testGoodCount += 1
        
        testTotalCount += 1   
    
        confusionMatrix[int(testTargets[i]), int(np.argmax(finalOutput))] += 1
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


