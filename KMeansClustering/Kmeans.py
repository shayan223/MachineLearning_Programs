import numpy as np
import math
import pandas as pd
import seaborn as sb
from PIL import Image
###################################
def euclidDist(x,y):
    return (np.sum((x-y)**2))**.5
    
###################################
def kmeans(numOfClusters):
    
        
    K = numOfClusters
    
    data = np.genfromtxt(fname="optdigits.train",delimiter=",")
    target = data[:,-1]
    data = data[:,:-1]
    rows = data.shape[0]
    columns = data.shape[1]
    
    
    centroidList = np.empty((K,data.shape[1]))
    
    #print(centerList.shape)
    
    index = np.random.choice(range(rows),K,replace=False)
    
    for i in range(K):
        centroidList[i,:] = data[index[i]]
        
    
    clusterMap = np.empty(rows)#maps points to their respective clusters/centroids
    
    prevCentList = np.empty((centroidList.shape))
    iterCount = 0
    
    while(np.array_equal(prevCentList[0], centroidList[0]) == False):
        prevCentList = np.copy(centroidList)
        
        for i in range(rows):#cluster all data points
            closest = 0#index of closest centroid
            distFromClosest = euclidDist(data[i,:],centroidList[0])#current closest distance
            
            for j in range(K):#check every centroid
                checkDist = euclidDist(data[i,:],centroidList[j])#dist from current centroid
                if(checkDist < distFromClosest):#if closer than current best, update
                    closest = j
                    distFromClosest = checkDist
                    
            clusterMap[i] = closest#update mapping for current point
            
             
        #re-calculate every centroid
        for i in range(K):#loops for every cluster
            total = np.zeros(columns)
            count = 0
            #find the average of the current cluster
            for j in range(rows):#loop through every data point
                if(clusterMap[j] == i):#if the j'th data point is in i'th cluster
                    total += data[j]
                    count += 1
            
            mean = total/count
            centroidList[i] = mean #update centroid to mean of its cluster
            
        
        iterCount += 1
        
        
        
    #find mean square error of each cluster
    mseList = []
    
    for c in range(K):#do this for each cluster c
        summation = 0
        clusterSize = 0
        for x in range(rows):#compare each point with the cluster center
            if(clusterMap[x] == c):
                summation += (euclidDist(data[x],centroidList[c])**2)
                clusterSize += 1
        mseList.append((summation/clusterSize))
        
    
    #find average mean square error across all clusters
    avgMse = (sum(mseList)/K)
    
    #find entropy of clusters, starting with finding the probability of 
    #each element within each cluster
    
    probClustList = []#holds one list of target classes for each cluster
    for c in range(K):
        cluster = []#holds the classification of every element in cluster
        for i in range(rows):
            if(clusterMap[i] == c):
                cluster.append(target[i])
        probClustList.append(cluster)
            
    
    entropyList = []
    for c in range(K):
        summation = 0          
        clusterSize = len(probClustList[c])#holds num. of elem. in cluster
        elementCount = np.zeros(int(np.amax(target)+1))#holds how many of each element
        for i in range(len(probClustList[c])):
            elementCount[int(probClustList[c][i]-1)] += 1
        for i in range(len(elementCount)):
            prob = elementCount[i]/clusterSize
            if(prob > 0):
                summation += (prob)*math.log2(prob)
        summation *= -1
        entropyList.append(summation)
        
    #compute mean entropy
    meanEntropy = 0
    for c in range(K):
        clusterSize = len(probClustList[c])
        meanEntropy += (clusterSize/rows)*entropyList[c]
        
    
    #find mean square seperation
    summation = 0
    for i in range(K):
        for j in range(K):
            if(np.array_equal(centroidList[i],centroidList[j]) == False):
                summation += (euclidDist(centroidList[i],centroidList[j])**2)
                
    meanSquareSep = summation/((K*(K-2))/2)
    
    returnList = [centroidList, 
                  avgMse, 
                  meanSquareSep, 
                  meanEntropy, 
                  clusterMap, 
                  target, 
                  probClustList]
    return returnList    
  
###############################    

K = 30
trials = 5
kMeansList = []
bestClustering = 0#holds index of the best Kmeans clustering
for i in range(trials):
    kMeansList.append(kmeans(K))
    if(kMeansList[i][1] < kMeansList[bestClustering][1]):
        bestClustering = i

finalClustering = kMeansList[bestClustering]
print("Average mse:", finalClustering[1])
print("Mean Square Seperation:", finalClustering[2])   
print("Mean Entropy:", finalClustering[3])

#######################

#Classify the test data
data = np.genfromtxt(fname="optdigits.test",delimiter=",")
target = data[:,-1]
data = data[:,:-1]
rows = data.shape[0]
columns = data.shape[1]

clusterClassMap = []
centroids = finalClustering[0]
trainCluster = finalClustering[4]
trainTarget = finalClustering[5]
classOcurrences = finalClustering[6]

#Assoiciate each cluster with their most frequent class
for c in range(K):
    mostCommonClass = np.bincount(classOcurrences[c]).argmax()
    clusterClassMap.append(mostCommonClass)

#calculate accuracy
total = rows
correct = 0
numOfClasses = int(np.amax(target)+1)
confusionMatrix = np.zeros((numOfClasses,numOfClasses))
clustersWithPoints = np.zeros(K)#keeps track of which clusters are empty

for i in range(rows):#cluster all data points
    closest = 0#index of closest centroid
    distFromClosest = euclidDist(data[i,:],centroids[0])#current closest distance
            
    for j in range(K):#check every centroid
        checkDist = euclidDist(data[i,:],centroids[j])#dist from current centroid
        if(checkDist < distFromClosest):#if closer than current best, update
            closest = j
            distFromClosest = checkDist
    if(target[i] == clusterClassMap[closest]):
        correct += 1
        clustersWithPoints[closest] += 1#indicate this cluster is not empty
    confusionMatrix[int(target[i]),int(clusterClassMap[closest])] += 1

accuracy = correct/total
print("Accuracy:",accuracy)

#make and display a confusion matrix
confusionMatrix = pd.DataFrame(confusionMatrix, range(numOfClasses), range(numOfClasses))#converts to data frame for confusion matrix
sb.heatmap(confusionMatrix, annot=True, fmt='g', cmap='Blues')#generates confusion matrix

sortedClusters = np.argsort(clustersWithPoints)

    
#normalise image values to between 0 and 255
centroids *= 255/centroids.max()
#change to uint8 from float to be compatible with visualization tool
centroids = centroids.astype(np.uint8)
#list to keep track of what classes have already been displayed
visited = []
i = -1
j = 1
while(j <= 10):#save top 10 clusters as images
    while(clusterClassMap[sortedClusters[i]] in visited):#comment this loop out for K <= 10
        i -=1
    visual = Image.fromarray(np.reshape(centroids[sortedClusters[i]],(8,8)),'L')#convert vector to grayscale image
    visual.save('Centroid' + str(abs(i)) +'Visual.png')#save image
    visited.append(clusterClassMap[sortedClusters[i]])
    i -= 1
    j += 1

#print(visited)










