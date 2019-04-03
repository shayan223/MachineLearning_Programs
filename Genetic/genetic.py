import numpy as np
import random as rand
import matplotlib.pyplot as plot

####### Hyper Perameters ######################
popSize = 100 #population size
generations = 50 #number of generations to train
mutationChance = .50 #probability for mutations per weight
sizeOfMutation = 6.0 #percent size of jump a mutation can make (1 = 100%)
################################################
def breedChild(p1,p2):#returns child given two parents
    length = len(p1)
    numToChoose = length // 2
    indexMap = list(range(len(p1)))
    #pick random weights to swap
    chosenIndices = np.random.choice(indexMap,size=numToChoose,replace=False)
    
    child = np.copy(p1)#start child as copy of parent 1
    for i in chosenIndices:
        #replace half of the weights from p1
        child[i] = p2[i]
        #apply chance for mutation
        if(np.random.rand() < mutationChance):
            #determine if mutation moves positively or negatively
            mutation = child[i] * sizeOfMutation
            if(np.random.rand() < .5):#50% chance for + and -
                mutation *= -1
            child[i] += mutation
            
    return child

#########################################################
    
top20 = int(popSize * .2)#finds how many individuals 20% of the pop is (round down)
remainingPop = popSize - top20

data = np.genfromtxt(fname="data.csv",delimiter=",")

target = data[:,-1]
data = data[:,1:-1]

rows = data.shape[0]
columns = data.shape[1]

#generate initial population of weights
pop = np.random.uniform(-.1,.1,(columns,popSize))#(population) matrix of weight vectors  



xAxis = list(range(generations))
yAxis = [] #average accuracy of each generation

for gen in range(generations):
    print(gen)#keeps track of progress
    
    #run data through weights and store output results
    output = np.dot(data,pop)
    
    #fitness of each individual will be based on accuracy
    total = rows
    correct = np.zeros(popSize)
    #use output to classify
    for i in range(output.shape[1]):#for every population individual
        for j in range(output.shape[0]):#for every data entry
            if(output[j,i] < 0 and target[j] == 0):
                correct[i] += 1
            elif(output[j,i] >= 0 and target[j] == 1):
                correct[i] += 1
            
    accuracy = correct/total
    yAxis.append(np.average(accuracy))

    #create new population with most successfull parents
    
    topPerformers = np.flip(np.argsort(accuracy))    
    newPop = np.zeros((pop.shape))
    #print(newPop.shape)
    #print(pop.shape)
    #print(topPerformers.shape)
    for i in range(top20):
        newPop[:,i] = pop[:,topPerformers[i]]
    for i in range(remainingPop):#top parents will now breed the remaining population
        p1 = newPop[:,rand.randint(0,top20)]
        p2 = newPop[:,rand.randint(0,top20)]
        position = top20 + i#holds on to next index available in population
        newPop[:,position] = breedChild(p1, p2)
        
    pop = np.copy(newPop)
    
print("Population Size:", popSize)
print("Number of Generations:", generations)
print("Chance for Mutation:", mutationChance)
print("Magnitude of mutations:", sizeOfMutation)
print("Maximum Accuracy:",max(yAxis))

plot.xlabel('Generation')
plot.ylabel('Average Accuracy')      
plot.plot(xAxis,yAxis)










