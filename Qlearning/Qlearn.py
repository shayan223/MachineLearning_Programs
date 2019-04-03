import numpy as np
import random as rand
import matplotlib.pyplot as plot

def generateEnv():
    env = np.random.randint(2,size=(12,12))
    env[:,0] = 3
    env[0,:] = 3
    env[-1,:] = 3
    env[:,-1] = 3
    return env

def qFunc():#q function taking a state and an action
    pass

class agent:
    def __init__(self, environment):  
        self.N = 5000 #number of episodes (plot at every 100 episodes)
        self.M = 200#steps per episode
        self.n = .2#learning rate
        self.y = .9#discount rate
        self.e = .1#increase by .01 every 50 epochs until it reaches 1
        self.score = 0
        self.env = generateEnv()
        self.qmatrix = {}#dictionary holds values for each state
        #self.qmatrix = np.zeros((100,5))#100 states, 5 actions
        #position represents position of agent (random initial placement)
        self.position = (np.random.randint(1,11),np.random.randint(1,11)) 
        self.path = []#list of past positions
        
    def moveUp(self):#moves agent up on the grid
        if(self.env[self.position[0]-1,self.position[1]] == 3):
            self.score -= 5
            return -5
        else:
            newPosition = (self.position[0]-1,self.position[1])
            self.position = newPosition
            return 0
        
    def moveDown(self):#moves agent down on the grid
        if(self.env[self.position[0]+1,self.position[1]] == 3):
            self.score -= 5
            return -5
        else:
            newPosition = (self.position[0]+1,self.position[1])
            self.position = newPosition
            return 0
                
    def moveLeft(self):#moves agent left on the grid
        if(self.env[self.position[0],self.position[1]-1] == 3):
            self.score -= 5
            return -5
        else:
            newPosition = (self.position[0],self.position[1]-1)
            self.position = newPosition
            return 0
    
    def moveRight(self):#moves agent right on the grid
        if(self.env[self.position[0],self.position[1]+1] == 3):
            self.score -= 5
            return -5
        else:
            newPosition = (self.position[0],self.position[1]+1)
            self.position = newPosition
            return 0
    
    def checkSelf(self):
        return self.env[self.position[0],self.position[1]]
    def checkUp(self):
        return self.env[self.position[0]-1,self.position[1]]
    def checkDown(self):
        return self.env[self.position[0]+1,self.position[1]]
    def checkLeft(self):
        return self.env[self.position[0],self.position[1]-1]
    def checkRight(self):
        return self.env[self.position[0],self.position[1]+1]

    def checkCurrent(self):#return value at current grid tile
        return self.env[self.position]

    def returnState(self):#returns hash value string representing state of agent
        states = [self.checkSelf(),
                  self.checkUp(),
                  self.checkRight(),
                  self.checkDown(),
                  self.checkLeft()]
        
        x = ''.join(map(str,states))
        return x
    
    def pickup(self):
        if(self.checkCurrent() == 0):
            self.score -= 1
            return -1
        else:
            self.env[self.position] = 0
            self.score += 5
            return 5
            
    
    def actions(self,actionNum):#maps number input to actions
        if(actionNum == 0):
            return self.pickup()
        if(actionNum == 1):
            return self.moveUp()
        if(actionNum == 2):
            return self.moveRight()
        if(actionNum == 3):
            return self.moveDown()
        if(actionNum == 4):
            return self.moveLeft()
    
    def epsilonGreedy(self,e):#uses epsilon "e" returning 1 for greedy and 0 for explore
        if(np.random.rand() < e):
            return 1
        return 0
    
    def takeTrainingAction(self):
        currentState = self.returnState()
        if(currentState not in self.qmatrix):
            actionList = [0,0,0,0,0]#in order:pick up, move up, right, down, left
            #select action randomly because we have no prior information
            decision = rand.randint(0,4)
            result = self.actions(decision)#take random action
            resultingState = self.returnState()#hold on to new state info
            #if the resulting state is new, initialize its state info
            if(resultingState not in self.qmatrix):
                self.qmatrix[resultingState] = [0,0,0,0,0]
            resultingMax = max(self.qmatrix[resultingState])
            actionList[decision] += (self.n*(result + self.y*resultingMax) - actionList[decision])
            self.qmatrix[currentState] = actionList
        else:#current state has been visited before
            actionList = self.qmatrix[currentState]
            #repeat steps from above, only actionList has known Q values
            #and use epsilon greedy choice
            if(self.epsilonGreedy(self.e) == 1):#take greedy action
                decision = np.argmax(actionList)
                result = self.actions(decision)
            else:#take exploratory action
                decision = rand.randint(0,4)
                result = self.actions(decision)
                
            resultingState = self.returnState()
            if(resultingState not in self.qmatrix):
                self.qmatrix[resultingState] = [0,0,0,0,0]
            resultingMax = max(self.qmatrix[resultingState])
            actionList[decision] += (self.n*(result + self.y*resultingMax) - actionList[decision])
            self.qmatrix[currentState] = actionList
        
        return self.score
    
    def takeTestAction(self):#same as training action, only we do not change q scores
        currentState = self.returnState()
        if(currentState not in self.qmatrix):
            actionList = [0,0,0,0,0]#in order:pick up, move up, right, down, left
            #select action randomly because we have no prior information
            decision = rand.randint(0,4)
            result = self.actions(decision)#take random action
            resultingState = self.returnState()#hold on to new state info
            #if the resulting state is new, initialize its state info
            if(resultingState not in self.qmatrix):
                self.qmatrix[resultingState] = [0,0,0,0,0]
            
        else:#current state has been visited before
            actionList = self.qmatrix[currentState]
            #repeat steps from above, only actionList has known Q values
            #and use epsilon greedy choice
            if(self.epsilonGreedy(self.e) == 1):#take greedy action
                decision = np.argmax(actionList)
                result = self.actions(decision)
            else:#take exploratory action
                decision = rand.randint(0,4)
                result = self.actions(decision)
                
            resultingState = self.returnState()
            if(resultingState not in self.qmatrix):
                self.qmatrix[resultingState] = [0,0,0,0,0]
            
        
        return self.score
        
env = generateEnv()
robby = agent(env)
#un-comment these two prints to see robby's initial position/environment
#print(robby.env)
#print(robby.position)

#plot reward across episodes
xAxis =[]#episode number
yAxis = []#reward for that episode

for N in range(robby.N):#for N episodes
    episodeScore = 0
    for M in range(robby.M):#take M steps
        robby.takeTrainingAction()
    if(N%100 == 0):#plot every 100 episodes
        xAxis.append(N)#plot episode score
        yAxis.append(robby.score)
    
    robby.score = 0#reset robby's score
    robby.env = generateEnv()#reset environment
    #give robby another random start location
    robby.position = (np.random.randint(1,11),np.random.randint(1,11))
    if(N%50 == 0 and robby.e < .9):#increase epsilon value by .01 every 50 epochs
        robby.e += .01
plot.xlabel('Episode')
plot.ylabel('Score')    
plot.plot(xAxis,yAxis)

print("Max score durning training:", max(yAxis))

#######################
#Test robby on new environment with trained qmatrix
xAxis2 = []
yAxis2 = []
robby.e = .9
for N in range(robby.N):#for N episodes
    episodeScore = 0
    for M in range(robby.M):#take M steps
        robby.takeTestAction()
    if(N%100 == 0):#plot every 100 episodes
        xAxis2.append(N)#plot episode score
        yAxis2.append(robby.score)
    
    robby.score = 0#reset robby's score
    robby.env = generateEnv()#reset environment
    #give robby another random start location
    robby.position = (np.random.randint(1,11),np.random.randint(1,11))
    
plot.figure(2)
plot.xlabel('Episode')
plot.ylabel('Score')      
plot.plot(xAxis2,yAxis2)

print("Max test score:", max(yAxis2))
print("Test average score:", np.mean(yAxis2))
print("Test score tandard deviation:", np.std(yAxis2))






