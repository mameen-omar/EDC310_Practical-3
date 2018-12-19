# Mohamed Ameen Omar
# 16055323
####################################
###      EDC 310 Practical 2     ###
###             2018             ###
###    MLSE- Viterbi Algotihm    ###
####################################

import numpy as np
from question_3 import BPSKmodulationSimulation
import copy

'''
MY IMPLEMENTATION


trellisNode store the time, the state, the next states, the alpha, the previous states of a node

first pass in the r,n,c

then it builds it by:
assigniedn fist node to 11 at t=0
then for every node, create its transitons and update all
including all alphas and deltas

to get path:
from left to right
check if there's contending, if not just add previous alpha to current
if there is, get alphas for each one, get smallest, remove all other deltas and continue until end 
'''

# Class to represent a single node in the Viterbi Trellis 
# used for the MLSE Equalizer
# Contains the time, the alpha value(sum of deltas along cheapest path to node),
# the state of the node, all previous states it is connected to and all next states (time > node's time)
# it is connected to
class trellisNode:
    def __init__(self,state,time):
        self.nextStates = []
        self.previousStates = []
        self.time = time
        self.state = state
        self.deltas = [] #delta array corresponding to the order of previous States
        self.alpha = 0
    #Add a new state that the current node is connected to.
    def addNext(self,state):
        self.nextStates.append(state)
    # Add a previous state that the node is connected to
    def addPrevious(self, state):
        self.previousStates.append(state)
####################### END OF CLASS ###########################

#class that conducts the MLSE algorithm for ANY BPSK modulated signal
# Requires:
# @param N = the number of data bits in the recieved signal
# @pram r = the recieved vector of bits
# @param c = the Channel Impulse response for the channel 

# it will build the trellis, thereafter calculate all deltas for all states or nodes in the trellis.
# then compute the cheapest path and disregard any nodes elliminated
class MLSE:
    def __init__(self, N = 0, r = [], c = []):
        self.n = N #data bits
        self.symbols = [1, -1]  # modulation symbols
        self.r = r #recieved symbols
        self.c = c #Convolution matrix
        self.trellisLength = self.n + len(self.c) - 1 #Trellis will be up to time T
        self.numStates = len(self.symbols)**(len(self.c) -1)
        self.numHT = len(c)-1
        self.Trellis = np.empty(shape=(self.numStates, self.trellisLength+1), dtype=trellisNode)
        self.detected = [] #detected symbols from trellis
        self.entireStream = [] #entire detected stream including head and tail 
        self.dataDetected = [] #data bits detected

    #print all properties of the problem to which MLSE is applied
    def printProperties(self):
        print("Number of data bits:", self.n)
        print("Signal recieved:", self.r)
        print("Convolution Matrix (C): ", self.c)
        print("Trellis Length (L): ", self.trellisLength)
        print("Number of states:", self.numStates)
        print("Number of Head and Tail symbols:", self.numHT)
        print("Modulation Symbols:", self.symbols)

    # Function to build the viterbi trellis
    # will begin with the first node state being = 1,1; following convention.
    # assigning fist node to 11 at t=0
    # then for every node, create its transitons and update all
    # including all alphas and deltas
    def buildTrellis(self, startState = [1,1]):
        self.Trellis[0][0] = trellisNode(startState, 0)
        numBits = len(startState)
        #for every node in the trellis compute the nodes it 
        # will transition to
        for t in range(0,self.trellisLength):
            for s in range(0,self.numStates):
                # check if this is none
                if self.Trellis[s][t] is None:
                    continue
                else:
                    #get all states:
                    allStates = []
                    tempState = []
                    #only upward transitions
                    if(t >= self.trellisLength-2):
                        tempState.append(1)
                        for x in range(0,numBits-1):
                            tempState.append(self.Trellis[s][t].state[x])
                        allStates.append(tempState)
                    #both 1,-1
                    else:
                        for x in range(0,len(self.symbols)):
                            tempState = []
                            tempState.append(self.symbols[x])
                            for y in range(0, numBits-1):
                                tempState.append(self.Trellis[s][t].state[y])
                            allStates.append(tempState)
                    #now we have all states
                    for index in range(0,len(allStates)):
                        newState = allStates[index]
                        stateIndex = self.getStateIndex(newState)
                        prevNode = self.Trellis[s][t]
                        if self.Trellis[stateIndex][t+1] is None:
                            self.Trellis[stateIndex][t+1] = trellisNode(newState,t+1)

                        newNode = self.Trellis[stateIndex][t+1]
                        prevNode.addNext(newNode)
                        delta = self.computeDelta(t+1,prevNode.state,newState)
                        newNode.deltas.append(delta)
                        newNode.alpha = delta
                        newNode.addPrevious(prevNode)

    # Function to print all the deltas for all the nodes in the Viterbi Trellis
    def printAllDeltas(self):
        for t in range(1,self.trellisLength+1):
            for s in range(0,self.numStates):
                myNode = self.Trellis[s][t]
                if(myNode is None):
                    continue
                for prev in range(0, len(myNode.previousStates)):
                    print("Delta", myNode.time, "from", self.getStateIndex(myNode.previousStates[prev].state), "to",self.getStateIndex(myNode.state), "is", myNode.deltas[prev] )            
            print()

    # Return the state index [1,1] = 0 and [1,-1] = 1 , and so forth
    def getStateIndex(self,state):
        if(state == [1,1]):
            return 0
        
        if(state == [1,-1]):
            return 1

        if(state == [-1, 1]):
            return 2

        if(state == [-1, -1]):
            return 3

    # Function to perform a delta calculation from state1 to state 2 @time = @param time
    #state 1 is the state originating, state 2 is the state going to
    def computeDelta(self,time,state1,state2):
        temp = 0        
        for x in range(0,len(self.c)):
            if(x < len(state2)):
                temp += self.c[x]*state2[x]
            else:
                temp += self.c[x]*state1[x-len(state2) +1]
        temp = np.abs(self.r[time-1] - temp)**2
        return temp
    
    # Function to calculate the cheapest path and in extension estimate the 
    # symbols recieved. Must build the trellis before calling this function
    # to get path:
    # from left to right
    # check if there's contending, if not just add previous alpha to current
    # if there is, get alphas for each one, get smallest, remove all other deltas and continue until end 
    def cheapestPath(self):
        for t in range(0,self.trellisLength+1):
            for s in range(0,self.numStates):
                if(self.Trellis[s][t] is None):
                    continue
                #if more than one state connected (contending)
                if(len(self.Trellis[s][t].previousStates) > 1):
                    #get smallest delta
                    #delete all other previous states, all other deltas, store complete path alpha
                    bestIndex = 0 #index of node with the best or shortest path so far
                    #just set to first
                    bestAlpha = self.Trellis[s][t].previousStates[0].alpha + self.Trellis[s][t].deltas[0]                 
                    for x in range(1, len(self.Trellis[s][t].previousStates)):
                        tempAlpha = self.Trellis[s][t].previousStates[x].alpha + self.Trellis[s][t].deltas[x]
                        if(tempAlpha < bestAlpha):
                            bestIndex = x
                            bestAlpha = tempAlpha
                    self.Trellis[s][t].alpha = bestAlpha
                    self.Trellis[s][t].deltas = [ self.Trellis[s][t].deltas[bestIndex] ]
                    self.Trellis[s][t].previousStates = [ self.Trellis[s][t].previousStates[bestIndex] ]
                else:
                    #compute the alpha
                    if(len(self.Trellis[s][t].previousStates)  == 0):
                        continue
                    self.Trellis[s][t].alpha += self.Trellis[s][t].previousStates[0].alpha     
        
        myTemp = self.Trellis[0][-1]
        self.detected = [myTemp.state[0]] + self.detected #this is just from the trellis        
        myTemp = myTemp.previousStates[0]
        while(myTemp.time> 0 ):
            self.detected = [myTemp.state[0]] + self.detected
            myTemp = myTemp.previousStates[0]        
        self.detected = [myTemp.state[0]] + self.detected #add t=0
        self.entireStream = copy.deepcopy(self.detected) #with head and tail
        while(len(self.entireStream) != (self.n+self.numHT+self.numHT)):
            self.entireStream = [1] + self.entireStream

        streamLength = len(self.entireStream)
        for x in range(0,streamLength):
            if(x < self.numHT or streamLength-x <= self.numHT):
                continue
            else:
                self.dataDetected.append(self.entireStream[x])
    
    # Start with building the trellis
    # have the deltas and all previous and next states for each node
    # compute the alpha values for all = total path cost including the current node so far
    '''
    go through from the last and add the remaining nodes in the Trellis that are connected
    '''                

################## END OF CLASS ###########################

