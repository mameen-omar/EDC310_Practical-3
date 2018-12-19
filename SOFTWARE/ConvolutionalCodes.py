# Mohamed Ameen Omar
# 16055323
####################################
###      EDC 310 Practical 3     ###
###             2018             ###
###      Convolutional Codes     ###
###           Simulation         ###
####################################

import numpy as np
from question_3 import BPSKmodulationSimulation
from Simulation import Simulation
import copy
import matplotlib.pyplot as plt
from MLSE import MLSE

# Class to represent a single node in the Viterbi Trellis
# used for the MLSE Equalizer
# Contains the time, the alpha value(sum of deltas along cheapest path to node),
# the state of the node, all previous states it is connected to and all next states (time > node's time)
# it is connected to
class trellisNode:
    def __init__(self, state, time):
        self.nextStates = []
        self.previousStates = []
        self.time = time
        self.state = state
        self.deltas = []  # delta array corresponding to the order of previous States
        self.alpha = 0
    #Add a new state that the current node is connected to.

    def addNext(self, state):
        self.nextStates.append(state)
    # Add a previous state that the node is connected to

    def addPrevious(self, state):
        self.previousStates.append(state)


######################### begin of class #######################


# Class to perform a simulation to determine the
# average BER for the Viterbi MLSE and DFE symbol
# estimation methods for Convolutional Encoders as 
# well as the performance of a linear Convolutional encoder
# taking into account the effects of Multipath with a linearly declining CIR
# The class takes in the number of data
# bits the generator polynomial, the constraint length and the number of Iterations to average the results over
# as constructor parameters
class convolutionSim:
    def __init__(self, n = 5, generator = [4,6,7], K = 3, numIter = 50):
        self.symbols = [1, -1]
        self.BpskSimulation = BPSKmodulationSimulation(n)
        self.K = K #constraint length
        self.n = n #uncoded data bits
        self.generator = generator #generator matrix - length of which is number of outputs
        self.numIter = numIter

    # Function to encode the given source bitStream using the 
    # given geneator polynomial with a Linear Convolutional encoding scheme
    # Returns the encoded parity bits
    def encode(self,generator = [4,6,7], bitStream = []):
        shiftRegister = []
        genBinary = []
        #get binary representation of generator
        for x in range(0, len(generator)):
            genBinary.append("{0:b}".format(generator[x]))

        #sort shift register to begin in state 0
        for x in range(0,len(genBinary[x])):
            shiftRegister.append(0) 
        # list to contain all parity bits
        parity = []
        #encode the data
        # for every bit in the bitStream, insert into shfit register
        # take shift reggister elements, multiply by genertor in binar%2
        for x in range(0,len(bitStream)):
            #print("Inserting", bitStream[x])
            shiftRegister.insert(0,bitStream[x])
            shiftRegister.pop()
            #print("Shift after insert:", shiftRegister)
            temp = 0
            for y in range(0,len(generator)):
                #print("Output bit ", y)
                temp = 0
                for z in range(0,len(shiftRegister)):
                    temp += shiftRegister[z]*int(genBinary[y][z]).__round__()
                #print("TEMP",temp)
                temp = temp % 2
                #print("temp after mod 2", temp)
                parity.append(temp)
                #print("Parity so far", parity)
        return parity

    #Function to build the Viterbi Trellis for decoding a Linear Convolutional encoded 
    # block of bits. 
    # The parity bits recieved are passed in as function paramters as well as the 
    # desired start state
    # Returns the constructed Viterbi Trellis
    def buildTrellis(self,parity,startState = [-1,-1]):
        #build the trellis
        Trellis = np.empty(shape=(2**(self.K-1), self.n+self.K), dtype=trellisNode)
       # print(len(Trellis))
        #print(len(Trellis[0]))
        Trellis[0][0] = trellisNode(startState, 0)
        #print("FIRST ONE", Trellis[0][0].state)
        numBits = len(startState)
        #for every node in the trellis compute the nodes it
        # will transition to
        for t in range(0,  self.n+self.K-1):
            for s in range(0, (2**(self.K-1))):
                #print(Trellis[0][0].state)
                # check if this is none
                if Trellis[s][t] is None:
                    continue
                else:
                    #get all states:
                    allStates = []
                    tempState = []
                    # #only upward transitions
                    if(t >= self.n+self.K-3):
                        #print("HERE")
                        tempState.append(-1)
                        for x in range(0, numBits-1):
                            tempState.append(Trellis[s][t].state[x])
                        allStates.append(tempState)

                    else:
                        for x in range(0, len(self.symbols)):
                            tempState = []
                            tempState.append(self.symbols[x])
                            #print(tempState, "TEMPSTATE")
                            for y in range(0, numBits-1):
                                tempState.append(Trellis[s][t].state[y])
                            allStates.append(tempState)
                    
                    #print(allStates)
                    #now we have all states
                    for index in range(0, len(allStates)):
                        #print("HERE")
                        #print(t)
                        newState = allStates[index]
                        #print(newState, "NEW STATE")
                        stateIndex = self.getStateIndex(newState)
                        prevNode = Trellis[s][t]
                        #print("Start Index", stateIndex)
                        if Trellis[stateIndex][t+1] is None:
                            Trellis[stateIndex][t +1] = trellisNode(newState, t+1)
                            #print("newStateHERE", newState)

                        newNode = Trellis[stateIndex][t+1]
                        prevNode.addNext(newNode)
                        delta = self.getDelta(t+1, prevNode.state, newState,parity)
                        newNode.deltas.append(delta)
                        newNode.alpha = delta
                        newNode.addPrevious(prevNode)
        return Trellis
    #Function to decode the recieved parity bits using the Viterbi MLSE algorithm for a Linear Convolutional encoded
    # block of bits.
    # The parity bits recieved are passed in as function paramters as well as the
    # desired start state
    # Returns the decoded source information
    def mlseDecode(self,parity,startState = [-1,-1]):
        Trellis = self.buildTrellis(parity,startState) 
        for t in range(0,  self.n+self.K):
            for s in range(0, (2**(self.K-1))):
                if(Trellis[s][t] is None):
                    continue
                #if more than one state connected (contending)
                if(len(Trellis[s][t].previousStates) > 1):
                    #get smallest delta
                    #delete all other previous states, all other deltas, store complete path alpha
                    bestIndex = 0 #index of node with the best or shortest path so far
                    #just set to first
                    bestAlpha = Trellis[s][t].previousStates[0].alpha + Trellis[s][t].deltas[0]                 
                    for x in range(0, len(Trellis[s][t].previousStates)):
                        tempAlpha = Trellis[s][t].previousStates[x].alpha + Trellis[s][t].deltas[x]
                        if(tempAlpha <= bestAlpha):
                            bestIndex = x
                            bestAlpha = tempAlpha
                    Trellis[s][t].alpha = bestAlpha
                    Trellis[s][t].deltas = [ Trellis[s][t].deltas[bestIndex] ]
                    Trellis[s][t].previousStates = [ Trellis[s][t].previousStates[bestIndex] ]
                else:
                    #compute the alpha
                    if(len(Trellis[s][t].previousStates)  == 0):
                        continue
                    Trellis[s][t].alpha += Trellis[s][t].previousStates[0].alpha
        myTemp = Trellis[0][-1]

        #print(myTemp.state)
        detected = [myTemp.state[0]]  #this is just from the trellis    
        #print("State left at time", myTemp.time, "is", myTemp.state)    
        myTemp = myTemp.previousStates[0]
        while(myTemp.time > 0):
            #print("State left at time", myTemp.time, "is", myTemp.state)
            detected = [myTemp.state[0]] + detected
            myTemp = myTemp.previousStates[0]        
        #detected = [myTemp.state[0]] + detected #add t=0
        #print("detected", detected)
        return detected

    # Return the state index [-1,-1] = 0 and [1,-1] = 1 , and so forth
    def getStateIndex(self, state):
        if(state == [1, 1]):
            return 3
        if(state == [1, -1]):
            return 2
        if(state == [-1, 1]):
            return 1
        if(state == [-1, -1]):
            return 0
    # Returns the delta value for time = time, from
    # state1 to state2 for the given parity bits recieved
    def getDelta(self,time,state1,state2,parityRec):
        #first need to get parity symbsols at time
        #get output symbols
        # print("THE PARITY AFTER SENT", parityRec)
        # print("Getting Delta for time", time)
        # print("State1", state1)
        # print("state2", state2)
        outputSymbols = self.getOutputSymbols(state1,state2)
        #print("Output Symbols", outputSymbols)
        recParity = []
        for x in range( (time-1)*len(self.generator), (time-1)*len(self.generator)+len(self.generator)):
            recParity.append(parityRec[x])
        temp = 0
        #print("Rec parity", recParity)        
        for x in range(0,len(recParity)):
            temp += np.abs(recParity[x]-outputSymbols[x])**2
        return temp 

   # Returns the output symbols as per the state diagram 
   # for a transition from state 1 to state2
    #state 1 is the orignial, state 2 is the new state   
    def getOutputSymbols(self,state1,state2):
        shiftRegister = copy.deepcopy(state1)
        #insert the first bit of state
        shiftRegister.insert(0, copy.deepcopy(state2[0]))
        # shiftRegister.pop()
        genBinary = []
        for x in range(0,len(shiftRegister)):
            if(shiftRegister[x] == -1):
                shiftRegister[x] = 0
        #get binary representation of generator
        for x in range(0, len(self.generator)):
            genBinary.append("{0:b}".format(self.generator[x]))
        # list to contain all parity bits
        parity = []        
        temp = 0
        #print("Shift register", shiftRegister)
        for y in range(0, len(self.generator)):            
            temp = 0
            for z in range(0, len(shiftRegister)):
                #print("temp +=" , shiftRegister[z], "*", int(genBinary[y][z]).__round__())
                temp += shiftRegister[z]*int(genBinary[y][z]).__round__()
            #print("TEMP",temp)
            temp = temp % 2
            if(temp == 0):
                temp = -1
            #print("temp after mod 2", temp)
            parity.append(temp)
            #print("Parity so far", parity)
        #print()
        return parity
    # Function to perform the simulation using the MLSE Viterbi algorhtm to decode
    # No multipath effects taken into account
    # Returns the BER list
    def SimulateMLSE(self):
        numIter = self.numIter
        print("Conducting a MLSE BPSK simulation with Convolutional Encoding")
        BER = []  # store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                myDataBits = self.BpskSimulation.generateBits()
                #append tail
                for x in range(0, self.K-1):
                    myDataBits.append(0)
                encodedBits = self.encode(self.generator, myDataBits)
                modulatedSignal = self.BpskSimulation.BpskModulate(encodedBits)
                signalSent = self.BpskSimulation.addNoise(SNR, modulatedSignal)
                decodedSymbols = self.mlseDecode(signalSent)
                decodedBits = self.BpskSimulation.BpskDemodulate(decodedSymbols)
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,decodedBits)/(self.n))
            BER.append((sum(tempBER)/len(tempBER)))
        print(BER)
        return BER
    # Function to perform the simulation using BPSK modulation, no multipath, no Convoltuional encoding 
    # Returns the BER list
    def simulateBPSK(self):
        numIter = self.numIter
        print("Conducting a BPSK simulation with no Convolutional Encoding")     
        BER = [] #store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):                
                # generate 300 bits
                myDataBits = self.BpskSimulation.generateBits() #raw data bits 
                # map bits to symbols
                modulatedSignal = self.BpskSimulation.BpskModulate(myDataBits)                                                
                signalSent = self.BpskSimulation.addNoise(SNR, modulatedSignal)
                detectedSignal = self.BpskSimulation.detectSignal(SNR, signalSent)
                # demodulate
                demodulatedSignal = self.BpskSimulation.BpskDemodulate(detectedSignal)
                # compare
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,demodulatedSignal)/self.n)               
            BER.append( (sum(tempBER)/len(tempBER) ))
        print(BER)
        return BER
    
    #Function to decode the recieved parity bits using the DFE algorithm for a Linear Convolutional encoded
    # block of bits.
    # The parity bits recieved are passed in as function paramters as well as the
    # desired start state
    # Returns the decoded source information
    def dfeDecode(self,parity,startState = [-1,-1]):
        detected = []
        detectedState = startState
        #detected.append(detectedState[0])
        for x in range(0,self.n+self.K):
            tempState1 = detectedState
            if(self.getDelta(x, tempState1, [-1, tempState1[0]], parity) > self.getDelta(x, tempState1, [1, tempState1[0]], parity)):
                detectedState = [1, tempState1[0]]
            else:
                detectedState = [-1, tempState1[0]]
            detected.append(detectedState[0])
        return detected[1:]

    # Function to perform the simulation using the DFE algorhtm to decode
    # No multipath effects taken into account
    # Returns the BER list
    def SimulateDfe(self):
        numIter = self.numIter
        print("Conducting a DFE BPSK simulation with Convolutional Encoding")
        BER = []  # store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                myDataBits = self.BpskSimulation.generateBits()
                #print("Data Bits:", myDataBits)
                #append tail
                for x in range(0, self.K-1):
                    myDataBits.append(0)
                #print("Data Bits after append:", myDataBits)
                #print("Encoding")
                encodedBits = self.encode(self.generator, myDataBits)
                #print("Encoded bits", encodedBits)
                modulatedSignal = self.BpskSimulation.BpskModulate(encodedBits)
                #print("Modulated Encoded", modulatedSignal)
                signalSent = self.BpskSimulation.addNoise(SNR, modulatedSignal)
                decodedSymbols = self.dfeDecode(signalSent)
                #print("Decoded Symbols", decodedSymbols)
                decodedBits = self.BpskSimulation.BpskDemodulate(decodedSymbols)
                #print("decoded Bits", decodedBits)

                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,decodedBits)/(self.n))
            BER.append((sum(tempBER)/len(tempBER)))
        print(BER)
        return BER
    # Function to perform the simulation using the MLSE Viterbi algorhtm to decode
    # Multipath effects taken into account
    # Returns the BER list
    def simulateMultiPath(self):
        numIter = self.numIter
        multiSim = Simulation(n=self.n*len(generator), linC=[0.89, 0.42, 0.12])
        print("Conducting a Multipath BPSK simulation with Convolutional Encoding")
        BER = []  # store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                myDataBits = self.BpskSimulation.generateBits()
                #print("Original Data", myDataBits)
                #append tail
                for x in range(0, self.K-1):
                    myDataBits.append(0)
                #print("ROIGNAL WITH APPEND", myDataBits)
                encodedBits = self.encode(self.generator, myDataBits)
                #print("Encoded", encodedBits)
                modulatedSignal = self.BpskSimulation.BpskModulate(encodedBits)
                convolutedSignal = multiSim.channelModification(multiSim.linC, modulatedSignal)
                signalSent = self.BpskSimulation.addNoise(SNR, convolutedSignal)
                myMLSE = MLSE(self.n*len(generator), signalSent, [0.89, 0.42, 0.12])
                myMLSE.buildTrellis()
                myMLSE.cheapestPath()
                # get data bits
                dataDetected = myMLSE.dataDetected
                #print("Data Detected", dataDetected)
                decodedSymbols = self.mlseDecode(signalSent)
                decodedBits = self.BpskSimulation.BpskDemodulate(decodedSymbols)
                #print("Decoded bits", decodedBits)
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,decodedBits)/(self.n))
            BER.append((sum(tempBER)/len(tempBER)))
        print(BER)
        return BER
    # Fucntion to perform and plot all the implemented simulations on a single curve
    def plotAll(self):
        print("Plotting all for", self.numIter, "iterations")
        SNR = np.linspace(-4, 8, 13)
        MlseConv = self.SimulateMLSE()
        bpskSim = self.simulateBPSK()
        dfeConv = self.SimulateDfe()
        mlseMultiPath = self.simulateMultiPath()
        plt.semilogy(SNR, MlseConv, 'r-', label='Viterbi MLSE Convolutional Encoder')
        plt.semilogy(SNR, bpskSim, 'g-', label='BPSK Optimal Detection')
        plt.semilogy(SNR, dfeConv, 'b-', label='DFE Convolutional Encoder')
        plt.semilogy(SNR, mlseMultiPath, 'y-', label='Viterbi MLSE Convolutional Encoder with Multipath')
        plt.grid(which='both')
        plt.ylabel("BER")
        plt.xlabel("SNR")
        title = "Plot of the BER vs SNR for the BPSK Simulation conducted investigating Convolutional Encoders with " + str(self.numIter) + " iterations."
        plt.title(title)
        plt.legend(loc='best')
        plt.show()
        print("Complete")
        return            
######################### end of class #######################

#uncomment last line to conduct a simulation of all simulations implemented
n = 100 #number unenconded bits
numIter = 20 #number of iterations to avergae the results over
generator = [4, 6, 7] #generator polynomial
K = 3 #constraint length
tempSim = convolutionSim(n,generator,K, numIter)
tempSim.plotAll()

    
