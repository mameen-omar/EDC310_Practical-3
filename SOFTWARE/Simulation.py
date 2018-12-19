# Mohamed Ameen Omar
# 16055323
####################################
###      EDC 310 Practical 2     ###
###             2018             ###
###          Simulation          ###
####################################

import numpy as np
from question_3 import BPSKmodulationSimulation
from MLSE import MLSE
import copy
from DFE import DFE
import matplotlib.pyplot as plt

# Class to perform a simulation of the 
# BER for the Viterbi MLSE and DFE symbol 
# estimation methods
# The class takes in the number of data 
# bits and the linear declining channel impulse response
# vector as constructor parameters
class Simulation:
    def __init__(self, n=300, linC=[0.89, 0.42, 0.12], numIterations=20):
        self.BpskSimulation = BPSKmodulationSimulation(n)
        self.linC = linC
        self.n = n
        self.numIterations = numIterations

    # Function to perform the Viterbi MLSE simulation
    # with a Gaussian random Channel Impulse Response
    # returns the BER array
    # simulation runs for @param self.numIterations, for each
    # SNR in the range (-4,9)
    def viterbiRandomCIR(self):
        numIter = self.numIterations
        print("Conducting a MLSE BPSK simulation with a Uniform Random CIR")     
        BER = [] #store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):                
                # generate 300 bits
                myDataBits = self.BpskSimulation.generateBits() #raw data bits 
                # map bits to symbols
                modulatedSignal = self.BpskSimulation.BpskModulate(myDataBits)
                # add tail
                for x in range(0,len(self.linC) -1):
                    modulatedSignal.append(1)
                # add convolution                
                randomCIR = self.generateRandomCIR(SNR)
                convolutedSignal = self.channelModification(randomCIR, modulatedSignal)
                signalSent = self.BpskSimulation.addNoise(SNR, convolutedSignal)               
                # check MLSE
                myMLSE = MLSE(self.n, signalSent, randomCIR)
                myMLSE.buildTrellis()
                myMLSE.cheapestPath()
                # get data bits
                dataDetected = myMLSE.dataDetected               
                # demodulate
                demodulatedSignal = self.BpskSimulation.BpskDemodulate(dataDetected)
                # compare
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,demodulatedSignal)/self.n)
                del myMLSE
                del modulatedSignal
                del convolutedSignal
                del demodulatedSignal
            BER.append( (sum(tempBER)/len(tempBER) ))
        return BER
    
    # Function to perform the Viterbi MLSE simulation
    # with a Linear Declining Channel Impulse Response
    # returns the BER array
    # simulation runs for @param self.numIterations, for each
    # SNR in the range (-4,9)
    def viterbiLinearCIR(self):
        numIter = self.numIterations
        print("Conducting a MLSE BPSK simulation with a linear declining CIR")     
        BER = [] #store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                
                # generate 300 bits
                myDataBits = self.BpskSimulation.generateBits() #raw data bits            
                # map bits to symbols
                modulatedSignal = self.BpskSimulation.BpskModulate(myDataBits)               
                # add tail
                for x in range(0,len(self.linC) -1):
                    modulatedSignal.append(1)                
                # add convolution  
                convolutedSignal = self.channelModification(self.linC, modulatedSignal)
                signalSent = self.BpskSimulation.addNoise(SNR, convolutedSignal)                
                # check MLSE
                myMLSE = MLSE(self.n, signalSent, self.linC)
                myMLSE.buildTrellis()
                myMLSE.cheapestPath()
                # get data bits
                dataDetected = myMLSE.dataDetected                
                # demodulate
                demodulatedSignal = self.BpskSimulation.BpskDemodulate(dataDetected)
                # compare
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,demodulatedSignal)/self.n)
                del myMLSE
                del modulatedSignal
                del convolutedSignal
                del demodulatedSignal
            BER.append( (sum(tempBER)/len(tempBER) ))
        return BER

    # Function to perform the DFE simulation
    # with a Linear Declining Channel Impulse Response
    # returns the BER array
    # simulation runs for @param self.numIterations, for each
    # SNR in the range (-4,9)
    def dfeLinearCIR(self):
        numIter = self.numIterations
        print("Conducting a DFE BPSK simulation with a linear declining CIR")     
        BER = [] #store the average BER for each SNR
        for SNR in range(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                # generate 300 bits
                myDataBits = self.BpskSimulation.generateBits() #raw data bits                     
                # map bits to symbols
                modulatedSignal = self.BpskSimulation.BpskModulate(myDataBits)                  
                # add convolution  
                convolutedSignal = self.channelModification(self.linC, modulatedSignal)               
                signalSent = self.BpskSimulation.addNoise(SNR, convolutedSignal)
                # check DFE
                myDFE = DFE(self.n, signalSent, self.linC)
                # get data bits
                dataDetected = myDFE.getDataSymbols()
                # demodulate
                demodulatedSignal = self.BpskSimulation.BpskDemodulate(dataDetected)
                # compare
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,demodulatedSignal)/self.n)
                del myDFE
                del modulatedSignal
                del convolutedSignal
                del demodulatedSignal
            BER.append((sum(tempBER)/len(tempBER)))
        return BER    

    # Function to perform the DFE simulation
    # with a Gaussian random Channel Impulse Response
    # returns the BER array
    # simulation runs for @param self.numIterations, for each 
    # SNR in the range (-4,9)
    def dfeRandomCIR(self):
        numIter = self.numIterations
        print("Conducting a DFE BPSK simulation with a Uniform Random CIR")     
        BER = [] #store the average BER for each SNR
        for SNR in np.arange(-4, 9): #9
            tempBER = [] #to store all the interations for one SNR
            for count in range(0, numIter):
                # generate 300 bits
                myDataBits = self.BpskSimulation.generateBits() #raw data bits 
                # map bits to symbols
                modulatedSignal = self.BpskSimulation.BpskModulate(myDataBits)
                # add convolution                
                randomCIR = self.generateRandomCIR(SNR)
                convolutedSignal = self.channelModification(randomCIR, modulatedSignal)
                signalSent = self.BpskSimulation.addNoise(SNR, convolutedSignal)               
                # check DFE
                myDFE = DFE(self.n, signalSent, randomCIR)               
                # get data bits
                dataDetected = myDFE.getDataSymbols()               
                # demodulate
                demodulatedSignal = self.BpskSimulation.BpskDemodulate(dataDetected)
                # compare
                tempBER.append(self.BpskSimulation.getNumErrors(myDataBits,demodulatedSignal)/self.n)
                del myDFE
                del modulatedSignal
                del convolutedSignal
                del demodulatedSignal
            BER.append( (sum(tempBER)/len(tempBER) ))
        return BER

    #returns a random CIR with "values" elements
    # Reqires the SNR ratio of the channel
    # uses the Gaussian random number generator implemented in practical 1.
    def generateRandomCIR(self, SNR, values=3):
        c = []
        for x in range(0,values):
            temp = self.BpskSimulation.numberGenerator.gaussian(stdDeviation=self.BpskSimulation.getStdDev(SNR))
            temp = temp/(np.sqrt(3))
            c.append(temp)
            temp = 0
        return c
    
    # pass in just the symbols.
    # return symbols with channel repsonse.
    # function to add the effects of the channel to the stream of
    # symbols being sent. (Apply the CIR)
    def channelModification(self, cir, stream):
        returnStream = copy.deepcopy(stream)
        for x in range(0,len(stream)):            
            temp1 = 0 #sk-2
            temp2 = 0 #sk-1
            if(x-2 < 0):
                temp1 = 1
            else:
                temp1 = stream[x-2]
            if(x-1 <0):
                temp2 = 1
            else:
                temp2 = stream[x-1]
            returnStream[x] = (stream[x]*cir[0] + temp2*cir[1] + temp1*cir[2])
        return returnStream
    

    # Function to perform the simulation of all 4 situations
    # DFE linear CIR, DFE Gaussian Random CIR, MLSE linear CIR and MLSE Gaussian Random CIR.
    # Plots all four simulations on the same plot
    def simulate(self):
        print("Plotting all")
        SNR = np.linspace(-4, 8, 13)
        viterbiLinDec = self.viterbiLinearCIR()
        viterbiRandom = self.viterbiRandomCIR()
        dfeRandom = self.dfeRandomCIR()
        dfeLin = self.dfeLinearCIR()
        plt.semilogy(SNR, viterbiLinDec, 'r-', label='Viterbi MLSE Linear Declining CIR')
        plt.semilogy(SNR, viterbiRandom, 'g-', label='Viterbi MLSE Random CIR')
        plt.semilogy(SNR, dfeRandom, 'b-', label='DFE Random CIR')
        plt.semilogy(SNR, dfeLin, 'y-', label='DFE Linear Declining CIR')
        plt.grid(which='both')
        plt.ylabel("BER")
        plt.xlabel("SNR")
        plt.title("Plot of the BER vs SNR for the BPSK Simulation conducted with Viterbi MLSE and DFE Equalizers")
        plt.legend(loc='best')
        plt.show()
        print("Complete")

# n = number of data bits (300)
# c = linear declining CIR
# numIterations = number of iterations to take the average of, before plotting - adjust as needed.
# if commented - uncomment the last line 
# "mySim.simulate()" to run the simulation and retrieve the plot
# n = 300
# c = [0.89,0.42,0.12]
# numIterations = 50
# mySim = Simulation(n,c, numIterations)
# mySim.simulate()
