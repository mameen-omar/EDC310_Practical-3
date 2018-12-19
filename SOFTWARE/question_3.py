#Mohamed Ameen Omar
#16055323
####################################
###      EDC 310 Practical 1     ###
###             2018             ###
###         Question 3           ###
####################################

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from scipy.stats import norm
from question_1 import randomNumber 

# Ensure that the source file question_1.py is in the current dorectory
# before running the script
 
# Class used for QPSK simulation
class QPSKmodulationSimulation:
    def __init__(self,numBits):
        self.numBits = numBits
        self.numberGenerator = randomNumber()
    
    def getStdDev(self, SNR):
        return (1/np.sqrt(10**(SNR/10)*2*2))

    #add noise for the real and imaginary parts of the symbol
    def addNoise(self,SNR,sentBits):
        recieved = []
        standardDeviation = self.getStdDev(SNR)
        for x in range(0,len(sentBits)):
            real = self.numberGenerator.gaussian(stdDeviation=standardDeviation)
            imag = self.numberGenerator.gaussian(stdDeviation=standardDeviation) * 1j
            recieved.append(sentBits[x]+real+imag)

        return recieved

    # generate random number using uniform random number generator
    # round the output to a 1 or 0 for a bit
    def generateBits(self):
        print("Generating",self.numBits, "random binary digits")
        toSend = []
        for x in range(0,self.numBits):
            toSend.append(int(np.round(self.numberGenerator.WHill())))
        return(toSend)

    # map an array of bits to a symbols
    # according to the BPSK constellation map
    def QpskModulate(self, original):
        print("Mapping bits to symbols using QPSK modulation")
        mappedMessage = []
        temp = list(map(str,original))
        
        for x in range(0, len(temp),2):
            toSymbol = temp[x] + temp[x+1]
            mappedMessage.append(self.bitToSymbol(toSymbol))
        return mappedMessage

    # return the Symbol for the
    # the bit passed in
    def bitToSymbol(self, toMap):
        if(toMap == "00"):
            return 1
        elif(toMap == "01"):
            return 1j
        elif(toMap == "11"):
            return -1
        elif(toMap == "10"):
            return -1j

    # demodulate the detected symbols 
    # for QPSK
    def QpskDemodulate(self, modulated):
        print("Mapping symbols to bits for QPSK demodulation")
        demod = []
        for x in range(0, len(modulated)):
            temp = self.symbolToBit(modulated[x])
            demod.append(int(temp[0]))
            demod.append(int(temp[1]))        
        return demod

    # return the bits for the 
    # the symbol passed in
    def symbolToBit(self,symbol):
        if(symbol == 1):
            return "00"
        elif(symbol == 1j):
            return "01"        
        elif(symbol == -1):
            return "11"        
        elif(symbol == -1j):
            return "10"
    
    # given the SNR and recieved signal
    # use optimum detection algorithm to 
    # determine the signal that was sent
    # return the symbols that were sent
    def detectSignal(self,SNR, recieved):
        stdDev = self.getStdDev(SNR)        
        detectedBits = []
        #for every bit recieved 
        for x in range(0, len(recieved)):
            probabilities = []  # 1,j,-1,-j
            temp = []
            temp.append(self.getExpProb(recieved[x], 1,stdDev))
            temp.append(self.getExpProb(recieved[x], 1j, stdDev))
            temp.append(self.getExpProb(recieved[x], -1, stdDev))
            temp.append(self.getExpProb(recieved[x], -1j, stdDev))
            probabilities = np.exp(temp)
            beta = self.getBeta(probabilities)
            for prob in range(0,len(probabilities)):
                probabilities[prob] = probabilities[prob] * beta            
            ind = np.argmax(probabilities)
            detectedBits.append(self.getSymbol(ind))
            
        return detectedBits

    # get exponent e is raised to for 
    # the symbol recieved and the symbol we think it is
    # need standard deviation for the channel as well
    def getExpProb(self,recieved,actual,stdDev):
        temp = np.abs(recieved-actual)
        temp = (temp**2)*(-1)
        temp = temp/(2*(stdDev**2))
        return temp

    # given an array of conditional probabilities
    # return scaling factor or normalization constant (beta)
    def getBeta(self,probs):
        temp = 0
        for x in range(0,len(probs)):
            temp += probs[x]
        return (1/temp)

    # given the index with the highest probability
    # returnn the QPSK symbol
    def getSymbol(self,index):
        if(index == 0):
            return 1
        elif(index == 1):
            return 1j        
        elif(index == 2):
            return -1
        elif(index == 3):
            return -1j

    #return the number of bit errors for the signal
    def getNumErrors(self,sentBits, recievedBits):
        errors = 0
        for x in range(0,len(sentBits)):
            if(sentBits[x] != recievedBits[x]):
                errors += 1
        return errors

    # simulate the sending and recieving
    # plot BER vs SNR as well
    # returns the BER array
    def simulate(self):
        print("Question 3")
        print("QPSK Simulation with", self.numBits, "bits")
        BER = []
        bits = self.generateBits()
        for SNR in range(-4,9):
            print("SNR set to", SNR)
            #map each bit to a Qpsk symbol
            sentSignal = self.QpskModulate(bits)
            print("Signal Sent")
            #print(sentSignal)
            recievedSignal = self.addNoise(SNR, sentSignal)
            print("Signal Recieved")
            #print(recievedSignal)
            detectedSignal = self.detectSignal(SNR, recievedSignal) #still symbols
            #print(detectedSignal)
            #convert symbols to bits
            detectedBits = self.QpskDemodulate(detectedSignal)
            print("Signal Has been demodulated")
            #print(detectedBits)
            BER.append(self.getNumErrors(bits,detectedBits)/self.numBits)
            print()

        print("The Bit Error rate for each SNR tested is given in the array below:")  
        print(BER)
        print()
        print("Plotting the BER vs SNR relationship")
        SNR = np.linspace(-4, 8, 13)
        plt.semilogy(SNR, BER)
        plt.grid(which='both')
        plt.ylabel("BER")
        plt.xlabel("SNR")
        plt.title("Plot of the BER vs SNR for the QPSK Simulation conducted with " + str(self.numBits) + " bits")
        plt.show()
        print("End of QPSK Simulation")
        return BER
################## End of class ###########################

class BPSKmodulationSimulation:
    def __init__(self, numBits):
        self.numBits = numBits
        self.numberGenerator = randomNumber()

    #return standard deviation for SNR with fbit =1
    def getStdDev(self, SNR):
        return (1/np.sqrt(10**(SNR/10)*2*1))

    def addNoise(self, SNR, sentBits):
        recieved = []
        standardDeviation = self.getStdDev(SNR)
        for x in range(0, len(sentBits)):
            real = self.numberGenerator.gaussian(stdDeviation=standardDeviation)
            recieved.append(sentBits[x]+real)
        return recieved

    # generate random number using uniform random number generator
    # round the output to a 1 or 0 for a bit
    def generateBits(self):
        #print("Generating", self.numBits, "random binary digits")
        toSend = []
        for x in range(0, self.numBits):
            toSend.append(int(np.round(self.numberGenerator.WHill())))
        return(toSend)

    # map an array of bits to a symbols
    # according to the BPSK constellation map
    def BpskModulate(self, original):
        #print("Mapping bits to symbols using BPSK modulation")
        mappedMessage = []
        for x in range(0, len(original)):
            mappedMessage.append(self.bitToSymbol(original[x]))
        return mappedMessage

    # map a single bit to a symbol
    # according to the BPSK constellation map
    def bitToSymbol(self, bit):
        if(bit == 1):
            return 1
        if(bit == 0):
            return -1
        else:
            print("Error occured, bit to map was not zero or one")
    
    # demodulate the detected symbols
    # for QPSK
    def BpskDemodulate(self, modulated):
        #print("Mapping symbols to bits for BPSK demodulation")
        demod = []
        for x in range(0, len(modulated)):
            demod.append(self.symbolToBit(modulated[x]))
        return demod

    # return the bit for the
    # the symbol passed in
    def symbolToBit(self, symbol):
        if(symbol == 1):
            return (1)

        elif(symbol == -1):
            return 0

    # given the SNR and recieved signal
    # use optimum detection algorithm to
    # determine the signal that was sent
    # return the symbols that were sent
    def detectSignal(self, SNR, recieved):
        stdDev = self.getStdDev(SNR)
        detectedBits = []
        #for every bit recieved
        for x in range(0, len(recieved)):
            probabilities = []  # 1,j,-1,-j
            temp = []
            temp.append(self.getExpProb(recieved[x], 1, stdDev))
            temp.append(self.getExpProb(recieved[x], -1, stdDev))
            probabilities = np.exp(temp)
            beta = self.getBeta(probabilities)
            for prob in range(0, len(probabilities)):
                probabilities[prob] = probabilities[prob] * beta
            ind = np.argmax(probabilities)
            detectedBits.append(self.getSymbol(ind))
        return detectedBits

    # get exponent e is raised to for
    # the symbol recieved and the symbol we think it is
    # need standard deviation for the channel as well
    def getExpProb(self, recieved, actual, stdDev):
        temp = np.abs(recieved-actual)
        temp = (temp**2)*(-1)
        temp = temp/(2*(stdDev**2))
        return temp

    # given an array of conditional probabilities
    # return scaling factor or normalization constant (beta)
    def getBeta(self, probs):
        temp = 0
        for x in range(0, len(probs)):
            temp += probs[x]
        return (1/temp)

    # given the index with the highest probability
    # return the BPSK symbol
    def getSymbol(self, index):
        if(index == 0):
            return 1

        elif(index == 1):
            return -1

    #return the number of bit errors for the signal
    def getNumErrors(self, sentBits, recievedBits):
        errors = 0
        for x in range(0, len(sentBits)):
            if(sentBits[x] != recievedBits[x]):
                errors += 1
        return errors

    # simulate the sending and recieving 
    # plot BER vs SNR as well
    # returns the BER array
    def simulate(self):
        print("Question 3")
        print("BPSK Simulation with", self.numBits, "bits")
        BER = []
        bits = self.generateBits()
        for SNR in range(-4, 9):
            print("SNR set to", SNR)
            #map each bit to a BPSK symbol
            sentSignal = self.BpskModulate(bits)
            print("Signal Sent")
            #print(sentSignal)
            recievedSignal = self.addNoise(SNR, sentSignal)
            print("Signal Recieved")
            #print(recievedSignal)
            detectedSignal = self.detectSignal(SNR, recievedSignal)  # still symbols
            #print(detectedSignal)
            #convert symbols to bits
            detectedBits = self.BpskDemodulate(detectedSignal)
            print("Signal Has been demodulated")
            #print(detectedBits)
            BER.append(self.getNumErrors(bits, detectedBits)/self.numBits)
            print()

        print("The Bit Error rate for each SNR tested is given in the array below:")
        print(BER)
        print()
        print("Plotting BER vs SNR function for the BPSK Simulation")
        SNR = np.linspace(-4, 8, 13)
        plt.semilogy(SNR, BER)
        plt.grid(which='both')
        plt.ylabel("BER")
        plt.xlabel("SNR")
        plt.title("Plot of the BER vs SNR for the BPSK Simulation conducted with " + str(self.numBits) + " bits")
        plt.show()
        print("End of BPSK Simulation")
        return BER

#plots the result of BPSK and QPSK modulation on a single plot
#must pass in BER for both
def plotToCompare(QpskBer, BpskBer):
    print("Plotting both")
    SNR = np.linspace(-4, 8, 13)
    plt.semilogy(SNR, QpskBer, 'r-', label='QPSK BER')
    plt.semilogy(SNR, BpskBer, 'g-', label='BPSK BER')
    plt.grid(which='both')
    plt.ylabel("BER")
    plt.xlabel("SNR")
    plt.title("Plot of the BER vs SNR for the BPSK and QPSK Simulation conducted")
    plt.legend(loc='best')
    plt.show()
    print("Complete")

# numBits is the number of bits 
# that we would like to use in the simulation of each
# may change the number of bits at will

# create two objects, one for the BPSK simulation
# one for the QPSK Simulation
#numBits = 1000000
#BpskSimulation = BPSKmodulationSimulation(numBits)
#QpskSimulation = QPSKmodulationSimulation(numBits)

#uncomment next line to run the Bpsk Simulation
#bpsk = BpskSimulation.simulate()

#uncomment next line to run the Qpsk Simulation
#qpsk = QpskSimulation.simulate()

#to plot both on the same axis, uncomment next line
#plotToCompare(qpsk,bpsk)
