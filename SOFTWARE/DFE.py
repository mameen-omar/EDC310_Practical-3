# Mohamed Ameen Omar
# 16055323
####################################
###      EDC 310 Practical 2     ###
###             2018             ###
###         BPSK DFE Algotihm    ###
####################################

import numpy as np
import copy 

# Class to run a DFE equalizer to dermine the bits sent over a AGWN channel
# Constructor paramaters:
#   @param N = the number of data bits of data being sent
#   @param r = a vector with all the recieved symbols
#   @param c = the channel impulse response vector
class DFE:
    def __init__(self, N = 0, r = [], c = []):
        self.n = N #number of data bits
        self.r = r #recieved vector - only data bits len(r) = self.n
        self.c = c #convolution matrix
        self.L = len(c) #length of the c vector
        self.numHeader = self.L-1 #number of header bits
        self.symbols = [1,-1]
        self.dataDetected = [] #just the data bits detected

    # Function to return the data symbols detected. 
    # It will detect or estimate or symbols recieved and return 
    # a vector with those symbols
    def getDataSymbols(self):
        if(self.dataDetected == []):
            self.detectSymbols()
            return self.dataDetected        
        else:
            return self.dataDetected

    # Function to detect the symbols in the recieved vector
    # using the DFE Equalizer
    def detectSymbols(self):
        for x in range(0,len(self.r)):
            self.dataDetected.append(self.getSymbol(x))
    # Function to detect a single sumbol using 
    # DFE Equalizer algorithm
    def getSymbol(self,time):
        if(self.getDelta(time, 1) > self.getDelta(time, -1)):
            return -1
        else:
            return 1

    # Function to get the delta value for a single symbol
    # symbol is the symbol we are estimating
    # time is the time instance for the symbol we are getting the delta
    def getDelta(self,time,symbol):
        temp = 0
        t = 0
        for x in range(0,self.L):
            if(x == 0):
                temp += self.c[x]*symbol
            else:
                #if we havent detected the first l symbols
                if(len(self.dataDetected)+t <0):
                    temp += self.c[x]*1
                else:
                    temp += self.c[x]*self.dataDetected[len(self.dataDetected)+t]                
            t = t-1
        temp = np.abs(self.r[time] - temp) **2
        return temp
################# END OF CLASS IMPLEMENTATION ########################################