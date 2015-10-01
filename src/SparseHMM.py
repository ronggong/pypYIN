import numpy as np
from math import *

class SparseHMM(object):

    def __init__(self):
        self.init = np.array([], dtype=np.float64)
        self.transProb = np.array([], dtype=np.float64)
        self.fromIndex = np.array([], dtype=np.uint64)
        self.toIndex = np.array([],dtype=np.uint64)

    def calculatedObsProb(self, data):
        # to be overloaded
        return data

    def decodeViterbi(self, obsProb):

        if len(obsProb) < 1: return np.array([], dtype=np.int)

        nState = len(self.init)
        nFrame = len(obsProb)

        # check for consistency
        nTrans = len(self.transProb)

        # declaring variables
        scale = np.array([], dtype=np.float64)
        delta = np.zeros((nState,), dtype=np.float64)
        oldDelta = np.zeros((nState,), dtype=np.float64)
        path = np.ones(nFrame, dtype=np.int) * (nState-1)  # the final output path

        deltasum = 0

        # initialise first frame in time 1, rabiner 32a
        for iState in range(nState):
            oldDelta[iState] = self.init[iState] * obsProb[0][iState]
            deltasum += oldDelta[iState]

        for iState in range(nState):
            oldDelta[iState] /= deltasum  # normalise (scale)

        scale = np.append(scale, np.double(1.0/deltasum))
        psi = [np.zeros(nState, dtype=np.int),]  # matrix of remembered indices of the best transitions

        # rest of forward step
        for iFrame in range(1, nFrame):
            deltasum = 0
            psi = psi + [np.zeros(nState, dtype=np.int)]

            # calculate best previous state for every current state

            # this is the "sparse" loop
            for iTrans in range(nTrans):
                fromState = self.fromIndex[iTrans]
                toState = self.toIndex[iTrans]
                currentTransProb = self.transProb[iTrans]

                currentValue = oldDelta[fromState] * currentTransProb
                if (currentValue > delta[toState]):  # to find the maximum
                    delta[toState] = currentValue  # just change the toState delta, will be multiplied by the right obs later!
                    psi[iFrame][toState] = fromState # rabiner 33b

            for jState in range(nState):
                delta[jState] *= obsProb[iFrame][jState]
                deltasum += delta[jState]

            if deltasum > 0:
                for iState in range(nState):
                    oldDelta[iState] = delta[iState] / deltasum  # normalise (scale)
                    delta[iState] = 0
                scale = np.append(scale, np.double(1.0/deltasum))
            else:
                print "WARNING: Viterbi has been fed some zero probabilities, at least they become zero at frame " +  str(iFrame) + " in combination with the model."
                for iState in range(nState):
                    oldDelta[iState] = 1.0/nState
                    delta[iState] = 0
                scale = np.append(scale, np.double(1.0/deltasum))

        # initialise backward step
        bestValue = 0
        for iState in range(nState):
            currentValue = oldDelta[iState]  # use directly the normalised delta
            if currentValue > bestValue:
                bestValue = currentValue #  rabiner 34b
                path[nFrame-1] = iState #  path of last frame

        for iFrame in reversed(range(nFrame-1)):
            path[iFrame] = psi[iFrame+1][path[iFrame+1]]

        return path, scale