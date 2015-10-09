# -*- coding: utf-8 -*-

'''
 * Copyright (C) 2015  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of pypYIN
 *
 * pypYIN is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 *
 * If you have any problem about this python version code, please contact: Rong Gong
 * rong.gong@upf.edu
 *
 * If you have any problem about this algorithm, I suggest you to contact: Matthias Mauch
 * m.mauch@qmul.ac.uk who is the original C++ version author of this algorithm
 *
 * If you want to refer this code, please consider this article:
 *
 * M. Mauch and S. Dixon,
 * “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions”,
 * in Proceedings of the IEEE International Conference on Acoustics,
 * Speech, and Signal Processing (ICASSP 2014), 2014.
 *
 * M. Mauch, C. Cannam, R. Bittner, G. Fazekas, J. Salamon, J. Dai, J. Bello and S. Dixon,
 * “Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency”,
 * in Proceedings of the First International Conference on Technologies for
 * Music Notation and Representation, 2015.
'''

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