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

from SparseHMM import SparseHMM
from math import *
import numpy as np


class MonoPitchHMM(SparseHMM):

    def __init__(self):
        SparseHMM.__init__(self)
        self.m_minFreq = 61.735
        self.m_nBPS = 5
        self.m_nPitch = 0
        self.m_selfTrans = 0.99
        self.m_yinTrust = 0.5
        self.m_transitionWidth = 5*(np.uint64(self.m_nBPS/2)) + 1  # 2 semi-tones of frame jump
        self.m_nPitch = 69 * self.m_nBPS  # 69 semi-tone, each semi-tone divided to 5, step is 20 cents
        self.m_freqs = np.zeros(2*self.m_nPitch, dtype=np.float64)
        for iPitch in range(self.m_nPitch):
            self.m_freqs[iPitch] = self.m_minFreq * pow(2, iPitch * 1.0 / (12 * self.m_nBPS))  # 0 to m_nPitch-1 positive pitch
            self.m_freqs[iPitch+self.m_nPitch] = -self.m_freqs[iPitch]  # m_nPitch to 2*m_nPitch-1 negative pitch
        self.build()

    def calculatedObsProb(self, pitchProb):
        # pitchProb is the pitch candidates of one frame
        out = np.zeros((2*self.m_nPitch+1,), dtype=np.float64)
        probYinPitched = 0.0
        # BIN THE PITCHES
        for iPair in range(len(pitchProb)):
            freq = 440. * pow(2.0, (pitchProb[iPair][0] - 69)/12.0)
            if freq <= self.m_minFreq: continue
            d = 0
            oldd = 1000
            for iPitch in range(self.m_nPitch):
                d = fabs(freq-self.m_freqs[iPitch])
                if oldd < d and iPitch > 0:
                    # previous bin must have been the closest
                    # when iPitch move far away from freq candidate
                    # add pitch prob to probYinPitched, break
                    out[iPitch-1] = pitchProb[iPair][1]
                    probYinPitched += out[iPitch-1]
                    break
                oldd = d

        probReallyPitched = self.m_yinTrust * probYinPitched
        # damn, I forget what this is all about...
        # don't understand this part, inspired by note tracking method
        for iPitch in range(self.m_nPitch):
            if probYinPitched > 0: out[iPitch] *= (probReallyPitched/probYinPitched) # times self.m_yinTrust
            #  non voiced pitch obs
            #  1 - sum(pitchProb)*0.5
            #  this observation prob is very small, but equal for every unvoiced state
            #  so that the sum of them are 1 - sum(pitchProb)*0.5
            out[iPitch+self.m_nPitch] = (1 - probReallyPitched) / self.m_nPitch

        return out

    def build(self):

        # initial vector, uniform distribution
        self.init = np.ones((2*self.m_nPitch), dtype=np.float64) * 1.0/2*self.m_nPitch

        # transitions
        for iPitch in range(self.m_nPitch):
            theoreticalMinNextPitch = int(iPitch)-int(self.m_transitionWidth/2)
            minNextPitch = iPitch-int(self.m_transitionWidth/2) if iPitch>self.m_transitionWidth/2 else 0
            maxNextPitch = iPitch+int(self.m_transitionWidth/2) if iPitch<self.m_nPitch-self.m_transitionWidth/2 else self.m_nPitch-1

            # weight vector, triangle, maximum is at iPitch
            weightSum = 0
            weights = np.array([], dtype=np.float64)
            for i in range(minNextPitch, maxNextPitch+1):
                if i <= iPitch:
                    weights = np.append(weights, np.float64(i-theoreticalMinNextPitch+1))
                else:
                    weights = np.append(weights, np.float64(iPitch-theoreticalMinNextPitch+1-(i-iPitch)))
                weightSum += weights[len(weights)-1]

            for i in range(minNextPitch, maxNextPitch+1):
                # from voiced to voiced
                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * self.m_selfTrans))

                # from voiced to non voiced
                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i+self.m_nPitch))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * (1-self.m_selfTrans)))

                # from non voiced to non voiced
                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch+self.m_nPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i+self.m_nPitch))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * self.m_selfTrans))

                # from non voiced to voiced
                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch+self.m_nPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * (1-self.m_selfTrans)))
