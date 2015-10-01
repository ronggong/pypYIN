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
        self.m_transitionWidth = 5*(np.uint64(self.m_nBPS/2)) + 1
        self.m_nPitch = 69 * self.m_nBPS
        self.m_freqs = np.zeros(2*self.m_nPitch, dtype=np.float64)
        for iPitch in range(self.m_nPitch):
            self.m_freqs[iPitch] = self.m_minFreq * pow(2, iPitch * 1.0 / (12 * self.m_nBPS))
            self.m_freqs[iPitch+self.m_nPitch] = -self.m_freqs[iPitch]
        self.build()

    def calculateObsProb(self, pitchProb):
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
                    out[iPitch-1] = pitchProb[iPair][1]
                    probYinPitched += out[iPitch-1]
                    break
                oldd = d

        probReallyPitched = self.m_yinTrust * probYinPitched
        # damn, I forget what this is all about...
        for iPitch in range(self.m_nPitch):
            if probYinPitched > 0: out[iPitch] *= (probReallyPitched/probYinPitched)
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

            # weight vector
            weightSum = 0
            weights = np.array([], dtype=np.float64)
            for i in range(minNextPitch, maxNextPitch+1):
                if i <= iPitch:
                    weights = np.append(weights, np.float64(i-theoreticalMinNextPitch+1))
                else:
                    weights = np.append(weights, np.float64(iPitch-theoreticalMinNextPitch+1-(i-iPitch)))
                weightSum += weights[len(weights)-1]

            for i in range(minNextPitch, maxNextPitch+1):
                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * self.m_selfTrans))

                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i+self.m_nPitch))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * (1-self.m_selfTrans)))

                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch+self.m_nPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i+self.m_nPitch))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * self.m_selfTrans))

                self.fromIndex = np.append(self.fromIndex, np.uint64(iPitch+self.m_nPitch))
                self.toIndex = np.append(self.toIndex, np.uint64(i))
                self.transProb = np.append(self.transProb, np.float64(weights[i-minNextPitch] / weightSum * (1-self.m_selfTrans)))
