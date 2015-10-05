import numpy as np
from SparseHMM import SparseHMM
from MonoNoteParameters import MonoNoteParameters
from math import *
from scipy.stats import norm

class MonoNoteHMM(SparseHMM):
    def __init__(self):
        SparseHMM.__init__(self)
        self.par = MonoNoteParameters()
        self.pitchDistr = []
        self.build()

    def calculatedObsProb(self, pitchProb):
        # pitchProb is a list of pairs (pitches and their probabilities)

        nCandidate = len(pitchProb)

        # what is the probability of pitched
        pIsPitched = 0.0
        for iCandidate in range(nCandidate):
            # pIsPitched = pitchProb[iCandidate].second > pIsPitched ? pitchProb[iCandidate].second : pIsPitched;
            pIsPitched += pitchProb[iCandidate][1]

        # pIsPitched = std::pow(pIsPitched, (1-par.priorWeight)) * std::pow(par.priorPitchedProb, par.priorWeight);
        pIsPitched = pIsPitched * (1-self.par.priorWeight) + self.par.priorPitchedProb * self.par.priorWeight

        out = np.zeros((self.par.n,), dtype=np.float64)
        tempProbSum = 0
        for i in range(self.par.n):
            if i % self.par.nSPP != 2:
                # std::cerr << getMidiPitch(i) << std::endl;
                tempProb = 0.0
                if nCandidate > 0:
                    minDist = 10000.0
                    minDistProb = 0.0
                    minDistCandidate = 0
                    for iCandidate in range(nCandidate):
                        currDist = fabs(self.getMidiPitch(i)-pitchProb[iCandidate][0])
                        if (currDist < minDist):
                            minDist = currDist
                            minDistProb = pitchProb[iCandidate][1]
                            minDistCandidate = iCandidate
                    tempProb = pow(minDistProb, self.par.yinTrust) * self.pitchDistr[i].pdf(pitchProb[minDistCandidate][0])
                else:
                    tempProb = 1
                tempProbSum += tempProb
                out[i] = tempProb

        for i in range(self.par.n):
            if i % self.par.nSPP != 2:
                if tempProbSum > 0:
                    out[i] = out[i] / tempProbSum * pIsPitched
            else:
                out[i] = (1-pIsPitched) / (self.par.nPPS * self.par.nS)

        return out

    def getMidiPitch(self, index):
        return self.pitchDistr[index].mean()

    def getFrequency(self, index):
        return 440 * pow(2.0, (self.pitchDistr[index].mean()-69)/12)

    def build(self):
        # the states are organised as follows:
        # 0-2. lowest pitch
        #    0. attack state
        #    1. stable state
        #   2. silent state
        # 3-5. second-lowest pitch
        #    3. attack state
        #    ...

        # observation distributions
        for iState in range(self.par.n):
            self.pitchDistr.append(norm(loc=0, scale=1))
            if iState % self.par.nSPP == 2:
                # silent state starts tracking
                self.init = np.append(self.init, np.float64(1.0/(self.par.nS * self.par.nPPS)))
            else:
                self.init = np.append(self.init, np.float64(0.0))

        for iPitch in range(self.par.nS * self.par.nPPS):
            index = iPitch * self.par.nSPP
            mu = self.par.minPitch + iPitch * 1.0/self.par.nPPS
            self.pitchDistr[index] = norm(loc=mu, scale=self.par.sigmaYinPitchAttack)
            self.pitchDistr[index+1] = norm(loc=mu, scale=self.par.sigmaYinPitchStable)
            self.pitchDistr[index+2] = norm(loc=mu, scale=1.0) # dummy

        noteDistanceDistr = norm(loc=0, scale=self.par.sigma2Note)

        for iPitch in range(self.par.nS * self.par.nPPS):
            # loop through all notes and set sparse transition probabilities
            index = iPitch * self.par.nSPP

            # transitions from attack state
            self.fromIndex = np.append(self.fromIndex, np.uint64(index))
            self.toIndex = np.append(self.toIndex, np.uint64(index))
            self.transProb = np.append(self.transProb, np.float64(self.par.pAttackSelftrans))

            self.fromIndex = np.append(self.fromIndex, np.uint64(index))
            self.toIndex = np.append(self.toIndex, np.uint64(index+1))
            self.transProb = np.append(self.transProb, np.float64(1-self.par.pAttackSelftrans))

            # transitions from stable state
            self.fromIndex = np.append(self.fromIndex, np.uint64(index+1))
            self.toIndex = np.append(self.toIndex, np.uint64(index+1)) # to itself
            self.transProb = np.append(self.transProb, np.float64(self.par.pStableSelftrans))

            self.fromIndex = np.append(self.fromIndex, np.uint64(index+1))
            self.toIndex = np.append(self.toIndex, np.uint64(index+2)) # to silent
            self.transProb = np.append(self.transProb, np.float64(self.par.pStable2Silent))

            # the "easy" transitions from silent state
            self.fromIndex = np.append(self.fromIndex, np.uint64(index+2))
            self.toIndex = np.append(self.toIndex, np.uint64(index+2))
            self.transProb = np.append(self.transProb, np.float64(self.par.pSilentSelftrans))

            # the more complicated transitions from the silent
            probSumSilent = 0.0

            tempTransProbSilent = []
            for jPitch in range(self.par.nS * self.par.nPPS):
                fromPitch = iPitch
                toPitch = jPitch
                semitoneDistance = fabs(fromPitch - toPitch) * 1.0 / self.par.nPPS

                if semitoneDistance == 0 or \
                        (semitoneDistance > self.par.minSemitoneDistance and semitoneDistance < self.par.maxJump):

                    toIndex = jPitch * self.par.nSPP  # note attack index

                    tempWeightSilent = noteDistanceDistr.pdf(semitoneDistance)
                    probSumSilent += tempWeightSilent

                    tempTransProbSilent.append(tempWeightSilent)

                    self.fromIndex = np.append(self.fromIndex, np.uint64(index+2))
                    self.toIndex = np.append(self.toIndex, np.uint64(toIndex))
            for i in range(len(tempTransProbSilent)):
                self.transProb = np.append(self.transProb,
                                          ((1-self.par.pSilentSelftrans) * tempTransProbSilent[i]/probSumSilent))