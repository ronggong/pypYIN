import numpy as np

class MonoNoteParameters(object):
    def __init__(self):
        self.minPitch = 35
        self.nPPS = 3
        self.nS = 69
        self.nSPP = 3  # states per pitch
        self.n = 0
        self.initPi = np.array([], dtype=np.float64)
        self.pAttackSelftrans = 0.9
        self.pStableSelftrans = 0.99
        self.pStable2Silent = 0.01
        self.pSilentSelftrans = 0.9999
        self.sigma2Note = 0.7
        self.maxJump = 13.0
        self.pInterSelftrans = 0.0
        self.priorPitchedProb = 0.7
        self.priorWeight = 0.5
        self.minSemitoneDistance = 0.5
        self.sigmaYinPitchAttack = 5.0
        self.sigmaYinPitchStable = 0.8
        self.sigmaYinPitchInter = 0.1
        self.yinTrust = 0.1

        self.n = self.nPPS * self.nS * self.nSPP