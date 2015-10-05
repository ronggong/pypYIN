from MonoNoteHMM import MonoNoteHMM
from MonoNoteParameters import MonoNoteParameters
import numpy as np
import time


class FrameOutput(object):
    def __init__(self, frameNumber, pitch, noteState):
        self.frameNumber = frameNumber
        self.pitch = pitch
        self.noteState = noteState

class MonoNote(object):

    def __init__(self):
        self.hmm = MonoNoteHMM()

    def process(self, pitchProb):
        obsProb = [self.hmm.calculatedObsProb(pitchProb[0]), ]
        for iFrame in range(1, len(pitchProb)):
            obsProb += [self.hmm.calculatedObsProb(pitchProb[iFrame])]
        out = []

        path, scale = self.hmm.decodeViterbi(obsProb)

        for iFrame in range(len(path)):
            currPitch = -1.0
            stateKind = 0

            currPitch = self.hmm.par.minPitch + (path[iFrame]/self.hmm.par.nSPP) * 1.0/self.hmm.par.nPPS
            stateKind = (path[iFrame]) % self.hmm.par.nSPP + 1

            out.append(FrameOutput(iFrame, currPitch, stateKind))

        return out