from MonoNoteHMM import MonoNoteHMM
from MonoNoteParameters import MonoNoteParameters
import numpy as np


class FrameOutput(object):
    def __init__(self, frameNumber, pitch, noteState):
        self.frameNumber = frameNumber
        self.pitch = pitch
        self.noteState = noteState

class MonoNote(object):

    def __int__(self):
        self.hmm = MonoNoteHMM()

    def process(self, pitchProb):
        obsProb = [MonoNoteHMM().calculatedObsProb(pitchProb[0]), ]
        for iFrame in range(1, len(pitchProb)):
            obsProb += [MonoNoteHMM().calculatedObsProb(pitchProb[iFrame])]

        out = []

        path, scale = MonoNoteHMM().decodeViterbi(obsProb)

        for iFrame in range(len(path)):
            currPitch = -1.0
            stateKind = 0

            currPitch = MonoNoteHMM().par.minPitch + (path[iFrame]/MonoNoteHMM().par.nSPP) * 1.0/MonoNoteHMM().par.nPPS
            stateKind = (path[iFrame]) % MonoNoteHMM().par.nSPP + 1

            out.append(FrameOutput(iFrame, currPitch, stateKind))

        return out