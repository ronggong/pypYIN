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