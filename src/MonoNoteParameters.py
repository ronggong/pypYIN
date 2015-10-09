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

class MonoNoteParameters(object):
    def __init__(self):
        self.minPitch = 35
        self.nPPS = 3  # 3 steps per semitone
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