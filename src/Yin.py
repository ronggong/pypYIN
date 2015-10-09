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
import YinUtil

class Yin(object):

    def __init__(self):
        self.m_frameSize = 2048
        self.m_inputSampleRate = 44100
        self.m_thresh = 0.2
        self.m_threshDistr = 2
        self.m_yinBufferSize = self.m_frameSize/2
        self.m_fast = True

    def Yin(self, frameSize, inputSampleRate, thresh = 0.2, fast = True):
        self.m_frameSize = frameSize
        self.m_inputSampleRate = inputSampleRate
        self.m_thresh = thresh
        self.m_threshDistr = 2
        self.m_yinBufferSize = frameSize/2
        self.m_fast = fast

    class YinOutput(object):

        def __init__(self, f0 = 0.0, periodicity = 0.0, rms = 0.0):
            self.f0 = f0
            self.periodicity = periodicity
            self.rms = rms
            self.salience = np.array([], dtype=np.float64)
            self.freqProb = np.array([], dtype=np.float64)

    def processProbabilisticYin(self, input):

        # calculate aperiodicity function for all periods, output stores in yinBuffer
        if self.m_fast:
            yinBuffer = YinUtil.fastDifference(input, self.m_yinBufferSize)
        else:
            yinBuffer = YinUtil.slowDifference(input, self.m_yinBufferSize)

        yinBuffer = YinUtil.cumulativeDifference(yinBuffer ,self.m_yinBufferSize)

        peakProbability = YinUtil.yinProb(yinBuffer, self.m_threshDistr, self.m_yinBufferSize, 0, 0)

        # calculate overall "probability" from peak probability, overall "probability" probSum seems never be used
        rms = sqrt(YinUtil.sumSquare(input, 0, self.m_yinBufferSize)/self.m_yinBufferSize)
        yo = Yin.YinOutput(0.0, 0.0, rms)

        firstStack = False
        for iBuf in range(self.m_yinBufferSize):
            yo.salience = np.append(yo.salience, peakProbability[iBuf])

            # if peakProb > 0, a fundamental frequency candidate is generated
            if peakProbability[iBuf] > 0:
                currentF0 = self.m_inputSampleRate * (1.0 / YinUtil.parabolicInterpolation(yinBuffer, iBuf, self.m_yinBufferSize))
                if firstStack == False:
                    yo.freqProb = np.array([np.array([currentF0, peakProbability[iBuf]], dtype=np.float64),])
                    firstStack = True
                else:
                    yo.freqProb = np.vstack((yo.freqProb, np.array([currentF0, peakProbability[iBuf]], dtype=np.float64)))
        return yo

    def setThreshold(self, parameter):

        self.m_thresh = parameter
        return 0

    def setThresholdDistr(self, parameter):

        self.m_threshDistr = parameter
        return 0

    def setFrameSize(self, parameter):

        m_frameSize = parameter
        m_yinBufferSize = m_frameSize/2
        return 0

    def setFast(self, parameter):

        m_fast = parameter
        return 0