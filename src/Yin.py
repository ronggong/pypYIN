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