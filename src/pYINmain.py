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
import copy
from math import *
from Yin import *
from YinUtil import RMS
from MonoPitch import MonoPitch
from MonoNote import MonoNote

class Feature(object):
    def __init__(self):
        self.values = np.array([], dtype=np.float64)

    def resetValues(self):
        self.values = np.array([], dtype=np.float64)

class FeatureSet(object):
    def __init__(self):
        self.m_oF0Candidates = []
        self.m_oF0Probs = []
        self.m_oVoicedProb = []
        self.m_oCandidateSalience = []
        self.m_oSmoothedPitchTrack = []
        self.m_oMonoNoteOut = []
        self.m_oNotes = []
        self.m_oNotePitchTracks = []

class PyinMain(object):

    def __init__(self):
        self.m_channels = 0
        self.m_stepSize = 256
        self.m_blockSize = 2048
        self.m_inputSampleRate = 44100
        self.m_fmin = 40
        self.m_fmax = 1600

        self.m_yin = Yin()

        self.m_threshDistr = 2.0
        self.m_outputUnvoiced = 2
        self.m_preciseTime = 0.0
        self.m_lowAmp = 0.1
        self.m_onsetSensitivity = 0.7
        self.m_pruneThresh = 0.1

        self.m_pitchProb = []
        self.m_level = np.array([], dtype=np.float32)

        self.fs = FeatureSet()

    def initialise(self, channels = 1, inputSampleRate = 44100, stepSize = 256, blockSize = 2048,
                   lowAmp = 0.1, onsetSensitivity = 0.7, pruneThresh = 0.1 ):

        if channels != 1:
            return False

        self.m_channels = channels
        self.m_inputSampleRate = inputSampleRate
        self.m_stepSize = stepSize
        self.m_blockSize = blockSize

        self.m_lowAmp = lowAmp
        self.m_onsetSensitivity = onsetSensitivity
        self.m_pruneThresh = pruneThresh

        self.reset()

        return True

    def reset(self):

        self.m_yin.setThresholdDistr(self.m_threshDistr)
        self.m_yin.setFrameSize(self.m_blockSize)
        self.m_yin.setFast(not self.m_preciseTime)

        self.m_pitchProb = np.array([], dtype=np.float64)
        self.m_level = np.array([], dtype=np.float32)

    def process(self, inputBuffers):

        dInputBuffers = np.zeros((self.m_blockSize,), dtype=np.float64)
        for i in range(self.m_blockSize):
            dInputBuffers[i] = inputBuffers[i]

        rms = RMS(inputBuffers, self.m_blockSize)

        isLowAmplitude = rms < self.m_lowAmp

        yo = self.m_yin.processProbabilisticYin(dInputBuffers)

        self.m_level = np.append(self.m_level, yo.rms)

        '''
        First, get the things out of the way that we don't want to output
        immediately, but instead save for later
        '''
        tempPitchProb = np.array([], dtype=np.float32)
        firstStack = False
        for iCandidate in range(yo.freqProb.shape[0]):
            tempPitch = 12.0 * log(yo.freqProb[iCandidate][0]/440.0)/log(2.0) + 69.0
            if not isLowAmplitude:
                if firstStack == False:
                    tempPitchProb = np.array([np.array([tempPitch, yo.freqProb[iCandidate][1]], dtype=np.float64),])
                    firstStack = True
                else:
                    tempPitchProb = np.vstack((tempPitchProb, np.array([tempPitch, yo.freqProb[iCandidate][1]], dtype=np.float64)))
            else:
                factor = ((rms+0.01*self.m_lowAmp)/(1.01*self.m_lowAmp))
                if firstStack == False:
                    tempPitchProb = np.array([np.array([tempPitch, yo.freqProb[iCandidate][1]*factor], dtype=np.float64),])
                    firstStack = True
                else:
                    tempPitchProb = np.vstack((tempPitchProb, np.array([tempPitch, yo.freqProb[iCandidate][1]*factor], dtype=np.float64)))
        if len(self.m_pitchProb) < 1 and len(tempPitchProb) > 0:
            self.m_pitchProb = [tempPitchProb,]
        elif len(self.m_pitchProb) >= 1:
            self.m_pitchProb = self.m_pitchProb + [tempPitchProb]

        # f0 CANDIDATES
        f = Feature()
        for i in range(yo.freqProb.shape[0]):
            f.values = np.append(f.values, yo.freqProb[i][0])
        self.fs.m_oF0Candidates.append(copy.copy(f))

        f.resetValues()
        voicedProb = 0.0
        for i in range(yo.freqProb.shape[0]):
            f.values = np.append(f.values, yo.freqProb[i][1])
            voicedProb += yo.freqProb[i][1]
        self.fs.m_oF0Probs.append(copy.copy(f))

        f.values = np.append(f.values, voicedProb)
        self.fs.m_oVoicedProb.append(copy.copy(f))

        # SALIENCE -- maybe this should eventually disappear
        f.resetValues()
        salienceSum = 0.0
        for iBin in range(yo.salience.shape[0]):
            f.values = np.append(f.values, yo.salience[iBin])
            salienceSum += yo.salience[iBin]
        self.fs.m_oCandidateSalience.append(copy.copy(f))

        return self.fs

    def getSmoothedPitchTrack(self):
        f = Feature()

        if len(self.m_pitchProb) == 0:
            return self.fs

        # MONO-PITCH STUFF
        mp = MonoPitch()
        mpOut = mp.process(self.m_pitchProb)
        for iFrame in range(len(mpOut)):
            if mpOut[iFrame] < 0 and self.m_outputUnvoiced == 0:
                continue
            f.resetValues()
            if self.m_outputUnvoiced == 1:
                f.values = np.append(f.values, np.fabs(mpOut[iFrame]))
            else:
                f.values = np.append(f.values, mpOut[iFrame])

            self.fs.m_oSmoothedPitchTrack.append(copy.copy(f))

        return mpOut

    def getRemainingFeatures(self,mpOut):
        f = Feature()

        if len(mpOut) == 0:
            return self.fs

        # if len(self.m_pitchProb) == 0:
        #     return self.fs
        #
        # # MONO-PITCH STUFF
        # mp = MonoPitch()
        # mpOut = mp.process(self.m_pitchProb)
        # for iFrame in range(len(mpOut)):
        #     if mpOut[iFrame] < 0 and self.m_outputUnvoiced == 0:
        #         continue
        #     f.resetValues()
        #     if self.m_outputUnvoiced == 1:
        #         f.values = np.append(f.values, np.fabs(mpOut[iFrame]))
        #     else:
        #         f.values = np.append(f.values, mpOut[iFrame])
        #
        #     self.fs.m_oSmoothedPitchTrack.append(copy.copy(f))

        # MONO-NOTE STUFF
        mn = MonoNote()
        smoothedPitch = []
        for iFrame in range(len(mpOut)):
            temp = []
            if mpOut[iFrame] > 0:  # negative value: silence
                tempPitch = 12 * log(mpOut[iFrame]/440.0)/log(2.0) + 69
                temp += [[tempPitch, 0.9]]
            smoothedPitch += [temp]

        mnOut = mn.process(smoothedPitch)

        self.fs.m_oMonoNoteOut = mnOut

        # turning feature into a note feature

        f.resetValues()

        onsetFrame = 0
        isVoiced = 0
        oldIsVoiced = 0
        nFrame = len(self.m_pitchProb)

        minNoteFrames = (self.m_inputSampleRate*self.m_pruneThresh)/self.m_stepSize

        notePitchTrack = np.array([], dtype=np.float32) # collects pitches for one note at a time
        for iFrame in range(nFrame):
            isVoiced = mnOut[iFrame].noteState < 3 \
            and len(smoothedPitch[iFrame]) > 0 \
            and (iFrame >= nFrame-2 or (self.m_level[iFrame]/self.m_level[iFrame+2]>self.m_onsetSensitivity))

            if isVoiced and iFrame != nFrame-1:
                if oldIsVoiced == 0: # beginning of the note
                    onsetFrame = iFrame
                pitch = smoothedPitch[iFrame][0][0]
                notePitchTrack = np.append(notePitchTrack, pitch) # add to the note's pitch
            else: # not currently voiced
                if oldIsVoiced == 1: # end of the note
                    if len(notePitchTrack) >= minNoteFrames:
                        notePitchTrack = np.sort(notePitchTrack)
                        medianPitch = notePitchTrack[int(len(notePitchTrack)/2)]
                        medianFreq = pow(2, (medianPitch-69)/12)*440
                        f.resetValues()
                        f.values = np.append(f.values, np.double(medianFreq))
                        self.fs.m_oNotes.append(copy.copy(f))
                        self.fs.m_oNotePitchTracks.append(copy.copy(notePitchTrack))
                    notePitchTrack = np.array([], dtype=np.float32)
            oldIsVoiced = isVoiced

        return self.fs
