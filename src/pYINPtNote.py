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

import os, sys
import pYINmain
import essentia.standard as ess
import numpy as np
from YinUtil import RMS

def pYINPtNote(filename1,fs=44100,frameSize=2048,hopSize=256):

    '''
    Given filename, return pitchtrack and note transcription track
    :param filename1:
    :param fs:
    :param frameSize:
    :param hopSize:
    :return:
    '''
    # initialise
    pYinInst = pYINmain.PyinMain()
    pYinInst.initialise(channels = 1, inputSampleRate = fs, stepSize = hopSize, blockSize = frameSize,
                   lowAmp = 0.25, onsetSensitivity = 0.7, pruneThresh = 0.1)

    # frame-wise calculation
    audio = ess.MonoLoader(filename = filename1, sampleRate = fs)()

    # rms mean
    # rms = []
    # for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    #     rms.append(RMS(frame, frameSize))
    # rmsMean = np.mean(rms)
    # print 'rmsMean', rmsMean

    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        fs = pYinInst.process(frame)

    # calculate smoothed pitch and mono note
    monoPitch = pYinInst.getSmoothedPitchTrack()

    # output smoothed pitch track
    print 'pitch track'
    for ii in fs.m_oSmoothedPitchTrack:
        print ii.values
    print '\n'

    fs = pYinInst.getRemainingFeatures(monoPitch)

    # output of mono notes,
    # column 0: frame number,
    # column 1: pitch in midi numuber, this is the decoded pitch
    # column 2: attack 1, stable 2, silence 3
    print 'mono note decoded pitch'
    for ii in fs.m_oMonoNoteOut:
        print ii.frameNumber, ii.pitch, ii.noteState
    print '\n'

    print 'note pitch tracks'
    for ii in fs.m_oNotePitchTracks:
        print ii
    print '\n'

    # median pitch in Hz of the notes
    print 'median note pitch'
    for ii in fs.m_oNotes:
        print ii.values
    print '\n'
