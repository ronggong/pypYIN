import os, sys
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import pYINmain
import essentia.standard as ess
import numpy as np

if __name__ == "__main__":

    # initialise
    filename1 = srcpath + '/testAudioShort.wav'
    fs = 44100
    frameSize = 2048
    hopSize = 256

    pYinInst = pYINmain.PyinMain()
    pYinInst.initialise(channels = 1, inputSampleRate = fs, stepSize = hopSize, blockSize = frameSize,
                   lowAmp = 0.25, onsetSensitivity = 0.7, pruneThresh = 0.1)

    # frame-wise calculation
    audio = ess.MonoLoader(filename = filename1, sampleRate = fs)()
    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        fs = pYinInst.process(frame)

    # calculate smoothed pitch and mono note
    fs = pYinInst.getRemainingFeatures()

    # output smoothed pitch track
    print 'pitch track'
    for ii in fs.m_oSmoothedPitchTrack:
        print ii.values
    print '\n'

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

