import pYINmain
import essentia.standard as ess
import numpy as np

if __name__ == "__main__":

    filename1 = 'testAudioShort.wav'
    fs = 44100
    frameSize = 2048
    hopSize = 256

    pYinInst = pYINmain.PyinMain()
    pYinInst.initialise(channels = 1, inputSampleRate = fs, stepSize = hopSize, blockSize = frameSize)

    audio = ess.MonoLoader(filename = filename1, sampleRate = fs)()
    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        fs = pYinInst.process(frame)
        #print fs.m_oF0Candidates[0].values, fs.m_oF0Probs[0].values, fs.m_oVoicedProb[0].values
    fs = pYinInst.getRemainingFeatures()

