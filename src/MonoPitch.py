from MonoPitchHMM import MonoPitchHMM
import numpy as np
from math import *

class MonoPitch(object):
    def __init__(self):
        self.hmm = MonoPitchHMM()

    def process(self, pitchProb):
        obsProb = [self.hmm.calculatedObsProb(pitchProb[0]),]
        for iFrame in range(1,len(pitchProb)):
            obsProb += [self.hmm.calculatedObsProb(pitchProb[iFrame])]

        out = np.array([], dtype=np.float32)

        path, scale = self.hmm.decodeViterbi(obsProb)

        for iFrame in range(len(path)):
            hmmFreq = self.hmm.m_freqs[path[iFrame]]
            bestFreq = 0.0
            leastDist = 10000.0
            if hmmFreq > 0:
                # This was a Yin estimate, so try to get original pitch estimate back
                # ... a bit hacky, since we could have direclty saved the frequency
                # that was assigned to the HMM bin in hmm.calculateObsProb -- but would
                # have had to rethink the interface of that method.
                for iPitch in range(len(pitchProb[iFrame])):
                    freq = 440. * pow(2.0, (pitchProb[iFrame][iPitch][0] - 69)/12.0)
                    dist = fabs(hmmFreq-freq)
                    if dist < leastDist:
                        leastDist = dist
                        bestFreq = freq
            else:
                bestFreq = hmmFreq
            out = np.append(out, bestFreq)
        return out