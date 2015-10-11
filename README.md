# pypYIN
python pYIN

A python version of pYIN of Matthias Mauch  
Pitch and note tracking in monophonic audio

## pYIN project page
[https://code.soundsoftware.ac.uk/projects/pyin](https://code.soundsoftware.ac.uk/projects/pyin)

## Dependencies
Numpy  
Scipy  
Essentia  

## Usage

### Initialise:  
Here are the parameters which need to be initialised before executing the main program:  

inputSampleRate:      sampling rate
stepSize:             hopSize  
blockSize:            frameSize  
lowAmp(0,1):          RMS of audio frame under lowAmp will be considered non voiced  
onsetSensitivity:     high value means note is easily be separated into two notes if low amplitude is presented.  
pruneThresh(second):  discards notes shorter than this threshold

### Output:
Transcribed notes in Hz  
Smoothed pitch track  
Pitch tracks of transcribed notes in MIDI note number  

### Other issues:
See demo.py

## License
 Copyright (C) 2015  Music Technology Group - Universitat Pompeu Fabra  
 
 This file is part of pypYIN  
 
 pypYIN is free software: you can redistribute it and/or modify it under  
 the terms of the GNU Affero General Public License as published by the Free  
 Software Foundation (FSF), either version 3 of the License, or (at your  
 option) any later version.  
 
 This program is distributed in the hope that it will be useful, but WITHOUT  
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS  
 FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more  
 details.  
 
 You should have received a copy of the Affero GNU General Public License  
 version 3 along with this program.  If not, see http://www.gnu.org/licenses/  

 If you have any problem about this python version code, please contact: Rong Gong  
 rong.gong@upf.edu  
 
 If you have any problem about this algorithm, I suggest you to contact: Matthias Mauch  
 m.mauch@qmul.ac.uk who is the original C++ version author of this algorithm  
 
 If you want to refer this code, please consider these articles: 
 
 > M. Mauch and S. Dixon,  
 > “pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions”,  
 > in Proceedings of the IEEE International Conference on Acoustics,  
 > Speech, and Signal Processing (ICASSP 2014), 2014.  
 
 > M. Mauch, C. Cannam, R. Bittner, G. Fazekas, J. Salamon, J. Dai, J. Bello and S. Dixon,  
 > “Computer-aided Melody Note Transcription Using the Tony Software: Accuracy and Efficiency”,  
 > in Proceedings of the First International Conference on Technologies for  
 > Music Notation and Representation, 2015.  

