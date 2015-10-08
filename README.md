# pypYIN
python pYIN

A python version of pYIN of Matthias Mauch  
pitch estimation and note transcription

## Dependencies
Numpy  
Scipy  
Essentia  

## Usage

### Initialise:  

inputSampleRate  
stepSize: hopSize  
blockSize: frameSize  
lowAmp(0,1): under lowAmp considered non voiced  
onsetSensitivity: high value - note is easily be separated into two notes if low amplitude is presented.
pruneThresh(second): discards notes shorter than this threshold

### Other issues:
See demo.py

