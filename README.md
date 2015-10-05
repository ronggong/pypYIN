# pypYIN
python pYIN

A python version of pYIN of Matthias Mauch  
pitch estimation and note transcription

## Dependencies
Numpy  
Scipy  
Essentia  

## usage

### Initialise:  

inputSampleRate  
stepSize: hopSize  
blockSize: frameSize  
lowAmp(0,1): under lowAmp considered non voiced  
onsetSensitivity: see Matthias's pYin paper for explication  
pruneThresh: see Matthias's pYin paper for explication

### other issues:
See demo.py

