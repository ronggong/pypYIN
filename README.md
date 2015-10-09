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

