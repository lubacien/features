# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from os import path
import librosa
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from preprocessing import *

# you need to download ffmpeg to convert mp3 files to wav. command line program.

filename = 'Voicesep.mp3'
sr = 44100
audio = MonoLoader(filename=filename,sampleRate = sr)()

audio = normalize(audio)

length = 7  # in seconds
plotoffset = 18

#Plotting librosa extraction:
C = 300
limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)

#writenotes('septguitar', audio, sr, limits)

pitchcont = pitchtocontinuous(pitchdisc, audio, sr, limits)
plotonoffsetspitch(audio, sr, length, plotoffset, limits, 'Separate extraction of onsets, offsets and pitches', pitch = pitchcont)

#print(limits)
plt.figure(2)


#Plotting essentia extraction:
hopSize = 128 #hopsize used to estimate pitch
pitchcont, limits = extractpitchlimitsessentia(audio, sr, hopSize)
plotonoffsetspitch(audio, sr, length, plotoffset, limits, 'Joint extraction of onsets, offsets and pitches with essentia', pitchcont, hopSize)

#Checking notes :
# print(librosa.hz_to_note(pitchdisc))
