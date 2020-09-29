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

filename = 'guitarsept.mp3'
sr = 44100
audio = MonoLoader(filename=filename,sampleRate = sr)()

audio = normalize(audio)

length = 10  # in seconds
plotoffset = 0

#Plotting librosa extraction:
C = 300
limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)
pitchcont = pitchtocontinuous(pitchdisc, audio, sr, limits)
plotonoffsetspitch(audio, sr, length, plotoffset, limits, pitch = pitchcont)

plt.figure(2)

#Plotting essentia extraction:
hopSize = 128 #hopsize used to estimate pitch
pitchcont, limits = extractpitchlimitsessentia(audio, sr, hopSize)
plotonoffsetspitch(audio, sr, length, plotoffset, limits, pitchcont, hopSize)

#Checking notes :
# print(librosa.hz_to_note(pitchdisc))
