import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *

filename = 'guitarsept.mp3'
#audio, sr, channelnum, md5, bitrate, codec = AudioLoader(filename = filename)()
sr = 44100
audio  = MonoLoader(filename = filename,sampleRate =sr)()
audio = normalize(audio)

#we get limits and pitches from librosa
C = 300
limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)


#print(pitchtodiscrete(pitchcont,hopSize,sr,limits)[1])
