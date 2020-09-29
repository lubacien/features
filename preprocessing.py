import librosa
import numpy as np
import matplotlib.pyplot as plt
import essentia as es
from essentia.standard import *

def stereotomono(audio):
    out = np.mean(audio,axis = 1)
    return out
#we normalize audio
def normalize(audio):
    return audio * (1/max(audio))

def makelimits(audio, sr,C ):
    onsets = librosa.onset.onset_detect(audio, sr, units='time')
    # we want to have 7 onsets in the first 9 seconds, but they must be computed on the full length

    limits = np.empty((onsets.shape[0] - 1, 2))
    limits[:, 0] = onsets[0:onsets.shape[0] - 1]  # onsets
    limits[:, 1] = onsets[1:onsets.shape[0]]  # soon offsets

    frame_length = 2048
    hop_length = 512
    for i in range(onsets.shape[0] - 1):
        energy = librosa.feature.rms(y=audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], frame_length=frame_length)
        total_energy = sum(sum(energy))
        max = np.argmax(energy[0])
        # a = np.searchsorted(energy[0],total_energy/1000, side='right')
        # we start from the maximum value and decrease until we go under the minimum energy. this is the offset
        for j in range(max, energy[0].shape[0]):

            if energy[0, j] < total_energy / C:
                #print(limits[i, 1])
                #print('becomes')

                limits[i, 1] = limits[i, 0] + (j * hop_length) / sr
                if limits[i, 1] == limits[i, 0]:
                    print('offset = onset -> increase C')
                break
        continue

    return limits

def plotonoffsetspitch(audio, sr, length, plotoffset, limits, pitch = None, hopSize = 1):
    times = np.array(range(audio.shape[0])) / sr  # librosa.times_like(audio)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    #plotaudio
    ax1.plot(times[0 + int(plotoffset * sr):int((plotoffset + length) * sr)],
             audio[0 + int(plotoffset * sr):int((plotoffset + length) * sr)])

    # plot offsets
    offsets = limits[:, 1]
    ax1.plot(offsets[(offsets < length + plotoffset) & (offsets > plotoffset)],
             np.ones(offsets[(offsets < length + plotoffset) & (offsets > plotoffset)].shape) * 0, marker='o',
             linewidth=0)

    #plot onsets
    onsets = limits[:,0]
    ax1.plot(onsets[(onsets < length + plotoffset) & (onsets > plotoffset)],
             np.ones(onsets[(onsets < length + plotoffset) & (onsets > plotoffset)].shape) * 0, marker='o', linewidth=0)

    #plot pitch
    if pitch is not None :
        print('yo')
        pitchtime = np.array(range(int(np.floor(audio.shape[0] / hopSize)))) / (sr / hopSize)
        ax2 = ax1.twinx()
        ax2.plot(pitchtime[int(plotoffset * sr / hopSize):int((plotoffset + length) * sr / hopSize)],
                 pitch[int(plotoffset * sr / hopSize):int((plotoffset + length) * sr / hopSize)], color='red')

    plt.show()
'''
def pitchtodiscrete(pitchcont,hopSize,sr,limits):
    discretepitch = np.array((limits.shape[0],0))
    print(discretepitch.shape)
    print(limits.shape[0])
    print(pitchcont.shape)
    for i in range(limits.shape[0]):
        discretepitch[i] = np.mean(pitchcont[ int(limits[i,0] * sr /hopSize) : int(limits[i,1] * sr /hopSize) ])
    return discretepitch
'''

def pitchtocontinuous(pitchdisc, audio, sr, limits):
    pitchcont = np.empty(audio.shape)
    for i in range(limits.shape[0]):
        pitchcont[int(limits[i,0]*sr):int(limits[i,1]*sr)] = pitchdisc[i]
    return pitchcont

def extractpitchlimitsessentia(audio, sr, hopSize):

    #audio =stereotomono(audio)

    eqfilter = EqualLoudness(sampleRate = sr)
    audio = eqfilter(audio)

    #frameSize = 2048
    pitchextractor = PitchMelodia()#frameSize 2048, hopSize 128.
    pitchcont, confidence = pitchextractor(audio)

    segmenter = PitchContourSegmentation(hopSize = hopSize)#hopsize128
    onsets, duration, MIDIpitches = segmenter(pitchcont, audio)

    limits = np.empty((onsets.shape[0],2))
    limits[:,0] = onsets
    limits[:,1] = onsets + duration

    return pitchcont, limits

def extractpitchlimitslibrosa(audio,sr, C):
    #if =100, onset = offset, doesnt work.
    limits = makelimits(audio, sr, C)
    pitchdisc = np.empty(limits[:,0].shape)
    for i in range(limits.shape[0]):
        freqsperframe = librosa.yin(audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], fmin=librosa.note_to_hz('C2'),
                                    fmax=librosa.note_to_hz('C7'))
        pitchdisc[i] = np.median(freqsperframe)  # WE REMOVE OUTLIERS

    return limits, pitchdisc