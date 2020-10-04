import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import essentia as es
from essentia.standard import *

def stereotomono(audio):
    out = np.mean(audio,axis = 1)
    return out
#we normalize audio

def normalize(audio):
    return audio * (1/max(audio))

def makeonsetsessentia(audio):
    # Phase 1: compute the onset detection function
    # The OnsetDetection algorithm provides various onset detection functions. Let's use two of them.

    od1 = OnsetDetection(method='hfc')
    od2 = OnsetDetection(method='complex')

    # Let's also get the other algorithms we will need, and a pool to store the results
    w = Windowing(type='hann')
    fft = FFT()  # this gives us a complex FFT
    c2p = CartesianToPolar()  # and this turns it into a pair (magnitude, phase)
    pool = essentia.Pool()

    # Computing onset detection functions.
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.hfc', od1(mag, phase))
        pool.add('features.complex', od2(mag, phase))

    # Phase 2: compute the actual onsets locations
    onsets = Onsets()

    onsets_complex = onsets(essentia.array([pool['features.complex']]), [1])
    #limits[:,0] = onsets[0 : onsets.shape[0] - 1]
    return onsets_complex

def makelimits(audio, sr,C ):
    #envelope = librosa.onset.onset_strength(audio, sr)
    onsets = librosa.onset.onset_detect(audio, sr, units='time', backtrack = True, hop_length = 256)
    #onsets = makeonsetsessentia(audio)

    limits = np.empty((onsets.shape[0] - 1, 2))
    limits[:, 0] = onsets[0:onsets.shape[0] - 1]  # onsets
    limits[:, 1] = onsets[1:onsets.shape[0]]  # soon offsets

    #offset detection:
    frame_length = 512
    hop_length = 128
    for i in range(onsets.shape[0] - 1):

        energy = librosa.feature.rms(y=audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], frame_length=frame_length, hop_length = hop_length)

        total_energy = sum(sum(energy))

        amax = np.argmax(energy[0, 0 : int(energy[0].shape[0] / 2)])
        #print(amax)
        # a = np.searchsorted(energy[0],total_energy/1000, side='right')

        # we start from the maximum value  in the first half of the note (otherwise if onset is not well placed, it takes the next onset as maximum)
        # and decrease until we go under the minimum energy. This is the offset
        for j in range(np.max((amax+1, int(0.1*sr/hop_length))), energy[0].shape[0]):#notes are at least 0.1s /!\ frequency is the frequency at which we compute energy

            if energy[0, j] < total_energy / C :#or energy[0,j] > energy[0,j-1]: #second condition checks there is not a new note starting. 1st condition: C high, longnotes

                #plt.figure()
                #plt.plot(audio[int(limits[i,0]*sr):int(limits[i, 1]*sr)])
                #print('becomes')

                limits[i, 1] = limits[i, 0] + (j * hop_length) / sr

                #plt.plot(audio[int(limits[i,0]*sr):int(limits[i, 1]*sr)])

                if limits[i, 1] == limits[i, 0]:
                    print('offset = onset -> increase C')
                break
        continue



    return limits

def plotonoffsetspitch(audio, sr, length, plotoffset, limits, tit,  pitch = None, hopSize = 1) :
    times = np.array(range(audio.shape[0])) / sr  # librosa.times_like(audio)

    fig, ax1 = plt.subplots()

    #plotaudio, offsets, onsets
    ax1.plot(times[0 + int(plotoffset * sr):int((plotoffset + length) * sr)],
             audio[0 + int(plotoffset * sr):int((plotoffset + length) * sr)],
             )
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('amplitude')
    plt.title(tit)

    # plot offsets
    offsets = limits[:, 1]
    ax1.scatter(offsets[(offsets < length + plotoffset) & (offsets > plotoffset)],
             np.ones(offsets[(offsets < length + plotoffset) & (offsets > plotoffset)].shape), marker=(5,2), c='tab:red')

    #plot onsets
    onsets = limits[:,0]
    ax1.scatter(onsets[(onsets < length + plotoffset) & (onsets > plotoffset)],
            np.ones(onsets[(onsets < length + plotoffset) & (onsets > plotoffset)].shape), marker=(5,2), c='tab:green')

    #plot pitch
    if pitch is not None :
        pitchtime = np.array(range(int(np.floor(audio.shape[0] / hopSize)))) / (sr / hopSize)
        ax3 = ax1.twinx()

        color = 'tab:red'
        ax3.plot(pitchtime[int(plotoffset * sr / hopSize):int((plotoffset + length) * sr / hopSize)],
                 pitch[int(plotoffset * sr / hopSize):int((plotoffset + length) * sr / hopSize)], color=color)
        ax3.set_ylabel('frequency [Hz]', color=color)  # we already handled the x-label with ax1
        ax3.tick_params(axis='y', labelcolor=color)


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
    pitchfilt = PitchFilter()(pitchcont,confidence)

    segmenter = PitchContourSegmentation(hopSize = hopSize)#hopsize128
    onsets, duration, MIDIpitches = segmenter(pitchfilt, audio)

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

def writenotes(songname, audio,sr,limits):
    os.mkdir(songname[:-4])
    for i in range(limits.shape[0]):
        note = audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)]
        MonoWriter(filename = songname[:-4] + '/note' + str(i) + '.wav',sampleRate = sr)(note)