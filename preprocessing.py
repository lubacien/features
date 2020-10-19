import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import essentia as es
from essentia.standard import *
from features import *

def stereotomono(audio):
    out = np.mean(audio,axis = 1)
    return out
#we normalize audio

def normalize(audio):
    return audio * (1/max(audio))

def makeonsetsessentia(audio, sr):
    # Phase 1: compute the onset detection function
    #
    od1 = OnsetDetection(method='hfc', sampleRate = sr)
    od2 = OnsetDetection(method='complex', sampleRate = sr)
    od3 = OnsetDetection(method='flux', sampleRate = sr)
    od4 = OnsetDetection(method='complex_phase', sampleRate = sr)

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
        pool.add('features.flux', od3(mag, phase))
        pool.add('features.complex_phase', od4(mag, phase))


    # Phase 2: compute the actual onsets locations
    onsets = Onsets(alpha = 0.2, delay = 5)#alpha high for less onsets

    onsets_mixed = onsets(essentia.array([ pool['features.complex_phase'], pool['features.flux'],
                                          pool['features.hfc'], pool['features.complex'] ]), [1,1,1,1])
    #limits[:,0] = onsets[0 : onsets.shape[0] - 1]
    #onsets_mixed = onsets(essentia.array([pool['features.hfc']]), [1])

    for i in range(len(onsets_mixed)-1):
        if onsets_mixed[i] == onsets_mixed[i+1]:
            onsets_mixed = np.delete(onsets_mixed, i+1)
            print('deleted onset')

    return onsets_mixed

def makelimits(audio, sr,C ):
    #envelope = librosa.onset.onset_strength(audio, sr)
    #onsets = librosa.onset.onset_detect(audio, sr, units='time', backtrack = True, hop_length = 256)
    onsets = makeonsetsessentia(audio,sr)
    onsets = np.unique(onsets)
    limits = np.empty((onsets.shape[0] - 1, 2))

    #offset detection:
    frame_length = 512
    hop_length = 128

    for i in range(onsets.shape[0] - 1):
        limits[i,0] = onsets[i]
        limits[i,1] = onsets[i+1]
        #print('audio: '+ str(audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)]))
        energy = librosa.feature.rms(y=audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], frame_length=frame_length, hop_length = hop_length)

        total_energy = sum(sum(energy))

        amax = np.argmax(energy[0, 0 : int(energy[0].shape[0] / 2)])#looks for the max energy in the first half of the note

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
    pitchextractor = PitchMelodia(sampleRate = sr, maxFrequency = 4000)#frameSize 2048, hopSize 128.
    pitchcont, confidence = pitchextractor(audio)
    pitchfilt = PitchFilter()(pitchcont, confidence)

    segmenter = PitchContourSegmentation(hopSize = hopSize, sampleRate = sr, rmsThreshold = -2, pitchDistanceThreshold = 100)#hopsize128
    onsets, duration, MIDIpitches = segmenter(pitchfilt, audio)

    limits = np.empty((onsets.shape[0],2))
    limits[:,0] = onsets
    limits[:,1] = onsets + duration

    pitchdiscrete = []
    for i in range(limits.shape[0]):
        pitchdiscrete.append(np.median(pitchfilt[int(limits[i,0] * (sr/128)) : int(limits[i,1] *(sr/128)) ]))

    return pitchfilt, pitchdiscrete, limits

def pitchfindmelodia(audio,sr,limits):

    pitchdisc = np.empty(limits[:, 0].shape)
    hopSize = 128
    pitchextractor = PitchMelodia(sampleRate=sr, maxFrequency=4000)  # frameSize 2048, hopSize 128.
    pitchcont, confidence = pitchextractor(audio)
    pitchfilt = PitchFilter()(pitchcont, confidence)

    for i in range(limits.shape[0]):
        #pitchdisc[i] = pitchcont[int(limits[i, 0] * (sr / hopSize)) + np.argmax(confidence[int(limits[i, 0] * (sr / hopSize)): int(limits[i, 1] * (sr / hopSize))])]
        pitchdisc[i] = np.median(pitchfilt[int(limits[i, 0] * (sr/hopSize)) : int(limits[i, 1] * (sr/hopSize))])
        print(pitchfilt)

    return pitchdisc



def extractpitchlimitslibrosa(audio,sr, C):
    #if =100, onset = offset, doesnt work.
    limits = makelimits(audio, sr, C)
    pitchdisc = np.empty(limits[:,0].shape)
    frame_length = int(2048)

    pitchdisc = pitchfindmelodia(audio, sr, limits)# melodia algorithm, long but maybe more robust

    th = 0.01#for harmonic analysis
    harmonicpercentage = []

    for i in range(limits.shape[0]):
        #if max(confidence[limits[i, 0] * (sr/hopSize) : limits[i, 1] * (sr/hopSize)]) > 0.8 :

        #does not work: pitchdisc[i] = pitchcont[int(limits[i, 0] * (sr/hopSize)) + np.argmax(confidence[int(limits[i, 0] * (sr/hopSize)) : int(limits[i, 1] * (sr/hopSize))])])

        #does not work either pitchdisc[i] = np.mean(pitchfilt[int(limits[i, 0] * (sr/hopSize)) : int(limits[i, 1] * (sr/hopSize))])

        freqsperframe = librosa.yin(audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], fmin=40, fmax=4000, sr = sr)
        pitchdisc[i] = np.median(freqsperframe)

        '''
        #Harmonic filtering
        _, _, _, _, hp, _ ,_ ,_ = calculate_note_features(audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], sr, 1024, pitchdisc[i])

        harmonicpercentage.append(np.mean(hp, axis = 0))
    
    harmonicpercentage = np.mean(harmonicpercentage, axis = 0)
    print(harmonicpercentage)
    if harmonicpercentage[0] < th :
        print('first harmonic empty')
        if harmonicpercentage[1] < th:
            print('second harmonic empty')
            if harmonicpercentage[2] < th:
                print('third harmonic is empty')
                if harmonicpercentage[3] < th:
                    print('fourth harmonic is empty, note removed')
                    pitchdisc[i] = 0
                else:
                    print('pitch * 4')
                    pitchdisc[i] = 4 * pitchdisc[i]
            else:
                print('pitch is tripled')
                pitchdisc[i] = 3 * pitchdisc[i]
        else:
            print('pitch is doubled')
            pitchdisc[i] = 2* pitchdisc[i]
    '''
    '''
    #instead of taking median, we take the highest probability: too slow and works less well
    f0, voiceflags, voiceprobs = librosa.pyin(audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)], fmin=40,
                                fmax=4000, sr = sr, frame_length= frame_length, hop_length = int(frame_length/2))
    if max(voiceprobs > threshold):
        pitchdisc[i] = f0[np.argmax(voiceprobs)]
    else:
        pitchdisc[i] = 0
    '''
    return limits, pitchdisc

def writenotes(songname, audio, sr,limits, pitchdisc):
    os.mkdir(songname[:-4])
    for i in range(limits.shape[0]):
        if pitchdisc[i] != 0:

            note = audio[int(limits[i, 0] * sr): int(limits[i, 1] * sr)]
            MonoWriter(filename = songname[:-4] + '/note' + str(i) + '.wav',sampleRate = sr)(note)