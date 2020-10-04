import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
import scipy.fftpack

filename = 'guitarsept.mp3'


sr = 44100
audio  = MonoLoader(filename = filename,sampleRate =sr)()
audio = normalize(audio)

#we get limits and pitches from librosa
C = 300
limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)
#writenotes(filename, audio, sr, limits)

n_fft = 1024
hop_length= int(n_fft/2)

zerocrossingrates = np.empty((limits.shape[0],2))
centroids = np.empty((limits.shape[0],2))
bandwidths = np.empty((limits.shape[0],2))

noteinharmonicities = np.empty((limits.shape[0],2))
noteharmonicpercentage = np.empty((limits.shape[0], 4, 2))

for i in range(limits.shape[0]):
    #note splitting
    note = audio[int(limits[i, 0]*sr): int(limits[i, 1]*sr)]

    #Zerocrossingrate
    # zerocrossingrates of all windows of all notes are put together
    ZCR = librosa.feature.zero_crossing_rate(note, frame_length=2048, hop_length=512)#46ms like in paper.
    zerocrossingrates[i,0] = np.mean(ZCR)
    zerocrossingrates[i,1] = np.std(ZCR)

    #Spectral Features
    Spectrogramnote = np.abs(librosa.stft(note, n_fft=n_fft, hop_length=hop_length))

    centroids[i,0] = np.mean(librosa.feature.spectral_centroid(S=Spectrogramnote, sr=sr))
    centroids[i,1] = np.std(librosa.feature.spectral_centroid(S=Spectrogramnote, sr=sr))

    bandwidths[i,0] = np.mean(librosa.feature.spectral_bandwidth(S=Spectrogramnote, sr=sr))
    bandwidths[i,1] = np.std(librosa.feature.spectral_bandwidth(S=Spectrogramnote, sr=sr))

    #Harmonic Features

    harmonicpercentage= np.empty((0))
    inharmonicity = np.empty(0)

    for frame in FrameGenerator(note, frameSize = n_fft, hopSize = hop_length, startFromZero = True):
        #print('frame' + str(idx))

        window = Windowing(type='blackmanharris92')(frame)
        spectrum = Spectrum(size = n_fft)(window)
        specdb = 10*np.log10(spectrum/min(spectrum))
        frequencies, magnitudes = SpectralPeaks(maxPeaks=100,sampleRate = sr)(specdb)#should be in dB, and best with blackmanharriswindow with 92db

        magnitudes = np.delete(magnitudes, np.where(frequencies == 0))
        frequencies = np.delete(frequencies, np.where(frequencies == 0))

        harmonicfreq, harmonicmag = HarmonicPeaks(maxHarmonics = 4, tolerance = 0.3)(frequencies, magnitudes, float(pitchdisc[i]))#we feed frequencies, magnitudes and pitch.

        percentage_bandwidth = pitchdisc[i] / 12 #in paper, 1/12 octave
        for k in range(harmonicfreq.shape[0]):
            #print(EnergyBandRatio(sampleRate = sr, startFrequency = harmonicfreq[k] - (percentage_bandwidth/2), stopFrequency = harmonicfreq[k] + (percentage_bandwidth/2))(spectrum))
            harmonicpercentage = np.append(harmonicpercentage, EnergyBandRatio(sampleRate = sr, startFrequency = harmonicfreq[k] - (percentage_bandwidth/2), stopFrequency = harmonicfreq[k] + (percentage_bandwidth/2))(spectrum) )

        inharmonicity = np.append(inharmonicity, Inharmonicity()(harmonicfreq, harmonicmag))#should we first mean the frequencies and magnitudes, than give this to inharmonicity?

    #print('inhamronicity' + str(inharmonicity))
    noteinharmonicities[i,0] = np.mean(inharmonicity)
    noteinharmonicities[i, 1] = np.std(inharmonicity)

    #print('harmonicpercent' + str(harmonicpercentage))
    harmonicpercentage = harmonicpercentage.reshape(-1,4) #gives second dimension, otherwise we do not know which harmonic peak it was.
    noteharmonicpercentage[i, :, 0] = np.mean(harmonicpercentage, axis = 0)
    noteharmonicpercentage[i, :, 1] = np.std(harmonicpercentage, axis= 0)

    '''
    #NOWINDOW
    if (note.shape[0] % 2) != 0:
        note = np.append(note, 0)
    spectrum = Spectrum()(note)

    dbspec = 10*np.log10(spectrum/min(spectrum))

    frequencies, magnitudes = SpectralPeaks(maxPeaks=100, sampleRate=sr)(dbspec)  # should be in dB, and best with blackmanharriswindow with 92db

    magnitudes = np.delete(magnitudes, np.where(frequencies == 0))
    frequencies = np.delete(frequencies, np.where(frequencies == 0))
    harmonicfreq, harmonicmag= HarmonicPeaks(maxHarmonics=4, tolerance = 0.3)(frequencies, magnitudes, float( #increasing tolerance gives better inharmonicity results
        pitchdisc[i]))  # we feed frequencies, magnitudes and pitch.
    noteinharmonicities[i] = Inharmonicity()(harmonicfreq, harmonicmag)
    harmonicfreqs[i] = harmonicfreq
    harmonicmags[i] = harmonicmag
    '''

def display(feature, name, save = False):
    # we want those distributions to be as small as possible, otherwise it means we have a big variety of the features
    # across different notes of the same song, therefore the features are useless.
    save = True
    plt.figure()
    plt.boxplot(feature[:,0])
    plt.title('Distribution of the means of ' + name + ' across notes')
    if save == True:
        plt.savefig('plots/boxplots' + filename[:-4] + 'mean_' + name)
    plt.figure()
    plt.boxplot(feature[:, 1])
    plt.title('Distribution of the deviations of ' + name + ' across notes')
    if save == True:
        plt.savefig('plots/boxplots/' + filename[:-4] + 'deviation_' + name)
    plt.show()

    feature = np.mean(feature, axis = 0)# here we take the means of the means and standard deviations computed on notes.
    print(name + ' :')
    print('mean : ' + str(feature[0]))
    print('standard deviation : ' + str(feature[1]))

display(zerocrossingrates,'ZCR')
display(centroids,'centroids')
display(bandwidths,'bandwidths')
display(noteinharmonicities,'inharmonicity')
display(noteharmonicpercentage[:, 0, :],'harmonic_1')#dim 1 is the note, dim2 is the feature, dim 3 is mean or std
display(noteharmonicpercentage[:, 1, :], 'harmonic_2')
display(noteharmonicpercentage[:, 2, :],'harmonic_3')
display(noteharmonicpercentage[:, 3, :], 'harmonic_4')
