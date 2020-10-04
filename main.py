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
#writenotes(filename, audio, sr, limits)


n_fft = 1024
hop_length= int(n_fft/2)
Saudio = np.abs(librosa.stft(audio, n_fft = n_fft, hop_length= hop_length))
centroids = librosa.feature.spectral_centroid(S= Saudio, sr = sr)


noteinharmonicities = np.empty(limits.shape[0])
shapes = np.empty(limits.shape[0])
harmonicpeaks = np.empty((limits.shape[0],4))

for i in range(limits.shape[0]):

    note = audio[int(limits[i, 0]*sr): int(limits[i, 1]*sr)]

    shapes[i] = note.shape[0]

    inharmonicity = np.empty(int(np.floor((note.shape[0] - 1 )/ hop_length)))
    harmonicfreqs = np.empty((int(np.floor((note.shape[0] - 1) / hop_length)), 4))
    harmonicmags = np.empty((int(np.floor((note.shape[0] - 1) / hop_length)), 4))
    idx=0
    for frame in FrameGenerator(note, frameSize = n_fft, hopSize = hop_length, startFromZero = True):
        print('frame' + str(idx))
        window = Windowing(type='blackmanharris92')(frame)

        spectrum = Spectrum(size = n_fft)(frame)

        #plt.plot(range(0,1000), 10*np.log10(spectrum[0:1000]))
        specdb = 10*np.log10(spectrum/min(spectrum))
        frequencies, magnitudes = SpectralPeaks(maxPeaks=100,sampleRate = sr)(specdb)#should be in dB, and best with blackmanharriswindow with 92db
        magnitudes = np.delete(magnitudes, np.where(frequencies == 0))
        frequencies = np.delete(frequencies, np.where(frequencies == 0))

        harmonicfreq, harmonicmag = HarmonicPeaks(maxHarmonics = 4,tolerance = 0.3)(frequencies, magnitudes, float(pitchdisc[i]))#we feed frequencies, magnitudes and pitch.

        harmonicfreqs[idx,:] = harmonicfreq
        harmonicmags[idx,:] = harmonicmag
        inharmonicity[idx] = Inharmonicity()(harmonicfreq, harmonicmag)
        idx = idx+1

    print(pitchdisc[i])
    print(inharmonicity)
    print(harmonicfreqs)
    print(harmonicmags)

    noteinharmonicities[i] = np.median(inharmonicity)# or mean?


print(noteinharmonicities)
print(np.mean(noteinharmonicities))
print(np.std(noteinharmonicities))

plt.scatter(noteinharmonicities,shapes)
plt.figure()
plt.boxplot(noteinharmonicities)
plt.show()





