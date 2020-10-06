from preprocessing import *

def calculate_note_features(note, sr, n_fft, pitch):
    hop_length = int(n_fft / 2)

    # zerocrossingrates of all windows of all notes are put together
    ZCR = librosa.feature.zero_crossing_rate(note, frame_length=2048, hop_length=512)  # 46ms like in paper.
    Spectrogramnote = np.abs(librosa.stft(note, n_fft=n_fft, hop_length=hop_length))
    centroids = librosa.feature.spectral_centroid(S=Spectrogramnote, sr=sr)
    bandwidths = librosa.feature.spectral_bandwidth(S=Spectrogramnote, sr=sr)

    harmonicpercentage = np.empty((0))
    inharmonicity = np.empty(0)

    for frame in FrameGenerator(note, frameSize=n_fft, hopSize=hop_length, startFromZero=True):
        # print('frame' + str(idx))

        window = Windowing(type='blackmanharris92')(frame)
        spectrum = Spectrum(size=n_fft)(window)
        # spectrum = np.delete(spectrum, np.where(spectrum == 0))
        specdb = 10 * np.log10(spectrum / min(s for s in spectrum if s > 0))
        frequencies, magnitudes = SpectralPeaks(maxPeaks=100, sampleRate=sr)(
            specdb)  # should be in dB, and best with blackmanharriswindow with 92db

        magnitudes = np.delete(magnitudes, np.where(frequencies == 0))
        frequencies = np.delete(frequencies, np.where(frequencies == 0))

        harmonicfreq, harmonicmag = HarmonicPeaks(maxHarmonics=4, tolerance=0.3)(frequencies, magnitudes, float(
            pitch))  # we feed frequencies, magnitudes and pitch.

        percentage_bandwidth = pitch / 12  # in paper, 1/12 octave

        for k in range(harmonicfreq.shape[0]):
            # print(EnergyBandRatio(sampleRate = sr, startFrequency = harmonicfreq[k] - (percentage_bandwidth/2), stopFrequency = harmonicfreq[k] + (percentage_bandwidth/2))(spectrum))
            harmonicpercentage = np.append(harmonicpercentage, EnergyBandRatio(sampleRate=sr,
                                                                               startFrequency=harmonicfreq[k] - (
                                                                                           percentage_bandwidth / 2),
                                                                               stopFrequency=harmonicfreq[k] + (
                                                                                           percentage_bandwidth / 2))(
                spectrum))

        inharmonicity = np.append(inharmonicity, Inharmonicity()(harmonicfreq,
                                                              harmonicmag))  # should we first mean the frequencies and magnitudes, than give this to inharmonicity?
    harmonicpercentage = harmonicpercentage.reshape(-1,
                                                    4)  # gives second dimension, otherwise we do not know which harmonic peak it was.
    return ZCR, centroids, bandwidths, inharmonicity, harmonicpercentage

def calculate_track_features(filename, sr, C, n_fft):

    audio  = MonoLoader(filename = filename, sampleRate =sr)()
    audio = normalize(audio)

    #we get limits and pitches from librosa
    limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)

    zerocrossingrates = np.empty((limits.shape[0],2))
    centroids = np.empty((limits.shape[0],2))
    bandwidths = np.empty((limits.shape[0],2))
    noteinharmonicities = np.empty((limits.shape[0],2))
    harmonicpercentage1 = np.empty((limits.shape[0],2))
    harmonicpercentage2 = np.empty((limits.shape[0], 2))
    harmonicpercentage3 = np.empty((limits.shape[0], 2))
    harmonicpercentage4 = np.empty((limits.shape[0], 2))

    #noteharmonicpercentage = np.empty((limits.shape[0], 4, 2))


    for i in range(limits.shape[0]):
        #note splitting
        note = audio[int(limits[i, 0]*sr): int(limits[i, 1]*sr)]

        ZCR, centroid, bandwidth, inharmonicity, harmonicpercentage = calculate_note_features(note, sr, n_fft, pitchdisc[i])

        #Zerocrossingrate

        zerocrossingrates[i,0] = np.mean(ZCR)
        zerocrossingrates[i,1] = np.std(ZCR)

        #Spectral Features

        centroids[i,0] = np.mean(centroid)
        centroids[i,1] = np.std(centroid)

        bandwidths[i,0] = np.mean(bandwidth)
        bandwidths[i,1] = np.std(bandwidth)

        #Harmonic Features

        #print('inhamronicity' + str(inharmonicity))
        noteinharmonicities[i,0] = np.mean(inharmonicity)
        noteinharmonicities[i, 1] = np.std(inharmonicity)

        #noteharmonicpercentage[i, :, 0] = np.mean(harmonicpercentage, axis = 0)
        #noteharmonicpercentage[i, :, 1] = np.std(harmonicpercentage, axis= 0)

        harmonicpercentage1[i,0] = np.mean(harmonicpercentage[:,0])
        harmonicpercentage1[i,1] = np.std(harmonicpercentage[:,0])

        harmonicpercentage2[i, 0] = np.mean(harmonicpercentage[:, 1])
        harmonicpercentage2[i, 1] = np.std(harmonicpercentage[:, 1])

        harmonicpercentage3[i, 0] = np.mean(harmonicpercentage[:, 2])
        harmonicpercentage3[i, 1] = np.std(harmonicpercentage[:, 2])

        harmonicpercentage4[i, 0] = np.mean(harmonicpercentage[:, 3])
        harmonicpercentage4[i, 1] = np.std(harmonicpercentage[:, 3])


    features = np.array([zerocrossingrates, centroids, bandwidths, noteinharmonicities, harmonicpercentage1, harmonicpercentage2, harmonicpercentage3, harmonicpercentage4])
    print (features.shape)
    return features

def calculate_tracks_features(filenames, sr, C, n_fft):
    tracks = {}
    notes = np.empty((8,0,2))
    for filename in filenames:
        print(filename)
        tracks[filename] = calculate_track_features(filename,sr,C,n_fft)
    return tracks, notes

def feature_variance(tracks):
    allnotes = np.empty((8,0,2))
    intravariances = np.empty((len(tracks), 8, 2))
    i=0
    for filename in tracks.keys():
        print(filename)
        print(tracks[filename].shape)
        allnotes = np.concatenate((allnotes, tracks[filename]), axis=1)
        intravariances[i] = np.std(tracks[filename], axis = 1) #track, 8, 2
        i = i+1

    intervariance = np.std(allnotes, axis = 1)
    print('intervariance:' +'\n'+  str(intervariance))
    meanintravariance = np.mean(intravariances, axis = 0)
    print('intravariance:' +'\n'+ str(meanintravariance))


sr = 44100
C = 300
n_fft = 1024
#filenames = ['data/September - Earth, Wind & Fire/*.mp3','data/September - Earth, Wind & Fire/*.mp3']
filenames = ['data/September - Earth, Wind & Fire/Bass.mp3', 'data/September - Earth, Wind & Fire/Lead Vocal.mp3']
tracks, notes = calculate_tracks_features(filenames, sr, C, n_fft)
feature_variance(tracks)
print('yes')

print(notes.shape)
