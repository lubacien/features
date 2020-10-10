from preprocessing import *
from argparse import ArgumentParser
import pickle
import time
import torch


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
    logtime, start, stop = LogAttackTime()(note)
    envelope = Envelope()(note)
    envflat = FlatnessSFX()(envelope)
    tempcentroid = TCToTotal()(envelope)

    return ZCR, centroids, bandwidths, inharmonicity, harmonicpercentage, logtime, envflat, tempcentroid

def calculate_track_features(filename, sr, C, n_fft):

    audio  = MonoLoader(filename = filename, sampleRate =sr)()
    audio = normalize(audio)

    #we get limits and pitches from librosa
    limits, pitchdisc = extractpitchlimitslibrosa(audio,sr,C)

    #noteharmonicpercentage = np.empty((limits.shape[0], 4, 2))

    features = np.empty((limits.shape[0], 19))
    for i in range(limits.shape[0]):
        #note splitting
        note = audio[int(limits[i, 0]*sr): int(limits[i, 1]*sr)]

        ZCR, centroid, bandwidth, inharmonicity, harmonicpercentage, logtime, envflat, tempcentroid = calculate_note_features(note, sr, n_fft, pitchdisc[i])

        features[i] = np.array([np.mean(ZCR), np.std(ZCR), np.mean(centroid), np.std(centroid),
                                np.mean(bandwidth), np.std(bandwidth), np.mean(inharmonicity), np.mean(inharmonicity),
                                 np.mean(harmonicpercentage[:,0]), np.std(harmonicpercentage[:,0]),
                                 np.mean(harmonicpercentage[:, 1]), np.std(harmonicpercentage[:, 1]),
                                 np.mean(harmonicpercentage[:, 2]), np.std(harmonicpercentage[:, 2]),
                                 np.mean(harmonicpercentage[:, 3]), np.std(harmonicpercentage[:, 3]),
                                logtime, envflat, tempcentroid ])

    return features

def calculate_tracks_features(songnames, sr, C, n_fft):
    instruments = {}
    for songname in songnames:
        filenames = os.listdir(str(args.indir) + '/' + songname)
        for filename in filenames:
            print(songname + '/' + filename)
            start = time.time()
            instruments[filename] = calculate_track_features(str(args.indir) + '/' + songname + '/' + filename, sr, C, n_fft)
            stop = time.time()
            print('computed in '+  str(stop-start) + 's')
    return instruments

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--indir', type=str, default='raw_dataset',
        help='directory where the tracks are')
    args = argparser.parse_args()

sr = 44100
C = 300
n_fft = 1024

songnames = os.listdir(args.indir)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else 'cpu')
torch.cuda.set_device(device)
print('use_cuda' + str(use_cuda))
print('cudadevice' + str(device))

#WRITE:
instruments = calculate_tracks_features(songnames, sr, C, n_fft)
picklename = 'instruments.pkl'
filehandler = open(picklename, 'wb')
pickle.dump(tracks, filehandler)
print('file written at '+ str(picklename))
filehandler.close()
