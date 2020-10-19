# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from os import path
import librosa
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from preprocessing import *
from synthesizer import Synthesizer, Waveform, Writer

def write_melody(notefreqs, notelimits, outpath):

    writer = Writer()

    print("write note sequence to " + outpath)
    rate = 44100
    synthesizer = Synthesizer(osc1_waveform=Waveform.sine, osc1_volume=1.0, use_osc2=False,rate = rate)
    waves = np.empty(0)
    for i, notefreq in enumerate(notefreqs):
        wave = synthesizer.generate_constant_wave(notefreq, notelimits[i, 1]-notelimits[i, 0])
        waves = np.concatenate((waves, wave))
        if i < len(notefreqs)-1:
            waves = np.concatenate( (waves, np.zeros(int( (notelimits[i+1, 0] - notelimits[i,1])*rate) )) )

    writer.write_waves(outpath, waves)

# you need to download ffmpeg to convert mp3 files to wav. command line program.

directory = 'test_preprocessing'

#['Voicesep.mp3',guitarsep,  'guitar_chain.mp3', 'Brass section.mp3', 'Piano.mp3', 'DrumKit_september.mp3', 'Organ 2.mp3', 'Bass_september.mp3']

filenames = os.listdir(directory)
length = 8
offset = [18, 0, 3, 11, 0, 8, 50, 20]

#filenames = ['Piano.mp3']
#length = 8
#offset= [18]

for i, filename in enumerate(filenames):
    print(filename)
    sr = 44100
    audio = MonoLoader(filename=directory + '/' + filename, sampleRate = sr)()

    audio = normalize(audio)

    #Plotting librosa extraction:

    C = 2000 * (sr/44100) #scaling wrt sr for offset since it's the energy computed in a frame
    limitssep, pitchdiscsep = extractpitchlimitslibrosa(audio,sr,C)
    pitchcont_sep = pitchtocontinuous(pitchdiscsep, audio, sr, limitssep)
    plotonoffsetspitch(audio, sr, length, offset[i], limitssep, 'Separate extraction of ' + filename[:-4] , pitch = pitchcont_sep)
    write_melody(pitchdiscsep[sum((limitssep[:, 0] < offset[i])): sum((limitssep[:, 0] < offset[i] + length))],
                 limitssep[sum((limitssep[:, 0] < offset[i])): sum((limitssep[:, 0] < offset[i] + length))],
                 outpath='synth/' + filename[:-4] + 'sepsynth')
    print(pitchdiscsep[sum((limitssep[:, 0] < offset[i])): sum((limitssep[:, 0] < offset[i] + length))])


    plt.figure(2)

    #Plotting essentia extraction:
    hopSize = 128 #hopsize used to estimate pitch
    pitchcont_joi, pitchdisc_joi, limits_joi = extractpitchlimitsessentia(audio, sr, hopSize)
    plotonoffsetspitch(audio, sr, length, offset[i], limits_joi, 'Joint extraction of '+ filename[:-4], pitchcont_joi, hopSize)
    plotonoffsetspitch(audio, sr, length, offset[i], limits_joi, 'Joint extraction of ' + filename[:-4], pitch = pitchtocontinuous(pitchdisc_joi,audio,sr,limits_joi))
    write_melody(pitchdisc_joi[sum((limits_joi[:, 0] < offset[i])): sum((limits_joi[:, 0] < offset[i] + length))],
                 limits_joi[sum((limits_joi[:, 0] < offset[i])): sum((limits_joi[:, 0] < offset[i] + length))],
                 outpath='synth/' + filename[:-4] + 'joinsynth')
    '''
    #writing audio files:
    #MonoWriter(filename='cut/' + filename[:-4] + 'cut.wav', sampleRate=sr)(audio[offset[i] * sr : (offset[i]+length) * sr])

    #writenotes(filename[:-4], audio, sr, limitssep, pitchdiscsep)


    '''