import essentia as es
from essentia.standard import *
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from features import *

#names =['LeadVocal_chain.mp3', 'LeadVocal_september.mp3', 'guitar_chain.mp3', 'guitar_september.mp3', 'DrumKit_chain.mp3', 'DrumKit_september.mp3', 'Bass_chain.mp3', 'Bass_september.mp3']
names =['LeadVocal_chain.mp3']
sr = 44100
C = 300
n_fft = 1024

ZCRs = []
centroids = []
bandwidths = []
inharmonicities = []
harmonics = []

for i in range(len(names)):
    print(names[i])
    ZCR, centroid, bandwidth, inharmonicity, harmonic = calculate_track_features(names[i], sr, C, n_fft)
    ZCRs.append(ZCR)
    centroids.append(centroid)
    bandwidths.append(bandwidth)
    inharmonicities.append(inharmonicity)
    harmonics.append(harmonic)


def display(feature, name, save = False):
    # we want those distributions to be as small as possible, otherwise it means we have a big variety of the features
    # across different notes of the same song, therefore the features are useless.

    plt.figure()
    plt.boxplot(feature[:,0])
    plt.title('Distribution of the means of ' + name + ' across notes')

    if save == True:
        plt.savefig('plots/boxplots/' + filename[:-4] + 'mean_' + name)
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


def display_list(featurelist, name, namelist, save=False):
    # we want those distributions to be as small as possible, otherwise it means we have a big variety of the features
    # across different notes of the same song, therefore the features are useless.
    save = True
    print(featurelist[0].shape)
    listmeans = []
    liststds = []
    for i in range(len(featurelist)):
        listmeans.append(featurelist[i][:,0])
        liststds.append(featurelist[i][:,1])

    fig1, ax1 = plt.subplots()
    ax1.boxplot(listmeans)

    ax1.set_xticklabels((namelist), rotation = 45, fontsize = 8)
    ax1.set_title('Distribution of the means of ' + name + ' across notes')
    if save == True:
        plt.savefig('plots/boxplots/mean_' + name)
    plt.show()


    fig2, ax2 = plt.subplots()
    ax2.boxplot(liststds)
    ax2.set_xticklabels((namelist), rotation = 45, fontsize = 8)
    ax2.set_title('Distribution of the standard deviations of ' + name + ' across notes')
    if save == True:
        plt.savefig('plots/boxplots/deviation_' + name)
    plt.show()

names = [name[:-4] for name in names] #removing .mp3

display_list(ZCRs, 'ZCR', names)
display_list(centroids,'centroids', names)
display_list(bandwidths,'bandwidths', names)
display_list(inharmonicities,'inharmonicity', names)
display_list([harmonic[:, 0, :] for harmonic in harmonics],'harmonic_1', names)#dim 1 is the note, dim2 is the feature, dim 3 is mean or std
display_list([harmonic[:, 1, :] for harmonic in harmonics], 'harmonic_2', names)
display_list([harmonic[:, 2, :] for harmonic in harmonics],'harmonic_3', names)
display_list([harmonic[:, 3, :] for harmonic in harmonics], 'harmonic_4', names)

