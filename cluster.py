import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go

def display_clusters(n_clusters, inst_labels, inst_names, cluster_labels):
    #see which tracks are present in the different clusters, make pie charts:
    if n_clusters %2 == 0:
        n= n_clusters
    else:
        n = n_clusters +1
    shape = [int(n/2), 2]
    fig, ax = plt.subplots(shape[0], shape[1],figsize = (12,12))
    r = np.arange(n).reshape(-1,2)

    tot_counts = inst_labels.shape[0]

    for i in range(n_clusters):

        counts = np.array(np.unique(inst_labels[cluster_labels == i], return_counts= True))#number of occurences of each instrument inside the cluster.

        clustercounts = sum(counts[1])

        slices = [] #number of notes in each slice
        labels = []

        for k in range(min(6, counts.shape[1])):
            slice = max(counts[1])
            slices.append(slice)
            label = counts[0][np.argmax(counts[1])]

            percent = int((slice/np.sum([inst_labels == label]) ) *100)#percentage of the instrument's notes represented in each slice
            labels.append(str(inst_names[label])[:-4] + ' ' + str(percent) + '%')#inst_names starts from 1, labels of kmeans starts from 0 -> lab-1, labels of dendrogram from 0
            counts = np.delete(counts, np.argmax(counts[1]), axis = 1)

        slices.append(np.sum(counts[1])) #'others'
        #labels = [str(inst_names[lab-1])[:-4] for lab in labels]
        labels.append('others')

        ax[np.argwhere(r == i)[0][0], np.argwhere(r == i)[0][1]].pie(slices, labels = labels, shadow = False, autopct='%.0f%%', textprops={'size': 'large'}, radius= (clustercounts/tot_counts) * n_clusters) #textprops={'size': 'smaller'}
        #[str(track_names[lab][1]) + str(track_names[lab][0]) for lab in labels] )

    plt.show()

def trackstonotes_label(tracks):
    '''
    labels is of same size as notes, and has values in the range of number of instruments.
    track_names[label] gives the instrument's name
    '''

    notes = np.concatenate(list(tracks.values()), axis=0)

    labels = list()
    inst_names = list()
    label = 0
    for filename in tracks.keys():
        inst_names.append(filename[:-4])
        for j in range(tracks[filename].shape[0]):
            labels.append(label)
        label = label + 1

    return notes, np.array(labels), inst_names


def feature_variance(tracks):

    allnotes = np.concatenate(list(tracks.values()), axis = 0)
    print(allnotes.shape)
    intravariances = np.empty((len(tracks), allnotes.shape[1]))
    i=0
    for filename in tracks.keys():
        print(filename)
        print(tracks[filename].shape)
        intravariances[i] = np.std(tracks[filename], axis = 0)
        i = i+1

    intervariance = np.std(allnotes, axis = 0)
    #print('intervariance:' +'\n' + str(intervariance))
    meanintravariance = np.mean(intravariances, axis = 0)
    #print('intravariance:' +'\n'+ str(meanintravariance))
    np.set_printoptions(suppress=True)
    print('intervariance/intravariance ratio:' + '\n' + str(intervariance/meanintravariance))




def hierarchical_clustering(notes, inst_labels, inst_names):
    # hierarchical clustering
    model = AgglomerativeClustering(distance_threshold=30000, n_clusters=None)

    print(len([np.mean(notes[inst_labels == i], axis=0) for i in range(len(inst_names))]))
    print(len([np.mean(notes[inst_labels == i], axis=0) for i in range(len(inst_names))][0]))
    model = model.fit([np.mean(notes[inst_labels == i], axis=0) for i in
                       range(len(inst_names))])  # we are clustering instruments, not notes

    print(model.n_clusters_)

    plt.title('Hierarchical Clustering Dendrogram of the instruments')
    # plot the top three levels of the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    plt.figure(figsize=(20, 30))
    dendrogram(linkage_matrix, leaf_font_size=10, labels=inst_names, leaf_rotation=0, orientation='right')

    plt.show()

def aggregate(datadir):
    songdicts = os.listdir(datadir)
    print(songdicts)
    instruments = {}
    for songdict in songdicts:
        filereader = open(datadir+'/'+songdict, 'rb')
        song = pickle.load(filereader)
        for inst in song.keys():
            if instruments.get(inst, None) is None:
                instruments[inst] = np.empty((0,19))
            instruments[inst] = np.concatenate((instruments[inst], song[inst]), axis = 0)
    return instruments

def feature_table(features):
    np.set_printoptions(precision = 4)
    fig = go.Figure(data=[go.Table(header=dict(values=['zcr_mean','zcr_std']),
                     cells=dict(values = features))])
    fig.show()

#READ FULL:
#filereader = open('instruments.pkl', 'rb')
#instruments = pickle.load(filereader)
#print os.path.getsize('instruments.pkl')
#READ SEPARATE:
instruments = aggregate('pre-results')
notes, inst_labels, inst_names = trackstonotes_label(instruments)

#Features:

feature_variance(instruments)

#PCA:
#variance normalization, all variances = 1 (for pca):
notespca = notes/np.std(notes, axis = 0)

pca = PCA().fit(notespca)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
print(pca.components_)

feature_table(pca.components_[0])
#question: if we compute clusters with the same number as the number of tracks, can we refind tracks?
#plot pca in 2d

#Cluster into categories, then see which tracks TYPES appear most in those categories


#Cluster:
print(np.unique(inst_labels))
hierarchical_clustering(notes, inst_labels, inst_names)

'''
#KMEANS CLUSTERING:
nclusters = 10
kmeans = KMeans(n_clusters = nclusters).fit(notes)# we have 25 different tracks. for each tracks, we want to know which label kmeans put on them.

display_clusters(nclusters, inst_labels, inst_names,  kmeans.labels_)

'''

'''
#PCA_PLOTTING:
#variance normalization, all variances = 1 (for pca):
notes = notes/np.std(notes, axis = 0)

pca = PCA(n_components = 2, whiten = False).fit(notes) # we whiten because the variances are very different between features.
print(pca.explained_variance_ratio_)#The amount of variance explained by each of the selected components.
print('pca components:')
print(pca.components_) #Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.


fig, (ax1) = plt.subplots()
fig2, (ax2) = plt.subplots()
scatter = ax1.scatter(pca.transform(notes)[:, 0], pca.transform(notes)[:, 1], c = kmeans.labels_)
scatter2 = ax2.scatter(pca.transform(notes)[:, 0], pca.transform(notes)[:, 1], c = track_labels)

legend1 = ax1.legend(*scatter.legend_elements(),
                    loc="upper right", title="KMeans Classes")
legend2 = ax2.legend(*scatter2.legend_elements(),
                    loc="upper right", title="tracks")
plt.show()
'''


