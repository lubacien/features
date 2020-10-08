import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def trackstonotes_label(tracks):

    notes = np.concatenate(list(tracks.values()), axis=0)

    labels = list()
    track_names = list()
    label = 0
    for filename in tracks.keys():
        label = label + 1
        track_names.append(filename)
        for j in range(tracks[filename].shape[0]) :
            labels.append(label)
    return notes, np.array(labels), track_names

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
    print('intervariance:' +'\n' + str(intervariance))
    meanintravariance = np.mean(intravariances, axis = 0)
    print('intravariance:' +'\n'+ str(meanintravariance))
    print('ratio:' + '\n' + str(intervariance/meanintravariance))

np.set_printoptions(suppress= True)
#READ:
filereader = open('trackschainsep.pkl', 'rb')
tracks = pickle.load(filereader)
feature_variance(tracks)

#question: if we compute clusters with the same number as the number of tracks, can we refind tracks?
#plot pca in 2d

#Cluster into categories, then see which tracks TYPES appear most in those categories

notes, track_labels, track_names = trackstonotes_label(tracks)


#KMEANS CLUSTERING:
nclusters = 5
kmeans = KMeans(n_clusters = nclusters).fit(notes)# we have 25 different tracks. for each tracks, we want to know which label kmeans put on them.

'''
#see which clusters are present in the different tracks:
i = 1
for track in tracks.keys():
    print(track)
    print(kmeans.labels_[np.where(track_labels == i)])
    print(np.unique(kmeans.labels_[np.where(track_labels == i)], return_counts= True))

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    axs[0].bar(names, values)
    plt.figure()
    plt.show()
    i = i + 1
'''

#see which tracks are present in the different clusters:
for i in range(nclusters):
    print(track_labels[kmeans.labels_ == i])
    counts = np.array(np.unique(track_labels[kmeans.labels_ == i], return_counts= True))
    print(counts)
    print(counts[0].shape)
    fig, ax = plt.subplots()

    slices = []
    labels = []
    for k in range(min(5, counts.shape[1])):
        slices.append(max(counts[1]))
        labels.append(counts[0][np.argmax(counts[1])])

        counts = np.delete(counts, np.argmax(counts[1]), axis = 1)
        print(counts)

    slices.append(np.sum(counts[1]))

    plt.figure()
    labels = [str(track_names[lab-1][1])[:-4] for lab in labels]
    labels.append('others')
    ax.pie(slices, labels = labels) #[str(track_names[lab][1]) + str(track_names[lab][0]) for lab in labels] )
    plt.show()

'''    # Adapt radius and text size for a smaller pie
    patches, texts, autotexts = axs[1, 0].pie(fracs, labels=labels,
                                              autopct='%.0f%%',
                                              textprops={'size': 'smaller'},
                                              shadow=True, radius=0.5)'''


'''
#variance normalization, all variances = 1 (for pca):
notes = notes/np.std(notes, axis = 0)

#PCA:
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