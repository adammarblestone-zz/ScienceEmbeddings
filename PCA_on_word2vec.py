'''Visualize semantic relations of words using PCA.'''

import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging

from matplotlib.mlab import PCA as PCA
from mpl_toolkits.mplot3d import Axes3D

subdir = "neuroscience_abstracts/"
indir = "../PubMed/"
outdir = "../ScienceEmbeddingsOutputs/" 

seed_words = ["dopamine", "GABA", "serotonin", "5HT", "acetylcholine", "glutamate","neuropeptide", "stimulator", "cognitive", "behavioral", "ethological", "genetic", "biochemical", "channel", "concentration", "dynamics", "receptor", "antibody","electrode", "Tetrode", "fMRI","fNIRS",  "EEG", "calcium","sodium", "nucleus", "axon", "soma", "dendrite", "synapse", "phosphatase","mitochondria", "connectome", "voltage", "optogenetics", "depression", "OCD","schizophrenia", "autism","synesthesia","blindness", "deafness", "ADHD", "paralysis", "mania", "anhedonia"]

def main():
    print "\nLoading Word2Vec model...\n"
    model = gensim.models.Word2Vec.load(outdir + subdir + "word2vec_model")
    model.init_sims(replace=True)

    vocab = model.index2word

    data_matrix = np.array([model[vocab[i]] for i in range(len(vocab))])
    
    print "Running PCA..."
    pca_results = PCA(data_matrix)
    
    seed_word_list = [s.lower() for s in seed_words]
    vectors = [model[s] for s in seed_word_list]
    projected_vectors = [pca_results.project(v) for v in vectors]
    

    plt.rc('legend',**{'fontsize':7})
    print "Plotting PCA results in 3D..."
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title("Principal Components of Word Vectors")
    
    import itertools
    marker = itertools.cycle(['o', '^', '*', "s", "h", "8"])
    colorList = ["r", "b", "g", "y", "k", "c", "m", "w", "HotPink", "Indigo", "Grey"]
    colors = itertools.cycle(colorList)
        
    m = marker.next()
    for i in range(len(seed_word_list)):
        col = colors.next()
        if i % len(colorList) == 0:
            m = marker.next()
        a = ax.plot([projected_vectors[i][0]], [projected_vectors[i][1]], [projected_vectors[i][2]], marker = m, markersize = 10, c = col, label = seed_words[i], linestyle = "none")
    ax.legend(numpoints = 1, loc = 5)

    print "Plotting PCA results in 2D..."
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Principal Components of Word Vectors")
        
    marker = itertools.cycle(['o', '^', '*', "s", "h", "8"])
    colorList = ["r", "b", "g", "y", "k", "c", "m", "w", "HotPink", "Indigo", "Grey"]
    colors = itertools.cycle(colorList)

    m = marker.next()
    for i in range(len(seed_word_list)):
        col = colors.next()
        if i % len(colorList) == 0:
            m = marker.next()
        a = ax.plot([projected_vectors[i][0]], [projected_vectors[i][1]], marker = m, markersize = 10, c = col, label = seed_words[i], linestyle = "none")
    ax.legend(numpoints = 1, loc = 5)

    plt.show()
            
if __name__ == '__main__':
    main()
    
    
