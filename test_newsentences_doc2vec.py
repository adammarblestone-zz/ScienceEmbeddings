import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging
from nltk.corpus import stopwords
import pickle

subdir = "neuroscience_abstracts/"
indir = "../PubMed/"
outdir = "../ScienceEmbeddingsOutputs/"

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def cleanDoc(doc):
    stopset = set(stopwords.words('english'))
    #stemmer = nltk.PorterStemmer()
    tokens = nltk.tokenize.WordPunctTokenizer().tokenize(doc)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    #final = [stemmer.stem(word) for word in clean]
    final = [word for word in clean]
    if final != []:
        final_str = "".join([final[i] + " " for i in range(len(final)-1)]) + final[len(final) - 1]
    else:
        final_str = ""
    return final_str

class SentenceList(object):
    def __init__(self, ind):
        self.indir = ind
        
    def __iter__(self):
	j = 0
        for line in iter_documents(self.indir):
            sentences = sentence_tokenizer.tokenize(line)
            for s in sentences:
                q = str(cleanDoc(s)).split()
                yield {"words":q, "labels":"SENT_" + str(j)}
		j += 1

def iter_documents(ind):
    for filename in os.listdir(ind):
        if filename[:9] == "abstracts":
            for line in open(ind + filename).readlines():
                if len(line) > 1:
                    yield line.decode('ascii', 'ignore').strip()

def add_new_labels(sentences, model): # Adapted from https://gist.github.com/zseder/4201551d7f8608f0b82b
	sentence_no = -1
	total_words = 0
	vocab = model.vocab
	model_sentence_n = len([l for l in vocab if l.startswith("SENT")])
	n_sentences = 0
	for sentence_no, sentence in enumerate(sentences):
		sentence_length = len(sentence.words)
		for label in sentence.labels:
			if label in vocab:
				vocab[label].count += sentence_length
			else:
				vocab[label] = gensim.models.word2vec.Vocab(count=sentence_length)
				vocab[label].index = len(model.vocab) - 1
				vocab[label].code = [0]
				vocab[label].sample_probability = 1.
				model.index2word.append(label)
				n_sentences += 1
				print "Added %s to the vocabulary." % label
	return n_sentences 

def main():

    print "Loading Doc2Vec model..."
    model = gensim.models.Doc2Vec.load(outdir + subdir + "doc2vec_model")

    pickle_filename = indir + subdir + "sentenceDict.pickle"
    if not os.path.isfile(pickle_filename):
    	print "Loading all the sentences..."
    	allTheSentences = SentenceList(indir + subdir)
    	print "Making a label dictionary for all the sentences..."
    	sentenceDict = {}
    	for s in allTheSentences:
	    sentenceDict[s["labels"]] = s["words"]
	print "Pickling the dictionary..."
   	pickle.dump(sentenceDict, open(pickle_filename, "wb"))
    else:
	print "Loading pickled dictionary..."
	sentenceDict = pickle.load(open(pickle_filename, "rb"))

    test_paragraphs = ["Vertical thalamocortical afferents give rise to the elementary functional units of sensory cortex, cortical columns. Principles that underlie communication between columns remain however unknown. Here we unravel these by reconstructing in vivo-labeled neurons from all excitatory cell types in the vibrissal part of rat primary somatosensory cortex (vS1). Integrating the morphologies into an exact 3D model of vS1 revealed that the majority of intracortical (IC) axons project far beyond the borders of the principal column. We defined the corresponding innervation volume as the IC-unit. Deconstructing this structural cortical unit into its cell type-specific components, we found asymmetric projections that innervate columns of either the same whisker row or arc, and which subdivide vS1 into 2 orthogonal [supra-]granular and infragranular strata. We show that such organization could be most effective for encoding multi whisker inputs. Communication between columns is thus organized by multiple highly specific horizontal projection patterns, rendering IC-units as the primary structural entities for processing complex sensory stimuli. ", "Although much interest has attended the cryopreservation of immature neurons for subsequent therapeutic intracerebral transplantation, there are no reports on the cryopreservation of organized adult cerebral tissue slices of potential interest for pharmaceutical drug development. We report here the first experiments on cryopreservation of mature rat transverse hippocampal slices. Freezing at 1.2 degrees C/min to -20 degrees C or below using 10 or 30% v/v glycerol or 20% v/v dimethyl sulfoxide yielded extremely poor results. Hippocampal slices were also rapidly inactivated by simple exposure to a temperature of 0 degree C in artificial cerebrospinal fluid (aCSF). This effect was mitigated somewhat by 0.8 mM vitamin C, the use of a more \"intracellular\" version of aCSF having reduced sodium and calcium levels and higher potassium levels, and the presence of a 25% w/v mixture of dimethyl sulfoxide, formamide, and ethylene glycol (\"V(EG) solutes\"; Cryobiology 48, pp. 22-35, 2004). It was not mitigated by glycerol, aspirin, indomethacin, or mannitol addition to aCSF. When RPS-2 (Cryobiology 21, pp. 260-273, 1984) was used as a carrier solution for up to 50% w/v V(EG) solutes, 0 degree C was more protective than 10 degrees C. Raising V(EG) concentration to 53% w/v allowed slice vitrification without injury from vitrification and rewarming per se, but was much more damaging than exposure to 50% w/v V(EG). This problem was overcome by using the analogous 61% w/v VM3 vitrification solution (Cryobiology 48, pp. 157-178, 2004) containing polyvinylpyrrolidone and two extracellular \"ice blockers.\" With VM3, it was possible to attain a tissue K(+)/Na(+) ratio after vitrification ranging from 91 to 108% of that obtained with untreated control slices. Microscopic examination showed severe damage in frozen-thawed slices, but generally good to excellent ultrastructural and histological preservation after vitrification. Our results provide the first demonstration that both the viability and the structure of mature organized, complex neural networks can be well preserved by vitrification. These results may assist neuropsychiatric drug evaluation and development and the transplantation of integrated brain regions to correct brain disease or injury."]

    labeled_test_sentences = []
    j = 0
    for line in test_paragraphs:
	sentences = sentence_tokenizer.tokenize(line.decode('ascii', 'ignore').strip())
	for s in sentences:
            q = str(cleanDoc(s)).split()
            r = gensim.models.doc2vec.LabeledSentence(words = q, labels = ["TEST_" + str(j)])
	    labeled_test_sentences.append(r)
	    j += 1

    n_sentences = add_new_labels(labeled_test_sentences, model)

    # add new rows to model.syn0, from https://gist.github.com/zseder/4201551d7f8608f0b82b
    n = model.syn0.shape[0]
    model.syn0 = np.vstack((
    model.syn0,
    np.empty((n_sentences, model.layer1_size), dtype=np.float32)))
 
    for i in xrange(n, n + n_sentences):
	np.random.seed(
	np.uint32(model.hashfxn(model.index2word[i] + str(model.seed))))
	a = (np.random.rand(model.layer1_size) - 0.5) / model.layer1_size
	model.syn0[i] = a
 
    # Set model.train_words to False and model.train_labels to True, from https://gist.github.com/zseder/4201551d7f8608f0b82b
    model.train_words = False
    model.train_lbls = True
 
    # Train
    print "Training model on test sentences..."
    num_epochs = 10
    model.alpha = 0.025
    model.min_alpha = 0.025
    for epoch in range(num_epochs):
	model.train(labeled_test_sentences)
	model.alpha -= 0.002
	model.min_alpha = model.alpha

    print "Testing the embeddings of the test sentences..."
    for s in labeled_test_sentences:
        print "The most similar items to"
	print s.words
        print "are:"
	m = model.most_similar(s.labels[0])
        print m
        print "...in other words:"
	for k in m:
	    if k[0][:5] == "SENT_":
		print sentenceDict[k[0]]
	    else: # if it is a single word
		print k[0]
        print "\n"

if __name__ == '__main__':
    main()
