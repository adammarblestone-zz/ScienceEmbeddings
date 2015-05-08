import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging
from nltk.corpus import stopwords

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

def getSentence(label, it):
    n = next(f["words"] for f in it if f["labels"] == label)
    return n

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

    print "Loading all the sentences..."
    allTheSentences = SentenceList(indir + subdir)

    test_paragraphs = ["The existence of a canonical cortical microcircuit can be inferred from neuroanatomy."]

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
		print getSentence(k[0], allTheSentences)
	    else: # if it is a single word
		print k[0]
        print "\n"

if __name__ == '__main__':
    main()
