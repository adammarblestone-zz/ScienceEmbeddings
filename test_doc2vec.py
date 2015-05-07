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
    for f in it:
        if f["labels"] == label:
	    return f["words"]

def main():

    print "\nLoading Doc2Vec model...\n"
    model = gensim.models.Doc2Vec.load(outdir + subdir + "doc2vec_model")
    for s in ["neuron", "glia", "tetrode", "opsin"]:
	allTheSentences = SentenceList(indir + subdir)
        print "Most similar sentences to: %s \n" % s
        m = model.most_similar(positive=[s], topn=5)
	for k in m:
	    if k[0][:5] == "SENT_":
		print getSentence(k[0], allTheSentences)
	    else: # if it is a single word
		print k[0]
        print "\n"

if __name__ == '__main__':
    main()
