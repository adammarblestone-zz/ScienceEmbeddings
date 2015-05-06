import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
import logging
import nltk
from nltk.corpus import stopwords

subdir = "neuroscience_abstracts/"
indir = "../PubMed/"
outdir = "../ScienceEmbeddingsOutputs/" 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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

def main():
    print "Setting up access to all the sentences..."
    allTheSentences = SentenceList(indir + subdir)
    print "Making Doc2Vec model..."
    model = gensim.models.Doc2Vec(alpha = 0.025, min_alpha = 0.025)
    model.build_vocab(allTheSentences)

    num_epochs = 10
    for epoch in range(num_epochs):
	model.train(allTheSentences)
	model.alpha -= 0.002
	model.min_alpha = model.alpha

    print model.most_similar("SENT_0") # for debugging purposes

    print "Saving Doc2Vec model..."
    model.save(outdir + subdir + "doc2vec_model")

def iter_documents(ind):
    print "Reading documents..."
    for filename in os.listdir(ind):
        if filename[:9] == "abstracts":
            print "Filename: %s" % filename
            for line in open(ind + filename).readlines():
                if len(line) > 1:
                    yield line.decode('ascii', 'ignore').strip()

class SentenceList(object):
    def __init__(self, ind):
        self.indir = ind
        
    def __iter__(self):
	j = 0
        for line in iter_documents(self.indir):
            sentences = sentence_tokenizer.tokenize(line)
            for s in sentences:
                q = str(cleanDoc(s)).split()
		r = gensim.models.doc2vec.LabeledSentence(words = q, labels = ["SENT_" + str(j)])
                if j == 0: # for debugging purposes
		    print q
                yield r
		j += 1

if __name__ == '__main__':
    main()
