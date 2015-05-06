import gensim
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import nltk
from nltk.corpus import stopwords

def cleanDoc(doc):
    stopset = set(stopwords.words('english'))
    #stemmer = nltk.PorterStemmer()
    tokens = nltk.tokenize.WordPunctTokenizer().tokenize(doc)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    #final = [stemmer.stem(word) for word in clean]
    final = [word for word in clean]
    final_str = "".join([f + " " for f in final])
    #print final_str
    return final_str

def main():
    subdir = "neuroscience_abstracts"
    indir = "../PubMed/"
    outdir = "../ScienceEmbeddingsOutputs/"

    # Construct the corpus
    c = abstractsCorpus(indir + subdir)

    # Save the corpus
    print "Saving the corpus."
    gensim.corpora.MmCorpus.serialize(outdir + subdir + "corpus_cleaned.mm", c)
    
def iter_documents(indir):
    print "Loading tokens into dictionary..."
    for filename in os.listdir(indir):
            print "Filename: %s" % filename
            for line in open(indir + "/" + filename).readlines():
                if len(line) > 1:
                    yield gensim.utils.tokenize(cleanDoc(line.decode('ascii', 'ignore').strip()), lower=True) # added a call to cleanDoc

class abstractsCorpus(object):
    def __init__(self, indir):
        self.indir = indir
        self.dictionary = gensim.corpora.Dictionary(iter_documents(self.indir))
        self.dictionary.filter_extremes(no_below = 1, keep_n = 30000) # check API docs for pruning params
        self.dictionary.save('dict_corpus_cleaned.dict')

    def __iter__(self):
        print "Creating vector corpus from tokens..."
        for tokens in iter_documents(self.indir):
            yield self.dictionary.doc2bow(tokens)

if __name__ == '__main__':
    main()