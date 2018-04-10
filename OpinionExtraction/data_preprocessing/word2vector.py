from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict, Counter
from nltk import word_tokenize
INPUT_FILE_TEXT='data/dse_text.txt'
WORD2VEC = 'pretrained_vectors/GoogleNews-vectors-negative300.bin'
word2vec= KeyedVectors.load_word2vec_format(WORD2VEC,binary=True)
EMBED_SIZE=300

def word_index(textfile):
    # count the max length of the sentences in the textfile.
    counter=Counter()
    fin=open(textfile,'r')
    maxlen=0
    num_recs=0
    for line in fin:
        sent=line.strip()
        words=[x.lower() for x in word_tokenize(sent)]
        if len(words)> maxlen:
            maxlen=len(words)
        for word in words:
            counter[word]+=1
        num_recs += 1
    fin.close()

    # construct the word2index, index2word dictionaries
    vocab_sz = len(counter) + 1 # UNK pseudo-word
    word2index= defaultdict(int)
    #word2index['PAD'] = 0
    word2index['UNK'] = 1
    for wid, word in enumerate(counter.most_common(len(counter))):
        word2index[word[0]] = wid+2
    index2word={v:k for k,v in word2index.items()}
    return maxlen, vocab_sz, num_recs, index2word, word2index
maxlen, vocab_sz, num_recs,index2word,word2index = word_index(INPUT_FILE_TEXT)

config_dict={'maxlen':maxlen,
              'vocab_sz':vocab_sz,
              'num_recs': num_recs,
              'embed_size':EMBED_SIZE}
import json
with open('data_/config.json','w')as fp:
  json.dump(config_dict,fp)

with open('data_/index2word.json','w')as fp:
  json.dump(index2word,fp)

with open('data_/word2index.json','w')as fp:
  json.dump(word2index,fp)

n_symbols = vocab_sz+1
embedding_weights = np.zeros((n_symbols, EMBED_SIZE))
for word, index in word2index.items():
    try:
        embedding_weights[index,:] = word2vec[word]
    except KeyError:
        pass
np.save('data_/google_word2vec.npy',arr=embedding_weights)
