############################################################################
##########Prepare data for the model training###############################
######### text data ===> 2D matrix #########################################
######## target ==> 3D matrix ##############################################
############################################################################
from collections import OrderedDict
from nltk import word_tokenize
import json
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
# import pandas as pd
from sklearn.utils import class_weight

tag2index={"O":0, "B_H":1, "I_H":2, "B_O":3, "I_O":4, "B_T":5, "I_T":6}
TRAIN_FILE = 'data/train_data_doc.json'
DEV_FILE = 'data/dev_data_doc.json'
TEST_FILE='data/test_data_doc.json'
INPUT_FILE='data/dse_.json'

ftp = open('data_/config.json')
config = json.load(ftp)
maxlen = config['maxlen']
vocab_sz = config['vocab_sz']
num_recs = config['num_recs']

ftp_word2index=open('data_/word2index.json')
word2index=json.load(ftp_word2index)


def load_data(jsonfile,num_recs):

    # construct data and target matrix.
    X = np.empty((num_recs, ), dtype = list)
    y = np.empty((num_recs,), dtype = list)
    yn = np.empty((num_recs,), dtype = list)
    Xwords = np.empty((num_recs,), dtype=list)
    i = 0
    with open(jsonfile, 'r+')as ftp:
        data = json.load(ftp)
        for d in data:
            for sent_id, sent_dict in d.items():
                word2index_list = []
                sent = []
                tag_list = []
                tagn = []
                sent_d = { int(k):v for k,v in sent_dict.items() }
                for token_id, tag_tupel in sorted(sent_d.items()):
                    tag = tag2index[tag_tupel[1]]
                    tagn.append(tag)
                    tag = to_categorical(tag, num_classes=7)
                    tag_list.append(tag.flatten().tolist())
                    word = tag_tupel[0].lower()
                    sent.append(word)
                    if word in word2index:
                        wid = word2index[word]
                        word2index_list.append(wid)
                    else :
                        word2index_list.append(word2index['UNK'])
                X[i] = word2index_list
                y[i] = tag_list
                yn[i] = tagn
                Xwords[i] = sent
                i += 1
        ftp.close()
    return X, y,yn,Xwords

xs, ys, yn,xwords=load_data(INPUT_FILE,num_recs=num_recs)
train_xs, train_ys, train_yn ,train_words= load_data(TRAIN_FILE,num_recs=5736)
test_xs, test_ys, test_yn, test_words = load_data( TEST_FILE,num_recs=1240)
dev_xs, dev_ys, dev_yn, dev_words = load_data( DEV_FILE, num_recs=1172)


def freq_targets(y):
    unique, count = np.unique(y, return_counts=True)
    d = dict(zip(unique, count))
    print(d)


def pretty_print(X, y, yn, xtext,i):
    print('input size X', X.shape)
    print('target size y',y.shape)
    print('target yn',yn.shape)
    print('text shape', xtext.shape)
    print('\n')
    print(X[i])
    print(y[i])
    print(yn[i])
    print(xtext[i])

def count_zero_tags(yn):
    zero_lines=[]
    for i in range(np.shape(yn)[0]):
        if not np.any(yn[i]):
            zero_lines.append(i)
    print(zero_lines)
    print(len(zero_lines))
    return zero_lines

def delete_zero_axis(X,y,yn,Xwords):
    zero_lines=count_zero_tags(yn)
    index_list=[]
    for i , index in enumerate(zero_lines):
        index=index-i
        index_list=[]
        X= np.delete(X,index,axis=0)
        y= np.delete(y,index,axis=0)
        yn=np.delete(yn,index,axis=0)
        Xwords=np.delete(Xwords,index,axis=0)
    print(X[-1])
    print(y[-1])
    print(yn[-1])
    print(Xwords[-1])
    return X,y,yn,Xwords

trainxs,trainys,trainyn,trainwords =delete_zero_axis(train_xs,train_ys,train_yn,train_words)
testxs,testys,testyn,testwords =delete_zero_axis(test_xs,test_ys,test_yn,test_words)
devxs,devys,devyn,devwords=delete_zero_axis(dev_xs,dev_ys,dev_yn,dev_words)


Xtrain = pad_sequences(trainxs, maxlen=maxlen, padding='post')
ytrain = pad_sequences(trainys, maxlen=maxlen, padding='post') # one hot target
ytrainn = pad_sequences(trainyn, maxlen=maxlen, padding='post')

#for computing sample_weights
ytrain_weight= pad_sequences(trainyn, maxlen=maxlen, padding='post', value=-1)

Xtest = pad_sequences(testxs, maxlen=maxlen, padding='post')
ytest = pad_sequences(testys, maxlen=maxlen, padding='post') # one hot
ytestn = pad_sequences(testyn, maxlen = maxlen, padding='post')


Xdev = pad_sequences(devxs, maxlen=maxlen, padding='post')
ydev = pad_sequences(devys, maxlen=maxlen, padding='post') # one hot.
ydevn = pad_sequences(devyn, maxlen=maxlen, padding ='post')




if __name__ == '__main__':
    pretty_print(trainxs,trainys,trainyn,trainwords,3000)
    pretty_print(testxs, testys, testyn, testwords,73)
    pretty_print(devxs, devys, devyn, devwords,129)
    freq_targets(ytrainn)
    freq_targets(ytestn)
    freq_targets(ydevn)

    #
    print(Xtrain[-1])
    print(ytrain[-1])
    print(ytrainn[-1])
    print(trainwords[-1])
    np.save('data/Xtrain',arr=Xtrain)
    np.save('data/ytrain',arr=ytrain)
    np.save('data/ytrainn',arr=ytrainn)

    np.save('data/Xtest',arr=Xtest)
    np.save('data/ytest',arr=ytest)
    np.save('data/ytestn',arr=ytestn)

    np.save('data/ytrain_weight',arr=ytrain_weight)

    np.save('data/Xdev',arr=Xdev)
    np.save('data/ydev',arr=ydev)
    np.save('data/ydevn',arr=ydevn)
    #
    np.save('data/trainwords',arr=trainwords)
    np.save('data/testwords',arr=testwords)
    np.save('data/devwords',arr=devwords)