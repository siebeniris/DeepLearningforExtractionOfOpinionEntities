
import numpy as np

ytrainn = np.load('data/ytrainn.npy')
ytestn = np.load('data/ytestn.npy')
ydevn = np.load('data/ydevn.npy')
trainwords=np.load('data/trainwords.npy')
testwords=np.load('data/testwords.npy')
devwords = np.load('data/devwords.npy')

Xtrain=np.load('data/Xtrain.npy')
Xtest=np.load('data/Xtest.npy')
Xdev = np.load('data/Xdev.npy')

filepath='experiments/cnn_noembed/crf/08'
testfile =filepath +'/predict_test.npy'
devfile =filepath+'/predict_dev.npy'
trainfile =filepath+'/predict_train.npy'

test= np.load(testfile)
dev= np.load(devfile)
train=np.load(trainfile)
def pretty_print(data,t,p,text):
    for i in range(t.shape[0]):
        print('==============================')
        print('data',data[i])

        print('target   ',t[i])
        print('predicted',p[i])
        try:
          print('text:    ',text[i])
        except Exception:
          pass
        print()

if __name__ == '__main__':
    import sys
    sys.stdout = open(filepath+ '/eval_text.txt','w+')
    print('train dataset ==========================================')
    pretty_print(Xtrain,ytrainn,train,trainwords)
    print('test dataset ===========================================')
    pretty_print(Xtest,ytestn,test,testwords)
    print('dev  dataset ===========================================')
    pretty_print(Xdev,ydevn,dev,devwords)
