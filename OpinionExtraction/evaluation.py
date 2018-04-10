from collections import Counter, defaultdict
import numpy as np
from more_itertools import consecutive_groups

import itertools



ytrainn = np.load('data/ytrainn.npy')
ytestn = np.load('data/ytestn.npy')
ydevn = np.load('data/ydevn.npy')
trainwords=np.load('data/trainwords.npy')
testwords=np.load('data/testwords.npy')
devwords = np.load('data/devwords.npy')


filepath='experiments/cnn_embed/trainable/nocrf/optimizers/sgd'
testfile =filepath +'/predict_test.npy'
devfile =filepath+'/predict_dev.npy'
trainfile =filepath+'/predict_train.npy'
test=np.load(testfile)
dev= np.load(devfile)
train=np.load(trainfile)


predicted =[test,dev,train]
target = [ytestn, ydevn,ytrainn]

# predicted=[test[-3:]]
# target=[ytestn[-3:]]

def word2indices(d):
    index2word={index: token for index,token in enumerate(d) }
    word2index = defaultdict(list)
    index_list=[]
    for index, token in index2word.items():
       if token!=0:
          if token==1 or token==2:
             word2index['holder'].append(index)
          elif token==3 or token==4:
              word2index['opinion'].append(index)
          elif token==5 or token==6:
              word2index['target'].append(index)

    word2indexlist=defaultdict(list)
    for tag, index_list in word2index.items():
        word2indexlist[tag]=[list(group) for group in consecutive_groups(index_list)]
    # print('word2index : ', word2index.items())
    # print('wprd2indexlist: ', word2indexlist.items())
    return  word2index, word2indexlist


def eval_global(t,p):
    predicted_counter_p=defaultdict(int)
    target_counter_p = defaultdict(int)
    overlap_target_p = defaultdict(int)
    overlap_predict_p =defaultdict(int)

    predicted_counter_b=defaultdict(int)
    target_counter_b=defaultdict(int)
    overlap_target_b=defaultdict(int)
    overlap_predict_b=defaultdict(int)
    for i in range(t.shape[0]):
        t_=t[i]
        p_=p[i]
        t_word2index, t_word2indexgroup= word2indices(t_)
        p_word2index, p_word2indexgroup= word2indices(p_)

        for k in ['holder','opinion','target']:
            v= t_word2index[k]
            v1=p_word2index[k]

        for k,v in p_word2indexgroup.items():
            predicted_counter_b[k] +=len(v)
            predicted_counter_p[k]+=len(v)
        for k,v in t_word2indexgroup.items():
            target_counter_b[k]+=len(v)
            target_counter_p[k]+=len(v)

        for k in ['holder','opinion','target']:
            v =t_word2indexgroup[k]
            # print('target v', v)
            v1 = p_word2indexgroup[k]
            # print('predicted v1',v1)
            counter_target_overlap_b=0
            counter_predict_overlap_b=0
            counter_target_overlap_p=[]
            counter_predict_overlap_p=[]

            vs = [x for x in itertools.chain(*v)]
            # print('target group:', vs)
            v1s = [x for x in itertools.chain(*v1)]
            # print('predict group:',v1s)
            overlapt = list(set(vs).intersection(set(v1s)))
            # print('overlapt: ', overlapt)

            for o in sorted(overlapt):
                for i, y in enumerate(v):
                    i_overlapt_len=[]
                    if o in y:
                        i_overlapt_len=[1 for o in overlapt if o in y]
                        counter_target_overlap_p.append(sum(i_overlapt_len)/len(y))
                        v.pop(i)
                        counter_target_overlap_b+=1
                for j, x in enumerate(v1):
                    j_overlapt_len=[]
                    if o in x:
                        j_overlapt_len=[1 for o in overlapt if o in x]
                        counter_predict_overlap_p.append(sum(j_overlapt_len)/len(x))
                        v1.pop(j)
                        counter_predict_overlap_b+=1

            overlap_target_p[k] += sum(counter_target_overlap_p)
            overlap_predict_p[k]+=sum(counter_predict_overlap_p)

            overlap_target_b[k]+=counter_target_overlap_b
            overlap_predict_b[k]+=counter_predict_overlap_b


    print('overlap proportional target',overlap_target_p.items())
    print('overlap proportional predicted', overlap_predict_p.items())
    print('predicted_p', predicted_counter_p.items())
    print('target_p', target_counter_p.items())
    print('==============================================')
    print('overlap binary',overlap_target_b.items())
    print('overlap predict binary ',overlap_predict_b.items())
    print('predicted_b', predicted_counter_b.items())
    print('target_b', target_counter_b.items())

    precision_proportional={}
    recall_proportional={}

    for k in ['holder','opinion','target']:
        t=target_counter_p[k]
        p=predicted_counter_p[k]
        ovp=overlap_predict_p[k]
        ovt = overlap_target_p[k]
        if p!=0:
            precision_proportional[k]=round(float(ovp)/p,4)*100
        else: precision_proportional[k]=0
        if t!=0:
            recall_proportional[k]= round(float(ovt)/t,4)*100
        else: recall_proportional[k]=0

    f1_score_proportional={}
    for k in ['holder','opinion','target']:
        pp=precision_proportional[k]
        rp=recall_proportional[k]
        if pp!=0 and rp!=0:
            f1_score_proportional[k]= round(2./(1./pp+1./rp),2)
        else:f1_score_proportional[k]=0

    precision_binary={}
    recall_binary={}
    for k in ['holder','opinion','target']:
        t=target_counter_b[k]
        p=predicted_counter_b[k]
        ovp=overlap_predict_b[k]
        ovt=overlap_target_b[k]
        if p!=0:
            precision_binary[k]=round(float(ovp)/p,4)*100
        else: precision_binary[k]=0
        if t!=0:
            recall_binary[k] = round(float(ovt)/t,4)*100
        else: recall_binary[k]=0

    f1_score_binary={}
    for k in ['holder','opinion','target']:
        pp=precision_binary[k]
        rp=recall_binary[k]
        if pp!=0 and rp!=0:
            f1_score_binary[k] = round(2./(1./float(pp)+1./float(rp)),2)
        else:f1_score_binary[k]=0

    return recall_proportional, precision_proportional,f1_score_proportional, recall_binary, precision_binary, f1_score_binary




if __name__ == '__main__':
    import sys
    sys.stdout = open(filepath+ '/eval.txt','w+')

    for t, p in zip(target,predicted):
        print('=======================================================================')
        rp, pp, fp,rb,pb ,fb=eval_global(t,p)
        print('global evaluation: ')
        print('proportional ===================================================')
        print('recall ',rp.items())
        print('precision', pp.items())
        print('f1 score', fp.items())
        print('binary ======== ===================================================')
        print('recall ',rb.items())
        print('precision', pb.items())
        print('f1 score', fb.items(),'\n')


