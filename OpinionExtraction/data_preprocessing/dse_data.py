import json
import numpy as np
data_file='data/dse_.json'
from collections import OrderedDict

with open(data_file,'r')as fp:
    data=json.load(fp=fp)
    np.random.shuffle(data)
    with open('data/train_data_doc.json','w')as tfp:
        json.dump(data[0:360],tfp)
    with open('data/test_data_doc.json','w')as testfp:
        json.dump(data[360:504],testfp)
    with open('data/dev_data_doc.json','w')as devfp:
        json.dump(data[504:],devfp)

dse_text=[]
with open('data/dse_.json', 'r+')as tfp:
    data =json.load(tfp, object_pairs_hook=OrderedDict)
    for d in data:
        for sent_id, sent_dict in sorted(d.items()):
            sent=[]
            for token_id, tag_tupel in sorted(sent_dict.items()):
                # print(tag_tupel[0])
                sent.append(tag_tupel[0])
            dse_text.append(' '.join(sent))

with open('data/dse_text.txt',encoding='utf-8', mode='w')as tf:
    for text in dse_text:
        tf.write(text+'\n')

train_text=[]
with open('data/train_data_doc.json', 'r+')as tfp:
    data =json.load(tfp, object_pairs_hook=OrderedDict)
    for d in data:
        for sent_id, sent_dict in sorted(d.items()):
            sent=[]
            for token_id, tag_tupel in sorted(sent_dict.items()):
                # print(tag_tupel[0])
                sent.append(tag_tupel[0])
            train_text.append(' '.join(sent))

with open('data/train_text.txt',encoding='utf-8', mode='w')as tf:
    for text in train_text:
        tf.write(text+'\n')

test_text=[]
with open('data/test_data_doc.json', 'r+')as tfp:
    data =json.load(tfp, object_pairs_hook=OrderedDict)
    for d in data:
        for sent_id, sent_dict in sorted(d.items()):
            sent=[]
            for token_id, tag_tupel in sorted(sent_dict.items()):
                # print(tag_tupel[0])
                sent.append(tag_tupel[0])
            test_text.append(' '.join(sent))

with open('data/test_text.txt',encoding='utf-8', mode='w')as tf:
    for text in test_text:
        tf.write(text+'\n')

dev_text=[]
with open('data/dev_data_doc.json', 'r+')as tfp:
    data =json.load(tfp, object_pairs_hook=OrderedDict)
    for d in data:
        for sent_id, sent_dict in sorted(d.items()):
            sent=[]
            for token_id, tag_tupel in sorted(sent_dict.items()):
                # print(tag_tupel[0])
                sent.append(tag_tupel[0])
            dev_text.append(' '.join(sent))

with open('data/dev_text.txt',encoding='utf-8', mode='w')as tf:
    for text in dev_text:
        tf.write(text+'\n')


print('train ',len(train_text))
print('test',len(test_text))
print('dev ',len(dev_text))
