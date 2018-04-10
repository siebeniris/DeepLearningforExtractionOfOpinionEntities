import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
trainwords=np.load('train_words.npy')
Xtrain = np.load('Xtrain.npy')
ytrain = np.load('ytrain.npy')
ytrainn = np.load('ytrainn.npy')

ytestn=np.load('ytestn.npy')
ydevn=np.load('ydevn.npy')
ytrain_weight=np.load('ytrain_weight.npy')
print(trainwords[-1])
print(Xtrain[-1])
yd=np.apply_along_axis(np.argmax,axis=1, arr=ytrain[-1])
print(ytrain[-1])
print(np.shape(trainwords))
print(np.shape(ytestn))
print(np.shape(ydevn))

# weights=compute_sample_weight('balanced',ytrainn.flatten())
# weights_= np.resize(weights,(5827,169))

# sample_weights= np.empty((5827,), dtype = list)
# for i in range(ytrain.shape[0]):
#   w_ =np.empty((169,),dtype=list)
#   for j in range(ytrain[i].shape[0]):
#     ele = ytrain[i][j]*weights_[i][j]
#     w_[j]=ele
#   sample_weights[i]=w_

# np.save('train_weights_3d.npy',arr=sample_weights)
# print(sample_weights[169])

# weights=compute_sample_weight('balanced',ytrain_weight.flatten())
# weights_= np.resize(weights,(5736,169))
# np.save('train_weights.npy',arr=weights_)
weights_=np.load('train_weights.npy')
print(ytrainn[-1])

print(weights_[-1])

def freq_targets(y):
    unique, count = np.unique(y, return_counts=True)
    d = dict(zip(unique, count))
    print(d)

freq_targets(ytrainn)
freq_targets(ytestn)
freq_targets(ydevn)
freq_targets(ytrain_weight)
# weights= np.load('train_weights.npy')
# print(weights[-1])

# {0: 922668, 1: 5471, 2: 7499, 3: 7587, 4: 6307, 5: 3372, 6: 16480}
# {0: 199945, 1: 1267, 2: 1934, 3: 1707, 4: 1412, 5: 548, 6: 2747}
# {0: 188681, 1: 1065, 2: 1522, 3: 1492, 4: 1010, 5: 724, 6: 3574}
# {0: 116633, 1: 5471, 2: 7499, 3: 7587, 4: 6307, 5: 3372, 6: 16480, -1: 806035}