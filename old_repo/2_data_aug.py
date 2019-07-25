import pickle
import pandas as pd
import numpy as np
import random

with open('./Data/pos_tag.pickle', 'rb') as f:
    pos_tag = pickle.load(f)

answer_tag = pos_tag['answer_tag']
advice_tag = pos_tag['advice_tag']
labels = pos_tag['labels']

data = np.array([labels,answer_tag,advice_tag]).T
bad = data[data[:,0] == 1]
good = data[data[:,0] == 0]

b_aug_label = []
b_aug_answer = []
b_aug_advice = []
for i in range(bad.shape[0]):
    i = i%good.shape[0]
    r = random.sample(list(range(good.shape[0])),1)[0]
    if r != i:
        b_aug_label.append(0)
        b_aug_answer.append(bad[i,1])
        b_aug_advice.append(good[r,2])

b_aug_label = np.array(b_aug_label)
b_aug_answer = np.array(b_aug_answer)
b_aug_advice = np.array(b_aug_advice)

aug_bad = np.array([b_aug_label,b_aug_answer,b_aug_advice]).T

g_aug_label = []
g_aug_answer = []
g_aug_advice = []
for i in range(good.shape[0]):
    i = i%bad.shape[0]
    r = random.sample(list(range(bad.shape[0])),1)[0]
    if r != i:
        g_aug_label.append(0)
        g_aug_answer.append(good[i,1])
        g_aug_advice.append(bad[r,2])

g_aug_label = np.array(g_aug_label)
g_aug_answer = np.array(g_aug_answer)
g_aug_advice = np.array(g_aug_advice)

aug_good = np.array([g_aug_label,g_aug_answer,g_aug_advice]).T

data[:,0] = 1

data_df = pd.DataFrame(data)
aug_good_df = pd.DataFrame(aug_good)
aug_bad_df = pd.DataFrame(aug_bad)
aug_data = pd.concat([data_df,aug_good_df,aug_bad_df])
aug_data = aug_data.sample(frac=1.0)

aug_labels = aug_data[0].values
aug_answer = aug_data[1].values
aug_advice = aug_data[2].values

aug_data = {
    'aug_labels' :aug_labels,
    'aug_answer' :aug_answer,
    'aug_advice' :aug_advice
}

with open('./Data/aug_data.pickle', 'wb') as f:
    pickle.dump(aug_data, f)