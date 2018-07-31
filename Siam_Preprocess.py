import numpy as np
import pandas as pd
import re
import utils
import pickle
import random
from konlpy.tag import Twitter
from gensim.models import Word2Vec

###################################################
#call data
data = pd.read_csv('./Data/Raw_Data.csv')

# data cleaning

#clean datas
question = data['question'].apply(utils.q_cleaning).values
advice = data['advice'].apply(utils.cleaning).values
good = data['good'].apply(utils.cleaning).values
bad = data['bad'].apply(utils.cleaning).values

#matching question and good/bad and advice
g_result = {}
b_result = {}
for q,a,g,b in zip(question,advice,good,bad):
    g_t,b_t = utils.advice_split(a)
    g_t = '/'.join(g_t)
    b_t = '/'.join(b_t)
    g_result[(g_t,g)] = q
    b_result[(b_t,b)] = q

#matched bad data to dataframe
b_list = list(b_result.items())
result = []
for temp in b_list:
    for a,b in zip(temp[0][0].split('/'),utils.gb_split(temp[0][1])):
        result.append([temp[1],b,a])

bad_qa = pd.DataFrame(np.array(result))
bad_qa[1] = bad_qa[1].apply(utils.df_cleaning)
bad_qa[2] = bad_qa[2].apply(utils.df_cleaning)


#matched good data to dataframe
g_list = list(g_result.items())
result = []
for temp in g_list:
    for a,b in zip(temp[0][0].split('/'),utils.gb_split(temp[0][1])):
        result.append([temp[1],b,a])

good_qa = pd.DataFrame(np.array(result))
good_qa[1] = good_qa[1].apply(utils.df_cleaning)
good_qa[2] = good_qa[2].apply(utils.df_cleaning)

#df column name change
bad_qa = pd.concat([pd.DataFrame(np.ones(bad_qa.shape[0],dtype=np.int32)),bad_qa],axis=1)
bad_qa.columns = ['label','question','answer','advice']
good_qa = pd.concat([pd.DataFrame(np.zeros(good_qa.shape[0],dtype=np.int32)),good_qa],axis=1)
good_qa.columns = ['label','question','answer','advice']
#labels = 0:good / labels = 1 : bad

#bad and good dataframe to csv
all_df = pd.concat([bad_qa,good_qa],axis=0)

# data augmentaion

all_df = all_df.reset_index()
del all_df['index']

temp = all_df['answer'][0]

#한개씩 묶은거 두개씩 묶은거 전체 묶은거
one_df=[]
two_df=[]
full_df=[]

for i in range(len(all_df)):
    temp = all_df.values[i]
    temp_split =temp[2].split('.')[:-1]

    for t in temp_split:
        one_df.append([temp[0],temp[1],t+'.',temp[3]])

    if len(temp_split) >=2:
        result = []
        for j in zip(*[temp_split[i:] for i in range(2)]):
            result.append(j[0] + '.' + j[1] +'.')

    for r in result:
        two_df.append([temp[0],temp[1],r,temp[3]])

    if len(temp_split) >=3:
        full_df.append(temp)

# training wordvec


twit = Twitter()
advice = all_df['advice'].apply(lambda x:utils.pos_tag(x,twit)).values
answer = all_df['answer'].apply(lambda x:utils.pos_tag(x,twit,save_puntuation=True)).values

sentences = list(answer.copy())
sentences.extend(advice)

wv_model = Word2Vec(sentences,size=256)

wv_vocab = ['<pad>']+list(wv_model.wv.vocab.keys())
wv_dict = {}
wv_wv = {}
for i,j in enumerate(wv_vocab):
    if j == '<pad>':
        wv_dict[j] = i
        wv_wv[i] = np.array([0]*wv_model.wv.vector_size)
    else:
        wv_dict[j] = i
        wv_wv[i] = wv_model.wv[j]

wv_save = {
    'wv_dict':wv_dict,
    'wv_wv':wv_wv
}

# making siam data


def making_saim_data(data):
    bad = data[data[:,0] == '1']
    good = data[data[:,0] == '0']

    b_aug = []
    for i in range(bad.shape[0]):
        i = i%good.shape[0]
        r = random.sample(list(range(good.shape[0])),1)[0]
        if r != i:
            b_aug.append([0,bad[i,1],good[r,2]])

    b_aug = np.array(b_aug)

    g_aug = []
    for i in range(good.shape[0]):
        i = i%bad.shape[0]
        r = random.sample(list(range(bad.shape[0])),1)[0]
        if r != i:
            g_aug.append([0,good[i,1],bad[r,2]])

    g_aug = np.array(g_aug)

    data[:,0] = 1

    data_df = pd.DataFrame(data)
    aug_good_df = pd.DataFrame(g_aug)
    aug_bad_df = pd.DataFrame(b_aug)
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
    return aug_data


one_arr = np.array(one_df)
two_arr = np.array(two_df)
full_arr = np.array(full_df)

one_arr = np.delete(one_arr,1,1)
two_arr = np.delete(two_arr,1,1)
full_arr = np.delete(full_arr,1,1)

one_aug = making_saim_data(one_arr)
two_aug = making_saim_data(two_arr)
full_aug = making_saim_data(full_arr)

wv_dict = {}
wv_dict['<pad>'] = 0
for i,j in enumerate(list(wv_model.wv.vocab.keys())):
    wv_dict[j]=i+1

wv_wv = wv_model.wv[wv_model.wv.vocab]

pad_wv = np.array([0]*wv_model.wv.vector_size).reshape(1,-1)

wv_wv = np.concatenate((pad_wv,wv_wv),axis=0)

# Padding

one_pos = list(map(lambda x:utils.pos_tag(x,twit,save_puntuation=True),one_aug['aug_answer']))
two_pos = list(map(lambda x:utils.pos_tag(x,twit,save_puntuation=True),two_aug['aug_answer']))
full_pos = list(map(lambda x:utils.pos_tag(x,twit,save_puntuation=True),full_aug['aug_answer']))

#one_len = list(map(len,list(one_pos)))
#two_len = list(map(len,list(two_pos)))
#full_len =  list(map(len,list(full_pos)))
#np.median(one_len),np.percentile(one_len,75) # (16.0, 22.0) 20
#np.median(two_len),np.percentile(two_len,75) #(33.0, 41.0) 35
#np.median(full_len),np.percentile(full_len,75) #(79.0, 109.0) 90

one_max_len = 20
two_max_len = 35
full_max_len = 90

one_pad = np.array(list(map(lambda x:utils.seq_pad(x,one_max_len,wv_dict),one_pos)))
two_pad = np.array(list(map(lambda x:utils.seq_pad(x,two_max_len,wv_dict),two_pos)))
full_pad = np.array(list(map(lambda x:utils.seq_pad(x,full_max_len,wv_dict),full_pos)))

#adv_len = list(map(len,advice))
#np.median(adv_len),np.percentile(adv_len,75) #(33.0, 47.0) 40

adv_max_len = 40

adv_pad = np.array(list(map(lambda x:utils.seq_pad(x,adv_max_len,wv_dict),advice)))

one_adv_pos = list(map(lambda x:utils.pos_tag(x,twit),one_aug['aug_advice']))
two_adv_pos = list(map(lambda x:utils.pos_tag(x,twit),two_aug['aug_advice']))
full_adv_pos = list(map(lambda x:utils.pos_tag(x,twit),full_aug['aug_advice']))

one_adv_pad = np.array(list(map(lambda x:utils.seq_pad(x,adv_max_len,wv_dict),one_adv_pos)))
two_adv_pad = np.array(list(map(lambda x:utils.seq_pad(x,adv_max_len,wv_dict),two_adv_pos)))
full_adv_pad = np.array(list(map(lambda x:utils.seq_pad(x,adv_max_len,wv_dict),full_adv_pos)))

one_aug['ans_pad'] = one_pad
one_aug['adv_pad'] = one_adv_pad
one_aug['ans_max_len'] = one_max_len
one_aug['adv_max_len'] = adv_max_len

two_aug['ans_pad'] = two_pad
two_aug['adv_pad'] = two_adv_pad
two_aug['max_len'] = two_max_len
two_aug['adv_max_len'] = adv_max_len

full_aug['ans_pad'] = full_pad
full_aug['adv_pad'] = full_adv_pad
full_aug['max_len'] = full_max_len
full_aug['adv_max_len'] = adv_max_len

data = pd.read_csv('./Data/Raw_Data.csv')

siam_pre = {
    'one' :one_aug,
    'two' :two_aug,
    'full':full_aug,
    'wv_dict':wv_dict,
    'wv_wv':wv_wv,
    'adv_pad':adv_pad,
    'advice':data['advice'].values
}

with open('./Data/siam_pre.pickle', 'wb') as f:
    pickle.dump(siam_pre, f)                

