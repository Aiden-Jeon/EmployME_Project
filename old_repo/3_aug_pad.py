import numpy as np
import pandas as pd
import pickle
import utils

with open('./Data/aug_data.pickle', 'rb') as f:
     aug_data = pickle.load(f)

with open('./Data/aug_wv_save.pickle', 'rb') as f:
     wv_model = pickle.load(f)

def filter_oov(sentence,dict):
    return list(filter(lambda x: x in dict.keys(),sentence))

ans_filt = list(map(lambda x:filter_oov(x,wv_model['wv_dict']),aug_data['aug_answer']))
adv_filt = list(map(lambda x:filter_oov(x,wv_model['wv_dict']),aug_data['aug_advice']))

ans_len = list(map(len,ans_filt))
adv_len = list(map(len,adv_filt))

'''
np.median(ans_len),np.percentile(ans_len,75)#(52.0, 82.0) 
np.median(adv_len),np.percentile(adv_len,75)#(32.0, 45.0)
'''

ans_max_len = 75
adv_max_len = 40

ans_pad = np.array(list(map(lambda x: utils.seq_pad(x,ans_max_len,wv_model['wv_dict']),aug_data['aug_answer'])))
adv_pad = np.array(list(map(lambda x: utils.seq_pad(x,adv_max_len,wv_model['wv_dict']),aug_data['aug_advice'])))

aug_pad = {
    'ans_pad':ans_pad,
    'adv_pad':adv_pad,
    'labels' :aug_data['aug_labels']	
    
}

with open('./Data/aug_pad.pickle', 'wb') as f:
    pickle.dump(aug_pad, f)