import numpy as np
import pandas as pd
import pickle
import utils
from konlpy.tag import Twitter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

data = pd.read_csv('./Data/all_df.csv')
data = data.dropna()
#word_vectors train

twit = Twitter()
answer = data['answer'].apply(lambda x:utils.pos_tag(x,twit)).values
advice = data['advice'].apply(lambda x:utils.pos_tag(x,twit)).values
labels = data['label'].values

sentences = list(answer.copy())
sentences.extend(advice)

wv_model = Word2Vec(sentences,size=256)

wv_model.wv.save_word2vec_format('./Data/wv_model')

wv_model = KeyedVectors.load_word2vec_format('./Data/wv_model')

wv_vocab = ['<pad>']+list(wv_model.vocab.keys())
wv_dict = {}
wv_wv = {}
for i,j in enumerate(wv_vocab):
    if j == '<pad>':
        wv_dict[j] = i
        wv_wv[i] = np.array([0]*wv_model.vector_size)
    else:
        wv_dict[j] = i
        wv_wv[i] = wv_model[j]

pos_tag = {
    'answer_tag' : answer,
    'advice_tag' : advice,
    'labels' : labels
}

wv_save = {
    'wv_dict':wv_dict,
    'wv_wv':wv_wv
}

with open('./Data/aug_wv_save.pickle', 'wb') as f:
    pickle.dump(wv_save, f)

with open('./Data/pos_tag.pickle', 'wb') as f:
    pickle.dump(pos_tag, f)