import numpy as np
import pandas as pd
import re
import utils

###################################################
#call data
data = pd.read_csv('./Data/Raw_Data.csv')

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
all_df.to_csv('./Data/all_df.csv',index=False)