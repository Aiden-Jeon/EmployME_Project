import re
import numpy as np
def cleaning(sentence):
    sentence = sentence.replace('\\r','')
    temp = re.sub('[^가-힣a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ]',' ',sentence)
    temp = re.sub(' \s+',' ',temp)
    temp = re.sub('adv\d','',temp)
    return temp.strip()

#advice split
def advice_split(sentence):
    temp = re.sub('좋은점','__좋은점',sentence)
    temp = re.sub('아쉬운점','__아쉬운점',temp)
    temp = list(filter(lambda x:len(x)>0,temp.split('__')))
    temp = list(map(lambda x: x.strip(),temp))
    g_adv = []
    b_adv = []
    for adv in temp:
        if adv[:3] == '좋은점':
            g_adv.append(adv.strip())
        if adv[:3] == '아쉬운':
            b_adv.append(adv.strip())
    return g_adv,b_adv

#good or bad split
def gb_split(sentence):
    result = []
    while len(sentence)>0:
        try:
            find = re.search('좋은점 \d|아쉬운점 \d',sentence)
            result.append(sentence[find.start():find.end()] + ' ' + sentence[:find.start()])
            sentence = sentence[find.end():]
        except:
            break
    return result

#question cleaning
def q_cleaning(sentence):
    sentence = sentence.replace('\\r','')
    temp = re.sub('Q\d','',sentence)
    temp = re.sub('\d+자|\d+줄','',temp)
    temp = re.sub('[^가-힣]',' ',temp)
    temp = re.sub('이내|최소|이상|보기|질문|작성|띄어쓰기|포함|바이트|최대|제한|로 작성|자 이내|단락|공백|내외|필수입력사항',' ',temp)
    temp = re.sub(' \s+',' ',temp)
    return temp.strip()

#after matching to clean answers
def df_cleaning(sentence):
    temp = re.sub('좋은점 \d|아쉬운점 \d','',sentence)
    return temp.strip()



def pos_tag(sentence,twit,stem=True):
    temp = twit.pos(sentence,stem=stem)
    if stem == True:
        result = []
        for t in temp:
            if t[1] in ['Noun','Verb','Adjective']:
                result.append('/'.join(t))
    else:
        result = list(map(lambda x: '/'.join(x),temp))
    return result


def seq_pad(sequence, max_len, dic):
    result = []
    for s in sequence:
        if s in dic.keys():
            result.append(s)
    sequence = result
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
        result = list(map(lambda x: dic[x],sequence))
    else:
        result = list(map(lambda x: dic[x],sequence))   
        for _ in range(max_len-len(sequence)): 
            result.append(dic['<pad>'])
    return np.array(result[:max_len])