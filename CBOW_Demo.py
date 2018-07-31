import tensorflow as tf
import pickle
from konlpy.tag import Twitter
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
def pos_tag(x,twit = Twitter()):
    temp = twit.pos(x)
    result = list(map(lambda x: '/'.join(x),temp))
    return result

#
def word_zip(x):
    grammer = [('PreEomi','Eomi'),
              ('Noun','Suffix')]
    idx = [] 
    for i in range(len(x)-1):
        for g in grammer:
            if (x[i][1],x[i+1][1]) == g:
                x[i+1] = (x[i][0]+x[i+1][0],x[i+1][1])
                if g != grammer[2]:
                    idx.append(x[i])
                else:
                    idx.append(x[i+1])
    for i in idx:
        del x[x.index(i)]
    return x
#
def padding(sentence,word_dict): 
    result = [] 
    for x in sentence: 
        try: 
            result.append(word_dict[x]) 
        except: 
            pass 
    return result
#
def cbow_model(features,labels,mode,params):
    embed_size = params['embed_size']
    vocab_size = params['vocab_size']

    embeddings = tf.get_variable(name='embeddings',
                             initializer=tf.random_uniform([vocab_size, embed_size], -1.0, 1.0),
                             trainable=True)

    embed = tf.nn.embedding_lookup(params=embeddings,ids=features['x'])
    embed_context = tf.reduce_mean(embed, 1)

    output = tf.layers.dense(inputs=embed_context,
                         units=vocab_size,
                         trainable=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:       
        prob_soft = tf.nn.softmax(output)
        prob_sig = tf.nn.sigmoid(output)
        soft_tf = tf.nn.softmax(prob_soft * params['tf_idf'])
        sig_tf = tf.nn.softmax(prob_sig * params['tf_idf']   )

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'prob_soft':prob_soft,
                            'prob_sig':prob_sig,
                            'soft_tf':soft_tf,
                            'sig_tf':sig_tf})
    else:
        labels_onehot = tf.one_hot(labels,depth=vocab_size)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=labels_onehot))
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss,global_step)

        return tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss)

def lstm_model(features,labels,mode,params):
    embed_size = params['embed_size']
    vocab_size = params['vocab_size']
    batch_size = params['batch_size']
    hidden_size = params['hidden_size']
    n_layers = params['n_layers']
    dropout = params['dropout']
    
    embeddings = tf.get_variable(name='embeddings',
                             initializer=params['wv_weight'],
                             trainable=True)
    embed = tf.nn.embedding_lookup(params=embeddings,ids=features['x'])
    
    stacked_rnn = []
    for _ in range(n_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=dropout)
        stacked_rnn.append(lstm_cell)
    lstm_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)

    init = lstm_cell_m.zero_state(batch_size=batch_size ,dtype=tf.float32)
    
    outputs,_ = tf.nn.dynamic_rnn(cell=lstm_cell_m,
                                initial_state=init,
                                inputs=embed,
                                dtype=tf.float32)

    last_output = outputs[:,-1,:]
    dense_output = tf.layers.dense(inputs=last_output,
                                    units=vocab_size,
                                    trainable=True)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        prob_soft = tf.nn.softmax(dense_output)
        prob_sig = tf.nn.sigmoid(dense_output)

        soft_tf = tf.nn.softmax(prob_soft * params['tf_idf'])
        sig_tf = tf.nn.softmax(prob_sig * params['tf_idf'])
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'prob_soft':prob_soft,
                            'prob_sig':prob_sig,
                            'soft_tf':soft_tf,
                            'sig_tf':sig_tf})
    else:
        labels_onehot = tf.one_hot(labels,depth=vocab_size)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_output,labels=labels_onehot))
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdagradOptimizer(learning_rate = 0.05).minimize(loss,global_step)

        accuracy=tf.metrics.accuracy(labels,tf.argmax(dense_output,1))
        eval_metric_ops = {'acc':accuracy}
        return tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

#
def main(new,model):
    pos = pos_tag(new)
    pos_zip = word_zip(pos)
    new_pad = padding(pos_zip,demo_data['word_dict'])
    if len(new_pad)>5:
        new_pad = new_pad[-5:]
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x":np.array(new_pad).reshape(1,-1)},
        num_epochs=1,
        shuffle=False)
    pred = model.predict(input_fn)
    
    # select_model 1: Sigmoid, 2: Softmax, 3: LSTM
    # use_tf 1: Yes, 2: No
    if use_tf == '1':
        if select_model == '1':
            result = [o['sig_tf'] for o in pred][0]
        else:
            result = [o['soft_tf'] for o in pred][0]
    else:
        if select_model == '1':
            result = [o['prob_sig'] for o in pred][0]
        else:
            result = [o['prob_soft'] for o in pred][0]

    try:
        chuchu = np.random.choice(list(demo_data['word_dict'].keys()),5,False,result)
    except:
        result += (1-sum(result))/len(result)
        chuchu = np.random.choice(list(demo_data['word_dict'].keys()),5,False,result)
    return chuchu

def list_up(chuchu):
    result = []
    for c in chuchu:
        try:
            temp = np.random.choice(demo_data['sub_dict'][c]['w'],1,False,demo_data['sub_dict'][c]['p'])[0]
            add = c.split('/')[0]+temp.split('/')[0]
        except:
            add = c.split('/')[0]
        result.append(add)
    return result

def ui(add):
    result = '['
    for i,a in enumerate(add):
        result += str(i+1)+':'+a+', '
    result += '6: 다른 추천 보기, 7: Exit]' 
    return result

if __name__ == "__main__":
    with open('./Data/demo_data.pickle', 'rb') as f:
        demo_data = pickle.load(f)
    
    params = {
        'vocab_size':len(demo_data['word_dict']),
        'embed_size' : 256,
        'tf_idf':demo_data['tf_idf'],
        'batch_size':1,
        'hidden_size':256,
        'n_layers':2,
        'dropout':1.0,
        'wv_weight':demo_data['wv']
    }
    
    print('모델을 선택하세요. 1: Sigmoid, 2: Softmax, 3: LSTM')
    select_model = input()
    if select_model == '1':
        model = tf.estimator.Estimator(cbow_model,params=params,model_dir='./Model/sigmoid')
    elif select_model =='2':
        model = tf.estimator.Estimator(cbow_model,params=params,model_dir='./Model/softmax')
    else:
        model = tf.estimator.Estimator(lstm_model,params=params,model_dir='./Model/lstm')
    if select_model == '3':
        print('Ouput 활성화 함수를 선택하세요. 1: Sigmoid, 2: Softmax')
        select_model = input()
    print('TF-IDF를 사용할까요? 1: Yes, 2: No')
    use_tf = input()
    
    print('문장을 입력하세요:')
    new = input()
    
    chuchu = main(new,model)
    add = list_up(chuchu)
    add_ui = ui(add)
    while True:
        print('단어를 선택하세요:',add_ui)
        click = int(input())
        try:
            new+= ' '+ add[click-1]
            print('현재문장:',new)
        except:
            if click == 7:
                break
            elif click == 6:
                chuchu = main(new,model)
                add = list_up(chuchu)
                add_ui = ui(add)
                pass
            else:
                print('Error')
        chuchu = main(new,model)
        add = list_up(chuchu)
        add_ui = ui(add)