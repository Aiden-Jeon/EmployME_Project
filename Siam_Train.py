import tensorflow as tf
import numpy as np
import pickle



def siames(features,labels,mode,params):
    hidden_dim = params['hidden_dim']
    n_layers = params['n_layers']
    dropout = params['dropout']
    batch_size = params['batch_size']

    emb = tf.get_variable(name='emb',
                        initializer=params['siam_wv'],
                        trainable=False)
    #left RNN
    stacked_rnn_left = []
    for _ in range(n_layers):
        left_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_left_cell = tf.contrib.rnn.DropoutWrapper(left_cell,output_keep_prob=dropout)
        stacked_rnn_left.append(lstm_left_cell)
    lstm_left_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_left, state_is_tuple=True)

    left_batch = tf.nn.embedding_lookup(params=emb,ids=features['left'])
    left_init = lstm_left_cell_m.zero_state(batch_size=batch_size ,dtype=tf.float32)
    with tf.variable_scope('left'):
        left_outputs,_ = tf.nn.dynamic_rnn(cell=lstm_left_cell_m,
                                           initial_state=left_init,
                                            inputs=left_batch,
                                            dtype=tf.float32)
    out1 = left_outputs[:,-1,:]
    
    if mode == tf.estimator.ModeKeys.PREDICT:           
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'left':out1})
    else:        
        #right RNN
        stacked_rnn_right = []
        for _ in range(n_layers):
            right_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0, state_is_tuple=True)
            lstm_right_cell = tf.contrib.rnn.DropoutWrapper(right_cell,output_keep_prob=dropout)
            stacked_rnn_right.append(lstm_right_cell)
        lstm_right_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_right, state_is_tuple=True)

        right_batch = tf.nn.embedding_lookup(params=emb,ids=features['right'])
        right_init = lstm_right_cell_m.zero_state(batch_size=batch_size ,dtype=tf.float32)
        with tf.variable_scope('right'):
            right_outputs,_ = tf.nn.dynamic_rnn(cell=lstm_right_cell_m, 
                                                initial_state=right_init,
                                                inputs=right_batch, 
                                                dtype=tf.float32)
        out2 = right_outputs[:,-1,:]
        
        #distance define
        distance = tf.sqrt(tf.reduce_sum(
                                        tf.square(tf.subtract(out1,out2)),
                                        1,keep_dims=True))
        distance = tf.div(distance,
                        tf.add(tf.sqrt(tf.reduce_sum(tf.square(out1),
                                                    1,keep_dims=True)),
                                tf.sqrt(tf.reduce_sum(tf.square(out2),
                                                    1,keep_dims=True))))
        distance = tf.reshape(distance, [-1], name="distance")
        #loss definition
        def contrastive_loss(y,d,batch_size):
            temp = y *tf.square(d)
            temp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
            return tf.reduce_sum(temp +temp2)/batch_size/2

        #loss
        loss = contrastive_loss(labels,distance,batch_size)
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss,global_step)

        #acc
        sim = tf.subtract(tf.ones_like(distance),tf.rint(distance)) #auto threshold 0.5
        accuracy=tf.metrics.accuracy(labels,sim)
        eval_metric_ops = {'acc':accuracy}
        return tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss,
                eval_metric_ops=eval_metric_ops)



with open('./Data/siam_pre.pickle', 'rb') as f:
    siam_pre = pickle.load(f)

one = siam_pre['one']
two = siam_pre['two']
full = siam_pre['full']

params = {'siam_wv':np.array(siam_pre['wv_wv'],dtype=np.float32),
        'hidden_dim' : 512,
        'n_layers':2,
        'batch_size':200,
        'dropout':0.5}

model = tf.estimator.Estimator(siames,params=params,model_dir='./Model/new_siam')

one_input = tf.estimator.inputs.numpy_input_fn(
    x = {"left":one['ans_pad'],
        "right":one['adv_pad']},
    y = one['aug_labels'],
    num_epochs=5,
    batch_size=200,
    shuffle=False)

model.train(one_input)

two_input = tf.estimator.inputs.numpy_input_fn(
    x = {"left":two['ans_pad'],
        "right":two['adv_pad']},
    y = two['aug_labels'],
    num_epochs=5,
    batch_size=200,
    shuffle=False)

model.train(two_input)

full_input = tf.estimator.inputs.numpy_input_fn(
    x = {"left":full['ans_pad'],
        "right":full['adv_pad']},
    y = full['aug_labels'],
    num_epochs=5,
    batch_size=200,
    shuffle=False)

model.train(full_input)