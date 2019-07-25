import pickle
import tensorflow as tf
import numpy as np 
from siamese_model import siamese

with open('./Data/aug_wv_save.pickle', 'rb') as f:
    wv_model = pickle.load(f)

params = {
    'wv_wv': np.array(list(wv_model['wv_wv'].values()),dtype=np.float32),
    'hidden_dim' : 512,
    'n_layers':2,
    'dropout':0.5,
    'batch_size':10}

model = tf.estimator.Estimator(siamese,params=params,model_dir='./Model/')

with open('./Data/aug_pad.pickle', 'rb') as f:
    data = pickle.load(f)

labels = np.array(data['labels'],dtype=np.float32)
answer = data['ans_pad']
advice = data['adv_pad']

#train
train_input = tf.estimator.inputs.numpy_input_fn(
    x={'left':answer,  
      'right':advice},
    y=labels,
    num_epochs=1,
    batch_size=10, #should match with params batch_size
    shuffle = True 
)

model.train(train_input)

#eval
#I will evaluate data 0 to 10 index
eval_input = tf.estimator.inputs.numpy_input_fn(
    
    x={'left':answer[:10], 
      'right':advice[:10]},
    y=labels[:10],
    num_epochs=1,
    batch_size=10,
    shuffle = False
)

print(model.evaluate(eval_input))
#{'acc': 0.7, 'loss': 0.10585511, 'global_step': 3}
#predict
##I will predict data 0 to 10 index
pred_input = tf.estimator.inputs.numpy_input_fn(
    x={'left':answer[:10]},
    num_epochs=1,
    batch_size=10,
    shuffle = False
)

pred = model.predict(pred_input)
result = []
for p in pred:
    result.append(p['left'])
print(result)
