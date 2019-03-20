import math
import copy
from keras.utils import plot_model
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda,Activation, Concatenate, LSTM, Masking
from keras.layers.merge import Add, Multiply, multiply
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
import keras.backend as K
import keras
import random

def task_to_matrix(task):
    res = [0,0]
    res[task] = 1
    return np.array([res])

random.seed()
maxLen = 10
batchSize = 1000
maxStep = 30
lm = False
#keras.preprocessing.sequence.pad_sequences(test, maxLen, 'float32','post','post', value =-100)
def create_actor_model():
    if (not lm):
        state_input = Input(shape=( maxStep, 1))
        state_2_input = Input(shape=( maxStep, 1))
        masked_state = Masking( mask_value= -10 )(state_input)
        masked_2_state = Masking(mask_value=-10)( state_2_input)
        memory_layer = LSTM(128, recurrent_dropout=0.2, dropout=0.2)(masked_state)
        memory_2_layer = LSTM(128, recurrent_dropout=0.2, dropout=0.2)(masked_2_state)
        memory_output = Concatenate()([memory_layer, memory_2_layer])
        h1 = Dense(256, activation='relu')(memory_output)
        output = Dense(2, activation='softmax')(h1)
        output_2 = Dense(2, activation = 'softmax')(h1)

        model = Model(input=[state_input, state_2_input], output= [output, output_2])
        # adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=Adam(lr = 0.001, clipnorm = 1.5))
        return model
    else:
        model = load_model('model_a_rnn')
        model.compile(loss="mse", optimizer=Adam(0.01))
        return model

def generate_batch():
    batch = []
    for i in range(batchSize):
        queue= [ [], []]
        #queue2= []
        #batch.append((copy.deepcopy(queue), task))
        for j in range(30):
            if( ( len(queue[0])>= maxLen) and (len(queue[1]) >=maxLen) ): break
            for k in range(2):
                task = random.randrange(2)
                batch.append( (copy.deepcopy(queue), task_to_matrix(0) ) )
                batch.append( (copy.deepcopy(queue), task_to_matrix(1) ) )
                useQueue = random.randrange(2)
                if (len(queue[useQueue]) >= maxLen):
                    useQueue = 1 - useQueue
                if (len(queue[useQueue]) >= maxLen):
                    break

                queue[useQueue].append(task + 1)
                if(k==1):
                    if (random.random() > 0.45):
                        if (len(queue[0]) > 0): queue[0].pop(0)
                    if (random.random() > 0.45):
                        if (len(queue[1]) > 0): queue[1].pop(0)

    states = []
    tasks = []
    for i, (input,task) in enumerate(batch):
        temp = input
        temp[0] = keras.preprocessing.sequence.pad_sequences( [temp[0]], maxLen, 'float32','post','post', value =0)[0]
        temp[1] = keras.preprocessing.sequence.pad_sequences( [temp[1]], maxLen, 'float32', 'post', 'post', value=0)[0]
        merged  = np.append(temp[0], temp[1])
        states.append( merged )
        tasks.append( batch[i][1][0] )
        batch[i] = (temp, batch[i][1] )
    return states, tasks, batch

model = create_actor_model()
plot_model(model, to_file='model.png')
trained_actor_model = load_model('model_a')
trained_actor_model.compile(optimizer= Adam(0.001), loss = 'mse')

def generate_and_train():
    global trained_actor_model
    global model
    states, tasks, batch = generate_batch()
    tasks = np.array(tasks)
    labels = trained_actor_model.predict( [ states, tasks ] )
    states_0 = []
    states_1 = []
    new_labels_0 = []
    new_labels_1 = []
    '''
    for i, data in enumerate( batch ):
        queue0 = data[0][0]
        queue0 = np.append(queue0, -1)
        queue1 = data[0][1]
        queue1 = np.append(queue0, -1)
        states_0.append(queue0)
        states_1.append(queue1)
    states_0 = keras.preprocessing.sequence.pad_sequences( states_0, maxStep,  'float32', 'post', 'post', value=-100)
    states_1 = keras.preprocessing.sequence.pad_sequences( states_1, maxStep,  'float32', 'post', 'post', value=-100)
    
    '''

    for i in range( len(labels) // 2):
        ind_0 = i*2
        ind_1 = i*2+1
        states_0.append( np.array( batch[ind_0][0][0]  ) )
        states_1.append( np.array( batch[ind_0][0][1]  ) )
        new_labels_0.append( np.array(  labels[ind_0]  ) )
        new_labels_1.append(np.array( labels[ind_1]  ) )
    states_0 = keras.preprocessing.sequence.pad_sequences(states_0, maxStep, 'float32', 'post', 'post', value=-100)
    states_1 = keras.preprocessing.sequence.pad_sequences(states_1, maxStep, 'float32', 'post', 'post', value=-100)
    states_0 = states_0.reshape(len(states_0), maxStep, 1)
    states_1 = states_1.reshape(len(states_1), maxStep, 1)
    model.fit([ states_0, states_1], [ np.array(new_labels_0), np.array(new_labels_1) ], epochs = 7, batch_size = 100 )

for i in range(100):
    generate_and_train()
    model.save('trained_RNN')


