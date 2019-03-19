import numpy as np
import copy

import random

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
bufferLength = 10
maxStep = 20
from keras.models import Sequential, Model, load_model
import keras
class environment:
    def __init__(self):
        global problems
        self.action_space = np.zeros( (2,) )
        self.observation_space = np.zeros( (bufferLength*2,) )
        self.task_space = np.zeros((2,))

def action_to_readable(action):
    #coordinate = np.argmax(action)
    #action = normalize_list(action)
    coordinate = np.random.choice(len(action[0]), 1, p=action[0])[0]
    ind = coordinate
    #ind = np.argmax(action)
    return ind

class RNNModel:
    def __init__(self):
        self.model = load_model('trained_RNN')
        self.model.compile(loss = 'mse', optimizer='adam')

    def state_to_input(self, state):
        res = copy.deepcopy(state)
        for i in range(len(res)):
            for j in range(len(res[i])):
                res[i][j] += 1
            while (len(res[i]) < bufferLength):
                res[i].append(0)
        return [res[0] , res[1]]

    def state_to_RNN_input(self, state):
        temp = self.state_to_input(state)
        temp[0].append(-1)
        temp[1].append(-1)
        temp = keras.preprocessing.sequence.pad_sequences( temp,  maxStep, 'float32','post','post', value =-100)
        return temp[0], temp[1]

    def get_action(self, state, task):
        translated = self.state_to_RNN_input(state)
        return self.model.predict( [ translated[0].reshape(1, len(translated[0]), 1), translated[1].reshape(1, len(translated[1]), 1) ] )

class MLPModel:
    def __init__(self):
        self.model = load_model('model_a_10_new')
        self.model.compile(loss = 'mse', optimizer='adam')

    def state_to_MLP_input(self, state):
        res = copy.deepcopy(state)
        for i in range(len(res)):
            for j in range(len(res[i])):
                res[i][j] += 1
            while (len(res[i]) < bufferLength):
                res[i].append(0)
        return np.array([res[0] + res[1]])

    def get_action(self, state, task):
        translated = self.state_to_MLP_input(state)
        return self.model.predict( [ translated,  task ] )

def action_to_readable(action):
    #coordinate = np.argmax(action)
    #action = normalize_list(action)
    coordinate = np.random.choice(len(action[0]), 1, p=action[0])[0]
    ind = coordinate
    #ind = np.argmax(action)
    return ind


#convert the observation to matrix used by NN training
def initialize_state():

    return [[],[]]

def state_to_input(state):
    res = copy.deepcopy(state)
    for i in range( len(res) ):
        for j in range( len(res[i])):
            res[i][j] += 1
        while( len(res[i])<bufferLength ):
            res[i].append(0)
    return np.array([res[0] + res[1]])

def working(state):
    A = 0
    B = 1
    p0a = 0.5
    p0b = 1
    if(A in state[0]): p0b*=0.5

    p1a = 0.6
    p1b = 0.6
    if(not B in state[1]): p1a*=0.5

    curP = random.random()
    if( len(state[0])>0 ):
        if(state[0][0] == A):
            if(curP < p0a): state[0].pop(0)
        else:
            if(curP < p0b): state[0].pop(0)
    if (len(state[1]) > 0):
        if (state[1][0] == A):
            if (curP < p1a): state[1].pop(0)
        else:
            if (curP < p1b): state[1].pop(0)

def update_state_by_action(state, action, task):
    if( len(state[action])>=bufferLength ): return 0
    else: state[action].append(task)
    return 1

def task_to_matrix(task):
    res = [0,0]
    res[task] = 1
    return np.array([res])

def main():
    model = MLPModel()
    random.seed()
    total_reward = 0
    f = open('evaluated.txt','w')
    temp = 0
    testRound = 10000
    for round in range(testRound):
        if(round%1000==0)and(round!=0):
            print(temp/round)
        cur_state = initialize_state()
        I = 1.0

        total_reward = 0

        for ind in range(40):

            for task_ind in range(2):
                new_task = random.randrange(2)
                action = model.get_action( cur_state, task_to_matrix(new_task))
                #if( len(cur_state[0]) > len(cur_state[1])): action_translated = 1
                #else: action_translated = 0

                if(len(action)>1):
                    action_translated = action_to_readable(action[new_task])
                else:
                    action_translated = action_to_readable(action)

                if(len( cur_state[action_translated] ) >= bufferLength):
                    action_translated = 1 - action_translated

                new_state = copy.deepcopy(cur_state)
                reward = update_state_by_action(new_state, action_translated, new_task)

                cur_state = new_state
                total_reward += reward
                if (reward == 0): break
            working(cur_state)
            if (reward == 0): break
        temp += total_reward
        f.write(str(total_reward))
        f.write('\n')
    f.close()
    print('avg reward: ', temp/testRound)



if __name__ == "__main__":
    main()