import numpy as np
import copy

#import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
bufferLength = 10


class environment:
    def __init__(self):
        global problems
        self.action_space = np.zeros( (2,) )
        self.observation_space = np.zeros( (bufferLength*2,) )
        self.task_space = np.zeros((2,))

def sample_action():
    #ind =  np.random.randint(len(problem_list) + len(concept_list))
    res = np.ones( (2,1) ).T
    res/=(res.size)
    return res

def normalize_list(action):
    #print(action.sum(), 'sum of action prob')
    if (action.sum() == 0):
        print('error!')
    action = action/action.sum()

    return action

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
    random.seed()
    #sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    #print(device_lib.list_local_devices())
    #K.set_session(sess)
    #env = environment()
    #actor_critic = ActorCritic(env, sess)
    total_reward = 0
    f = open('random.txt','w')
    temp = 0
    for round in range(40000):
        #print(round)
        total_t = 0
        total_correct = 0
        total_incorrect = 0
        #random.seed()

        #random.shuffle(users)
        cur_state = initialize_state()
        I = 1.0

        total_reward = 0

        for ind in range(30):

            for task_ind in range(2):
                #if( random.randrange(10)>8 ): continue
                new_task = random.randrange(2)
                #new_task = task_to_matrix(new_task)
                #action_translated = 0

                if(len(cur_state[0])<len(cur_state[1])):
                    action_translated = 0
                else:
                    action_translated = 1

                #action_translated = random.randrange(2)
                if( len(cur_state[action_translated])>=bufferLength ):
                    action_translated = 1 - action_translated
                    #print('gaga')
                #action_translated = action_to_readable(action)
                new_state = copy.deepcopy(cur_state)
                reward = update_state_by_action(new_state, action_translated, new_task)

                cur_state = new_state
                total_reward += reward
                if (reward == 0): break
            #print(cur_state)
            working(cur_state)
            if (reward == 0): break
        temp += total_reward
        f.write(str(total_reward))
        f.write('\n')
    f.close()
    print('total reward: ', temp/40000)



if __name__ == "__main__":
    main()