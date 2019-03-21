import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda,Activation, Concatenate
from keras.layers.merge import Add, Multiply, multiply
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
import keras.backend as K
#from modules.dynamic_programming.value_function_approximation.module_graph.actionTranslator import *
import copy

import tensorflow as tf

import random
bufferLength = 5
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class DQN:


    def build_critic_network(self, opt):
        if(self.load_model):
            model = load_model('model_c_dqn')
            model.compile(optimizer = opt, loss = 'mse')
            self.eps = 0.005
            return model.output, model
        state_input = Input(shape=self.env.observation_space.shape)
        task_input = Input(shape=self.env.task_space.shape)
        input = Concatenate()([state_input, task_input])
        h1 = Dense(256, activation='relu')(input)
        h2 = Dense(256, activation='relu')(h1)
        h3 = Dense(2, activation = 'linear')(h2)
        output = h3
        model = Model(input = [state_input,task_input], output = output)
        model.compile(optimizer = opt, loss = 'mse' )
        return output,model

    def __init__(self, env, sess, load_model = False):
        self.load_model = load_model
        self.gamma = 0.9
        self.lr = 0.0003
        self.eps = 0.99999
        self.eps_decay_rate = 0.00001
        self.min_eps = 0.005
        self.critic_optimizer = Adam(self.lr)
        self.target_critic_optimizer = Adam(self.lr)
        self.buffer_sizeLim = 5000
        self.buffer_header = 0
        self.buffer = []
        self.batch_size = 250
        self.update_interval = 100

        self.sess = sess
        self.env = env

        self.critic_model_output, self.critic_model = self.build_critic_network(self.critic_optimizer)
        self.target_critic_model_output, self.target_critic_model = self.build_critic_network(self.target_critic_optimizer)

        self.update_critic = []
        for variable, target in zip(self.critic_model.trainable_weights, self.target_critic_model.trainable_weights):
            self.update_critic.append(tf.assign(target, variable))

        self.sess.run(tf.global_variables_initializer())


    def remember(self, cur_state, action_ind, reward, new_state, done,  task_old, task_new):
        if( len(self.buffer) == self.buffer_sizeLim ):
            self.buffer[ self.buffer_header ] = (cur_state, action_ind, reward, new_state, done,  task_old, task_new )
        else:
            self.buffer.append((cur_state, action_ind, reward, new_state, done,  task_old, task_new) )
        self.buffer_header = (self.buffer_header + 1 )%self.buffer_sizeLim

    def train(self):
        if( len(self.buffer) < self.batch_size ): return
        if(self.buffer_header%self.update_interval==0):
            #print('haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            print(self.eps)
            self.sess.run(self.update_critic)
        self.eps = max(self.eps-self.eps_decay_rate, self.min_eps)
        train_batch_indexes = np.random.choice(len(self.buffer), self.batch_size)
        train_batch = []
        for i in train_batch_indexes: train_batch.append(self.buffer[i])
        cur_state_data = [ ]
        action_data = []
        reward_data = []
        new_state_data = []
        done_data = []
        task_data = []
        new_task_data = []
        for cur_state, action_ind, reward, new_state, done,task_old, task_new in train_batch:
            cur_state_data.append(cur_state[0])
            action_data.append(action_ind)
            new_state_data.append(new_state[0])
            reward_data.append(reward)
            task_data.append(task_old[0])
            new_task_data.append(task_new[0])

        next_q_by_eval = self.critic_model.predict([new_state_data, new_task_data])
        next_actions = np.argmax( next_q_by_eval, axis = 1)
        next_q_by_target = self.target_critic_model.predict([new_state_data, new_task_data])
        cur_q_by_eval = self.critic_model.predict([cur_state_data, task_data])
        for ind,action_ind in enumerate(next_actions):
            best_q = next_q_by_target[ind][action_ind]
            curAction = action_data[ ind ]
            if(not done):
                cur_q_by_eval[ind][ curAction ] = best_q*self.gamma + reward_data[ind]
            else:
                cur_q_by_eval[ind][ curAction ] = 0
        self.critic_model.fit([cur_state_data, task_data], cur_q_by_eval, epochs = 1, verbose=0)

    def prediction(self, cur_state, task = None):
        if(random.random()<self.eps):
            ind = random.randrange( 2 )
            return ind
        else:
            q = self.critic_model.predict([cur_state,task])
            #print(q)
            best_action = np.argmax(q)
            return best_action


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
    return action


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
    cnt = 0
    if( len(state[0])>0 ):
        if(state[0][0] == A):
            if(curP < p0a):
                state[0].pop(0)
        else:
            if(curP < p0b):
                state[0].pop(0)
    if (len(state[1]) > 0):
        if (state[1][0] == A):
            if (curP < p1a):
                state[1].pop(0)
        else:
            if (curP < p1b):
                state[1].pop(0)

def update_state_by_action(state, action, task):
    if( len(state[action])>=bufferLength ): return 0
    else:
        state[action].append(task)
    return 1

def task_to_matrix(task):
    res = [0,0]
    res[task] = 1
    return np.array([res])

def main():
    f = open('res_DQN.txt','w')
    f.close()
    sum_for_avg = 0
    sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    #print(device_lib.list_local_devices())
    K.set_session(sess)
    env = environment()
    actor_critic = DQN(env, sess)
    holdQueue = []
    for round in range(20000):
        if( (round+1)%500 == 0):
            actor_critic.critic_model.save('model_c_dqn')
        if(round%100 == 0)and(round!=0):
            print(round)
            print(sum_for_avg/100)
            sum_for_avg = 0
        total_t = 0
        total_correct = 0
        total_incorrect = 0
        #random.seed()

        #random.shuffle(users)
        cur_state = initialize_state()
        I = 1.0
        total_waiting_time = 0
        ind = 0
        cur_state = initialize_state()
        I = 1.0
        total_reward = 0
        new_task = random.randrange(2)
        prev_task = copy.deepcopy(new_task)
        for ind in range(20):

            for task_ind in range(2):
                #if (random.randrange(10) > 8): continue

                #new_task = task_to_matrix(new_task)
                action = actor_critic.prediction(state_to_input(cur_state), task_to_matrix(new_task))
                #print(action)
                if len(cur_state[action]) >= bufferLength :
                    action = 1 - action
                # print(action)
                action_translated = action_to_readable(action)

                # print('gaga')
                # action_translated = action_to_readable(action)
                new_state = copy.deepcopy(cur_state)
                reward = update_state_by_action(new_state, action_translated, new_task)
                if(task_ind== 1): working(new_state)
                #print(cur_state)
                new_task = random.randrange(2)
                actor_critic.remember(state_to_input(cur_state), action_translated, reward, \
                                      state_to_input(new_state), reward==0,  task_to_matrix(prev_task), task_to_matrix(new_task))
                prev_task = copy.deepcopy( new_task )
                actor_critic.train()
                cur_state = copy.deepcopy(new_state)
                total_reward += reward
                I = I * actor_critic.gamma
                if(reward == 0): break
                #
            if (reward == 0): break

        f = open('res_DQN.txt','a+')
        #print('total reward: ', total_reward)
        sum_for_avg+=total_reward
        f.write(str(total_reward))
        f.write('\n')
        f.close()



if __name__ == "__main__":
    main()
    #printStates()