import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda,Activation, Concatenate
from keras.layers.merge import Add, Multiply, multiply
from tensorflow.python.client import device_lib
from keras.optimizers import Adam
import keras.backend as K
from analyze.analyze_utils.testbed import get_concept_node_assignment, get_all_users
from classes.graph.ising_graph import IsingGraph
from classes.observation.observation import Observation
from utils.tools.tree_operations import verify_lowest_0_pair_is_found
from classes.graph.graph_utils import get_all_lower_level_nodes, get_all_upper_level_nodes
from classes.concept_node.concept_node import compare_two_concept_nodes
from modules.dynamic_programming.value_function_approximation.module_graph.actionTranslator import *
import copy

#import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
bufferLength = 5
class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.critic_lr = 0.00003
        self.actor_lr = 0.00002
        self.critic_opt = tf.train.GradientDescentOptimizer(0)
        self.actor_opt = tf.train.GradientDescentOptimizer(0)
        self.target_critic_opt = tf.train.GradientDescentOptimizer(0)
        self.target_actor_opt = tf.train.GradientDescentOptimizer(0)
        self.gamma = .95
        self.tau = .125
        self.memorySize = 4000
        self.batchSize = 40
        self.gradient_clip_thres = 2
        self.temp_critic_opt = tf.train.GradientDescentOptimizer(self.critic_lr/self.batchSize)
        self.temp_actor_opt = tf.train.GradientDescentOptimizer(self.actor_lr/self.batchSize)
        self.updateInterval = 40
        self.trainCounter = 0
        self.memory = []
        self.actor_state_input, self.actor_task_input, self.actor_model = self.create_actor_model(self.actor_opt)
        self.temp_actor_state_input, self.temp_actor_task_input, self.temp_actor_model = self.create_actor_model(
            self.temp_actor_opt)
        _, _, self.target_actor_model = self.create_actor_model(self.target_actor_opt)
        #actor_model_weights = self.actor_model.trainable_weights

        self.critic_state_input,  \
        self.critic_model = self.create_critic_model(self.critic_opt)
        self.temp_critic_state_input, \
        self.temp_critic_model = self.create_critic_model(self.temp_critic_opt)
        _, self.target_critic_model = self.create_critic_model(self.target_critic_opt)

        self.update_critic = []
        for variable, target in zip(self.critic_model.trainable_weights, self.target_critic_model.trainable_weights):
            self.update_critic.append(tf.assign(target,  self.tau * variable + (1 - self.tau) * target ))

        self.update_actor = []
        for variable, target in zip(self.actor_model.trainable_weights, self.target_actor_model.trainable_weights):
            self.update_actor.append(tf.assign(target, self.tau * variable + (1 - self.tau) * target ))

        self.update_temp_critic = []
        for variable, target in zip(self.critic_model.trainable_weights, self.temp_critic_model.trainable_weights):
            self.update_temp_critic.append(tf.assign(variable, target))

        self.update_temp_actor = []
        for variable, target in zip(self.actor_model.trainable_weights, self.temp_actor_model.trainable_weights):
            self.update_temp_actor.append(tf.assign(variable, target))

        # Initialize for later gradient calculations
        self.init_gradient()
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self, opt):

        state_input = Input(shape=self.env.observation_space.shape)
        task_input = Input(shape=self.env.task_space.shape)
        input = Concatenate()([state_input, task_input])
        h1 = Dense(64, activation='relu')(input)
        output  = Dense(self.env.action_space.shape[0], activation='softmax')(h1)

        model = Model(input=[state_input, task_input], output=output)
        #adam = Adam(lr=0.001)

        model.compile(loss="mse", optimizer=opt)
        return model.input[0], model.input[1], model


    def create_critic_model(self,opt):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(64, activation='relu')(state_input)
        output = Dense(1, activation='linear')(state_h1)
        model = Model(input=state_input , output=output)
        model.compile(loss="mse", optimizer=opt)
        return state_input, model

    def init_gradient(self):
        self.delta = tf.placeholder(tf.float32, None)
        self.action_ind = tf.placeholder(tf.int32, None)
        self.I = tf.placeholder(tf.float32, None)
        self.actor_grad_input = tf.negative( tf.math.log(self.actor_model.output[0, self.action_ind]) * self.I * self.delta )
        self.critic_grad_input = tf.negative(self.critic_model.output * self.I * self.delta)
        self.actor_grads = tf.gradients( self.actor_grad_input,
                                   self.actor_model.trainable_weights)
        self.critic_grads = tf.gradients(self.critic_grad_input,
                                         self.critic_model.trainable_weights)

        for i, g in enumerate(self.actor_grads):
            if g is not None:
                self.actor_grads[i] = tf.clip_by_norm(g, self.gradient_clip_thres)

        for i, g in enumerate(self.critic_grads):
            if g is not None:
                self.critic_grads[i] =  tf.clip_by_norm(g, self.gradient_clip_thres)

        actor_grads = zip(self.actor_grads, self.temp_actor_model.trainable_weights)
        critic_grads = zip(self.critic_grads, self.temp_critic_model.trainable_weights)

        self.train_critic = self.temp_critic_opt.apply_gradients(critic_grads)
        self.train_actor = self.temp_actor_opt.apply_gradients(actor_grads)

    def train(self):
        if len(self.memory)<self.batchSize : return
        self.trainCounter+=1
        if(self.trainCounter%self.updateInterval == 0):
            print('----------------------weight updated!----------------------------')
            self.sess.run( self.update_critic )
            self.sess.run( self.update_actor )

        for i in range(self.batchSize):
            ind = random.randrange(len(self.memory))
            (cur_state, action_ind, reward, new_state, done, I, task) = self.memory[ind]
            if (True):
                v_future = self.target_critic_model.predict(new_state)
            else:
                v_future = np.array([[0]])
            v_current = self.critic_model.predict(cur_state)
            # print(v_current, ',' , v_future)
            delta = reward + v_future[0] * self.gamma - v_current[0]
            # print('delta ', delta)
            '''
            grad_res = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.delta: delta[0],
                self.I: I,
                self.action_ind: action_ind,
            })

            grad_res2 = self.sess.run(self.actor_grads, feed_dict={
                self.actor_state_input: cur_state,
                self.delta: delta[0],
                self.mask_input: mask,
                self.I: I,
                self.action_ind: action_ind,
            })
            '''
            '''
            for grad in grad_res:
                if( np.isnan(grad).any() ):
                    print('fc up!')
            for grad in grad_res2:
                if( np.isnan(grad).any() ):
                    print('fc up!')
            '''
            # print('before train', self.critic_model.predict(cur_state))
            #oldweights = copy.deepcopy(self.actor_model.layers[1].get_weights())
            self.sess.run(self.train_critic, feed_dict={
                self.critic_state_input: cur_state,
                self.delta: delta[0],
                self.I: I,
                self.action_ind: action_ind,
            })
            # print('after train', self.critic_model.predict(cur_state))
            self.sess.run(self.train_actor, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_task_input: task,
                self.delta: delta[0],
                self.I: I,
                self.action_ind: action_ind,
            })
            #newweights = copy.deepcopy(self.actor_model.layers[1].get_weights())

        self.sess.run(self.update_temp_critic)
        self.sess.run(self.update_temp_actor)
        #print('train done')

    def remember(self, cur_state, action_ind, reward, new_state, done, I, task):
        self.memory.append( (cur_state,action_ind, reward, new_state, done, I, task))
        if ( len(self.memory)>self.memorySize ): self.memory.pop(0)

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state, task):
        '''
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return sample_action()
        '''
        return self.target_actor_model.predict( [cur_state,task] )


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

        for ind in range(20):

            for task_ind in range(2):
                #if( random.randrange(10)>8 ): continue
                new_task = random.randrange(2)
                #new_task = task_to_matrix(new_task)
                #action_translated = 0
                '''
                if(len(cur_state[0])<len(cur_state[1])):
                    action_translated = 0
                else:
                    action_translated = 1
                '''
                action_translated = random.randrange(2)
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
    print('total reward: ', temp/10000)



if __name__ == "__main__":
    main()