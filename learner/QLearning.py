import numpy as np
import tensorflow as tf
import random as rand
from AbstractLearner import AbstractLearner

class QLearning(AbstractLearner):
    '''
    Simple Q learner.
    '''

    def __init__(self,**kwargs):
        #########################################
        ## read inputs
        super(QLearning,self).__init__(**kwargs)
        self.actions = kwargs.get('action_space',np.array([[-1],[0],[1]])) # set of actions
        self.qvalue_net = kwargs.get('q_net',lambda s,a:tf.layers.dense(s,1)[:,0]) # q net
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        a_dim = kwargs.get('A_DIM',1) # dimension size of action
        lr = kwargs.get('LR',0.0001) # learning rate
        self.gamma = kwargs.get('gamma',0.95) # discount factor

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
            self.tfq = tf.placeholder(tf.float32, [None], 'qvalue')

            self.q = self.qvalue_net(self.tfs,self.tfa)
            self.loss = tf.reduce_mean(tf.square(self.q - self.tfq))
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def get_action(self, state, typ):
        q = self.get_qvalue(state)
        return self.act_with_q(state, q, typ)

    def get_qvalue(self, state):
        states = np.stack([state for _ in range(len(self.actions))],axis=0)
        q = self.sess.run(self.q, {self.tfs: states, self.tfa: self.actions})
        return q

    def act_with_q(self, state, q, typ):
        if typ == 'infer':
            i = np.argmax(q)
        elif typ == 'train':
            # softmax exploration
            q_ = np.exp(q - max(q))
            p_ = np.cumsum(q_/sum(q_))
            i = sum(rand.random() > p_)
        else:
            i = 0
        return self.actions[i]

    def learn(self):
        state,action,reward,next_state = self.get_experience_sample()
        if state is not None:
            # calculate q value
            state_for_q = np.stack([next_state for _ in range(len(self.actions))],axis=1).reshape((len(self.actions)*len(next_state),-1))
            action_for_q = np.stack([self.actions for _ in range(len(next_state))],axis=0).reshape((len(self.actions)*len(next_state),-1))
            v = self.sess.run(self.q, {self.tfs: state_for_q, self.tfa: action_for_q}).reshape((len(next_state),len(self.actions)))
            q = reward + self.gamma * v.max(axis=-1)
            self.sess.run(self.train_op, {self.tfs: state, self.tfa: action, self.tfq: q})

            super(QLearning,self).learn()
