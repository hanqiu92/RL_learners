import numpy as np
import tensorflow as tf
import random as rand
from AbstractLearner import AbstractLearner

class DoubleDQN(AbstractLearner):
    '''
    Double Deep Q Network learner with experience replay.
    '''

    def __init__(self,**kwargs):
        #########################################
        ## read inputs
        super(DoubleDQN,self).__init__(**kwargs)
        self.actions = kwargs.get('action_space',np.array([[-1],[0],[1]])) # set of actions
        self.qvalue_net = kwargs.get('q_net',lambda s,a:tf.layers.dense(s,1)[:,0]) # q net
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        a_dim = kwargs.get('A_DIM',1) # dimension size of action
        lr = kwargs.get('LR',0.0001) # learning rate
        self.gamma = kwargs.get('gamma',0.95) # discount factor
        self.update_target_iter = kwargs.get('update_iter',200) # target network updates every * steps
        self.learn_step_counter = 0

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None, a_dim], 'action')
            self.tfq = tf.placeholder(tf.float32, [None], 'qvalue')

            with tf.variable_scope('eval'):
                self.q_eval = self.qvalue_net(self.tfs,self.tfa)
            with tf.variable_scope('target'):
                self.q_target = self.qvalue_net(self.tfs,self.tfa)
            self.loss = tf.reduce_mean(tf.square(self.q_eval - self.tfq))
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

            eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
            target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
            self.update_ops = [t.assign(e) for e, t in zip(eval_params, target_params)]

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
            q_ = np.exp(q - max(q))
            p_ = np.cumsum(q_/sum(q_))
            i = sum(rand.random() > p_)
        else:
            i = 0
        return self.actions[i]

    def learn(self):
        state,action,reward,next_state = self.get_experience_sample()
        if state is not None:
            if self.learn_step_counter % self.update_target_iter == 0:
                self.sess.run(self.update_ops)
            # calculate q value
            state_for_q = np.stack([next_state for _ in range(len(self.actions))],axis=1).reshape((len(self.actions)*len(next_state),-1))
            action_for_q = np.stack([self.actions for _ in range(len(next_state))],axis=0).reshape((len(self.actions)*len(next_state),-1))
            v_eval = self.sess.run(self.q_eval, {self.tfs: state_for_q, self.tfa: action_for_q})
            next_action_idx = v_eval.reshape((len(next_state),len(self.actions))).argmax(axis=-1) # select optimal action according to the evaluation network
            next_action = self.actions[next_action_idx]
            v = self.sess.run(self.q_eval, {self.tfs: next_state, self.tfa: next_action}) # compute future return by the target network
            q = reward + self.gamma * v
            self.sess.run(self.train_op, {self.tfs: state, self.tfa: action, self.tfq: q})
            self.learn_step_counter += 1

            super(DoubleDQN,self).learn()
