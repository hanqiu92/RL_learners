import numpy as np
import tensorflow as tf
from AbstractLearner import AbstractLearner

class PDPO(AbstractLearner):
    '''
    Proximal Deterministic Policy Optimization learner, combining design of PPO and DDPG.
    '''

    def __init__(self,**kwargs):
        super(PDPO,self).__init__(**kwargs)
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        c_lr = kwargs.get('C_LR',0.0001) # learning rate for critic
        a_lr = kwargs.get('A_LR',0.0001) # learning rate for actor
        self.gamma = kwargs.get('gamma',0.95) # discount factor
        self.tau = kwargs.get('tau',0.01) # target network update rate
        self.sig = kwargs.get('sig',0.1) # action exploration standard deviation
        epsilon = kwargs.get('epsilon',0.1) # trust region size

        q_net = lambda s,a:tf.layers.dense(s,1)[:,0]
        a_net = lambda s:tf.layers.dense(s,1)[:,0]

        self.qvalue_net = kwargs.get('q_net',q_net) # q net
        self.act_net = kwargs.get('a_net',a_net) # act net

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None], 'action')
            self.tfy = tf.placeholder(tf.float32, [None], 'q_target')

            with tf.variable_scope('actor'):
                with tf.variable_scope('update'):
                    self.a = self.act_net(self.tfs)
                with tf.variable_scope('previous'):
                    self.al = self.act_net(self.tfs)
                with tf.variable_scope('target'):
                    self.ap = self.act_net(self.tfs)
                a_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/update')
                al_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/previous')
                ap_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/target')

            with tf.variable_scope('critic'):
                with tf.variable_scope('update'):
                    self.q_critic = self.qvalue_net(self.tfs,self.tfa)
                with tf.variable_scope('update',reuse=True):
                    self.q_act = self.qvalue_net(self.tfs,self.a)
                with tf.variable_scope('update',reuse=True):
                    self.q_act_clip = self.qvalue_net(self.tfs,self.al +
                                                      tf.clip_by_value(self.a - self.al,
                                                                      -epsilon,
                                                                      epsilon))
                with tf.variable_scope('target'):
                    self.qp = self.qvalue_net(self.tfs,self.ap)
                q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/update')
                qp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target')

            self.closs = tf.reduce_mean(tf.square(self.tfy - self.q_critic))
            self.ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(self.closs,var_list=q_params)

            self.aloss = -tf.reduce_mean(tf.minimum(self.q_act,self.q_act_clip))
            self.atrain_op = tf.train.AdamOptimizer(a_lr).minimize(self.aloss,var_list=a_params)

            self.update_previous_op = [pp.assign(p) for p, pp in zip(a_params,al_params)]
            self.update_target_op = [pp.assign((1-self.tau)*pp+self.tau*p) for p, pp in zip(a_params + q_params,ap_params + qp_params)]

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def learn(self):
        state,action,reward,next_state = self.get_experience_sample()
        if state is not None:
            y = reward + self.gamma * self.sess.run(self.qp,{self.tfs:next_state})
            _,a_loss = self.sess.run([self.atrain_op, self.aloss], {self.tfs: state})
            _,c_loss = self.sess.run([self.ctrain_op, self.closs], {self.tfs: state, self.tfa: action, self.tfy: y})
            self.sess.run(self.update_target_op)
            self.sess.run(self.update_previous_op)

            super(PDPO,self).learn()

    def get_action(self, state, typ):
        states = np.stack([state],axis=0)
        action = self.sess.run(self.a, {self.tfs: states})
        if typ == 'train':
            action = action + np.random.normal(loc=0,scale=self.sig,size=action.shape)
        return action[0]
