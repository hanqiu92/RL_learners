import numpy as np
import tensorflow as tf
from AbstractLearner import AbstractLearner

class PG(AbstractLearner):
    '''
    Simple policy gradient learner based on REINFORCE.
    '''

    def __init__(self,**kwargs):
        #########################################
        ## read inputs
        super(PG,self).__init__(**kwargs)
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        lr = kwargs.get('LR',0.001) # learning rate for actor
        self.gamma = kwargs.get('gamma',0.95) # discount factor

        def a_net(s):
            mu = tf.layers.dense(s,1)[:,0]
            sig = tf.nn.softplus(tf.dense(s,1))[:,0]
            pi = tf.distributions.Normal(mu,sig)
            return pi
        self.act_net = kwargs.get('a_net',a_net) # act net

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None], 'action')
            self.tfdcr = tf.placeholder(tf.float32, [None], 'discounted_reward')

            pi = self.act_net(self.tfs)
            self.probs = pi.probs
            self.sample_a = pi.sample()
            self.logprob = pi.log_prob(self.tfa)
            self.loss = - tf.reduce_sum(self.tfdcr*self.logprob)
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def learn(self):
        states = []
        actions = []
        dcrs = []
        for exp_pool_id in len(self.exp_pool):
            state,action,reward,_ = self.get_experience(exp_pool_id)
            discount = np.power(self.gamma,np.arange(len(reward)))
            dcr = (reward * discount)[::-1].cumsum()[::-1]
            states += [state]
            actions += [action]
            dcrs += [dcr]
        states = np.concatenate(states,axis=0)
        actions = np.concatenate(actions,axis=0)
        dcr_mean = np.mean(np.stack(dcrs,axis=0),axis=0)
        dcrs = np.concatenate([dcr - dcr_mean for dcr in dcrs],axis=0)
        _,loss = self.sess.run([self.train_op, self.loss], {self.tfs: states, self.tfa: actions, self.tfdcr: dcrs})
        super(PG,self).learn()
        self.clear_experience()

    def get_action(self, state, typ):
        states = np.stack([state],axis=0)
        action = self.sess.run(self.sample_a, {self.tfs: states})
        return action[0]
