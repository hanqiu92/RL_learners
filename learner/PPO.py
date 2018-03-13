import numpy as np
import tensorflow as tf
from AbstractLearner import AbstractLearner

class PPO(AbstractLearner):
    '''
    Proximal poliy optimization learner. GAE is used for reward shaping.
    '''

    def __init__(self,**kwargs):
        #########################################
        ## read inputs
        super(PPO,self).__init__(**kwargs)
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        c_lr = kwargs.get('C_LR',0.0001) # learning rate for critic
        a_lr = kwargs.get('A_LR',0.0001) # learning rate for actor
        self.c_steps = kwargs.get('C_UPDATE_STEPS',10) # update steps for critic
        self.a_steps = kwargs.get('A_UPDATE_STEPS',10) # update steps for actor
        epsilon = kwargs.get('epsilon',0.1) # trust region size
        self.gamma = kwargs.get('gamma',0.95) # discount factor
        self.lam = kwargs.get('lam',0.9) # reward shaping factor

        v_net = lambda s:tf.layers.dense(s,1)[:,0]
        def a_net(s):
            mu = tf.layers.dense(s,1)[:,0]
            sig = tf.nn.softplus(tf.dense(s,1))[:,0]
            pi = tf.distributions.Normal(mu,sig)
            return pi

        self.value_net = kwargs.get('v_net',v_net) # value net
        self.act_net = kwargs.get('a_net',a_net) # act net

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None], 'action')
            self.tfdcr = tf.placeholder(tf.float32, [None], 'discounted_reward')
            self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')

            # critic: l = d(dcr,v)
            with tf.variable_scope('critic'):
                self.tfv = self.value_net(self.tfs)
                self.closs = tf.reduce_mean(tf.square(self.tfdcr - self.tfv))
                self.ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(self.closs)

            # actor: l = f(s,adv)
            with tf.variable_scope('actor'):
                with tf.variable_scope('pi'):
                    pi = self.act_net(self.tfs)
                    self.probs = pi.probs
                with tf.variable_scope('old_pi'):
                    oldpi = self.act_net(self.tfs)
                pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/pi')
                oldpi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/old_pi')
                self.update_oldpi_ops = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

                self.sample_a = pi.sample()
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                self.aloss = -tf.reduce_mean(tf.minimum(
                    ratio * self.tfadv,
                    tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.tfadv))
                self.atrain_op = tf.train.AdamOptimizer(a_lr).minimize(self.aloss,var_list=pi_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.vars = pi_params

    def learn(self):
        state,action,reward,next_state = self.get_experience()
        state_for_v = np.concatenate([state,next_state[[-1],:]],axis=0)
        values = self.get_values(state_for_v)
        deltas = reward + self.gamma * values[1:] - values[:-1]
        # dcr = reward + self.gamma * values[1:]
        dcr = self.discounted_reward(reward,self.gamma,values[-1]) # discounted reward
        adv = self.discounted_reward(deltas,self.gamma * self.lam,0) # GAE as discounted reward
        for _ in range(self.a_steps):
            _,a_loss = self.sess.run([self.atrain_op, self.aloss], {self.tfs: state, self.tfa: action, self.tfadv: adv})
        for _ in range(self.c_steps):
            _,c_loss = self.sess.run([self.ctrain_op, self.closs], {self.tfs: state, self.tfdcr: dcr})
        self.sess.run(self.update_oldpi_ops)
        super(PPO,self).learn()
        self.clear_experience()

    def discounted_reward(self, reward, gamma, value0):
        dcr = np.zeros(reward.shape)
        dcr[-1] = reward[-1] + gamma * value0
        for t in reversed(range(reward.shape[0]-1)):
            dcr[t] = reward[t] + gamma * dcr[t+1]
        return dcr

    def get_action(self, state, typ):
        states = np.stack([state],axis=0)
        action = self.sess.run(self.sample_a, {self.tfs: states})
        return action[0]

    def get_values(self, states):
        return self.sess.run(self.tfv, {self.tfs: states})
