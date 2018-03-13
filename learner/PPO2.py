import numpy as np
import tensorflow as tf
from AbstractLearner import AbstractLearner
from PPO import PPO

class PPO2(PPO):
    '''
    Proximal poliy optimization learner with value net and actor net
    sharing parameters via preprocess net.
    '''

    def __init__(self,**kwargs):
        #########################################
        ## read inputs
        AbstractLearner.__init__(self,**kwargs)
        s_dim = kwargs.get('S_DIM',1) # dimension size of state
        c_lr = kwargs.get('C_LR',0.0001) # learning rate for critic
        a_lr = kwargs.get('A_LR',0.0001) # learning rate for actor
        self.c_steps = kwargs.get('C_UPDATE_STEPS',10) # update steps for critic
        self.a_steps = kwargs.get('A_UPDATE_STEPS',10) # update steps for actor
        epsilon = kwargs.get('epsilon',0.1) # trust region size
        self.gamma = kwargs.get('gamma',0.95) # discount factor
        self.lam = kwargs.get('lam',0.9) # reward shaping factor

        p_net = lambda s:tf.layers.dense(s,9)
        v_net = lambda s:tf.layers.dense(s,1)[:,0]
        def a_net(s):
            mu = tf.layers.dense(s,1)[:,0]
            sig = tf.nn.softplus(tf.dense(s,1))[:,0]
            pi = tf.distributions.Normal(mu,sig)
            return pi

        self.value_net = kwargs.get('v_net',v_net) # value net
        self.act_net = kwargs.get('a_net',a_net) # act net
        self.process_net = kwargs.get('p_net',p_net) # preprocess net

        #########################################
        ## construct computation graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            self.tfs = tf.placeholder(tf.float32, [None, s_dim], 'state')
            self.tfa = tf.placeholder(tf.float32, [None], 'action')
            self.tfdcr = tf.placeholder(tf.float32, [None], 'discounted_reward')
            self.tfadv = tf.placeholder(tf.float32, [None], 'advantage')

            with tf.variable_scope('pro'):
                self.tfsp = self.process_net(self.tfs)
            with tf.variable_scope('old_pro'):
                self.old_tfsp = self.process_net(self.tfs)
            pro_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pro')
            oldpro_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pro')
            # critic: l = d(dcr,v)
            with tf.variable_scope('critic'):
                self.tfv = self.value_net(self.tfsp)
                self.closs = tf.reduce_mean(tf.square(self.tfdcr - self.tfv))
                self.ctrain_op = tf.train.AdamOptimizer(c_lr).minimize(self.closs)

            # actor: l = f(s,adv)
            with tf.variable_scope('actor'):
                with tf.variable_scope('pi'):
                    pi = self.act_net(self.tfsp)
                    self.probs = pi.probs
                with tf.variable_scope('old_pi'):
                    oldpi = self.act_net(self.old_tfsp)
                pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/pi')
                oldpi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/old_pi')
                self.update_oldpi_ops = [oldp.assign(p) for p, oldp in zip(pi_params + pro_params,
                                                                           oldpi_params + oldpro_params)]

                self.sample_a = pi.sample()
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                self.aloss = -tf.reduce_mean(tf.minimum(
                    ratio * self.tfadv,
                    tf.clip_by_value(ratio, 1.-epsilon, 1.+epsilon)*self.tfadv))
                self.atrain_op = tf.train.AdamOptimizer(a_lr).minimize(self.aloss,var_list=(pi_params + pro_params))

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.vars = pi_params + pro_params
