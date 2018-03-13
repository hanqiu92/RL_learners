import numpy as np

class AbstractLearner(object):
    '''
    Abstract class for general (numerical) policy learner. Provides three types of interface:
    1. functions for interaction with environment: get_action & learn
    2. functions for internal memory management: add_experience, clear_experience, ...
    3. functions for saving and retrieving model: save & restore
    '''

    def __init__(self,**kwargs):
        self.exp_pool = {0:[]} # dict of list of (s,a,r,s')
        self.exp_pool_size = {0:0}
        self.max_exp_pool_size = int(kwargs.get('max_exp_pool_size',1e8)) # max size of each exp pool
        self.exp_sample_size = kwargs.get('exp_sample_size',256)
        self.callback_func = kwargs.get('callback',None) # callback function for debugging in the learning process
        self.saver = None # TF saver for the model

    def add_experience(self, new_exp, exp_pool_id=0):
        if exp_pool_id not in self.exp_pool:
            self.exp_pool[exp_pool_id] = []
            self.exp_pool_size[exp_pool_id] = 0

        self.exp_pool[exp_pool_id].append(new_exp)
        self.exp_pool_size[exp_pool_id] += 1
        # maintain exp pool size
        if self.exp_pool_size[exp_pool_id] > self.max_exp_pool_size:
            self.exp_pool[exp_pool_id].pop(0)
            self.exp_pool_size[exp_pool_id] = self.max_exp_pool_size

    def get_experience(self, exp_pool_id=0):
        '''
        get all experience in a certain experience pool
        '''
        exp = self.exp_pool[exp_pool_id]
        state,action,reward,next_state = tuple([np.stack(item,axis=0) for item in zip(*exp)])
        return state,action,reward,next_state

    def get_experience_sample(self, exp_pool_id=0):
        '''
        obtain experience sample batch from a certain experience pool
        '''
        exp = self.exp_pool[exp_pool_id]
        if len(exp) >= max(self.exp_sample_size * 8,1024):
            sample_idx = np.random.choice(len(exp),size=(self.exp_sample_size,),replace=False)
            sample_exp = [exp[i] for i in sample_idx]
            state,action,reward,next_state = tuple([np.stack(item,axis=0) for item in zip(*sample_exp)])
            return state,action,reward,next_state
        else:
            return None,None,None,None

    def clear_experience(self):
        self.exp_pool = {0:[]}
        self.exp_pool_size = {0:0}

    def get_action(self, state):
        pass

    def learn(self):
        if self.callback_func is not None:
            self.callback_func(self)

    def save(self, fname=''):
        if self.saver is not None:
            self.saver.save(self.sess,fname)

    def restore(self, fname=''):
        if self.saver is not None:
            self.saver.restore(self.sess,fname)
