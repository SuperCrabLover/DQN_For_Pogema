import numpy as np
import gym

class OneAgentWrapper(gym.Wrapper):
    def __init__(self, env, n_agents):
        super().__init__(env)
        self.env = env
        self.n_agents = n_agents
    def step(self, action):
        '''
        action is one dimentional so we append zeros to int
        '''
        actions = []
        for i in range(self.n_agents):
            actions.append(0)
        actions[0] = int(action)
        next_state, reward, done, info= self.env.step(actions)
        n_s_t =  np.array(next_state[0]).flatten()
        return n_s_t, reward[0], done[0], info[0]
    def reset(self):
        s_t = np.array(self.env.reset()[0]).flatten()
        return s_t