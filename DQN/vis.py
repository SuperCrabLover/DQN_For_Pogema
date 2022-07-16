import gym
import pogema
import numpy as np
from numpy.random.mtrand import rand
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from heapq import heappop, heappush
from pogema.animation import AnimationMonitor
from IPython.display import SVG, display

import TrainLibForDQN as TL
from ModelForDQN import QModel


def generate_session(env, policy_model, agents_amount, t_max=1000, epsilon=0):
    total_reward = np.zeros(agents_amount)
    s = env.reset()
    for t in range(t_max):
        a_arr = []
        for i in range(agents_amount):
            a_arr.append(TL.select_action_eps_greedy(policy_model, s[i].flatten(), agents_amount, epsilon))
        next_s, r, done, _ = env.step(a_arr)
       
        total_reward += r
        s = next_s
        if all(done):
            break

    return total_reward

env_orig = gym.make('Pogema-8x8-normal-v0').unwrapped
agents_amount = 2

n_actions = env_orig.action_space.n
state_dim = env_orig.observation_space.shape

HIDDEN = 128
policy_model = QModel(state_dim[0] * state_dim[1] * state_dim[2], n_actions, HIDDEN)
policy_model.load_state_dict(torch.load("/home/huawei/NonDisPogema/DQN/model.pth"))
policy_model.eval()


env = gym.make("Pogema-8x8-normal-v0")
env = AnimationMonitor(env)

generate_session(env, policy_model, agents_amount, t_max=1000, epsilon=0)

env.save_animation("render.svg", egocentric_idx=None)
display(SVG('render.svg'))
env.close()