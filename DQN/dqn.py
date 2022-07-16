import gym
import pogema
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from pogema.animation import AnimationMonitor
from IPython.display import SVG, display

from WrappersForDQN import OneAgentWrapper
from LoggerForDQN import Logger
import TrainLibForDQN as TL
from ModelForDQN import QModel

def generate_session(env, opt, logger, batch_size, policy_model, target_model, target_update, agents_amount, i, t_max=1000, epsilon=0, train=False):
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = TL.select_action_eps_greedy(policy_model, s, agents_amount, epsilon)
        next_s, r, done, _ = env.step(a)
       
        if train:
            logger.add_log([s, a, r, next_s, done])
            TL.train_model(np.array([s]), np.array([a]), np.array([r]), np.array([next_s]), np.array([done]), opt, policy_model, target_model, gamma=0.99)
            
            if logger.is_ready(batch_size):
              states, actions, rewards, next_ss, dones = logger.sample_logs(batch_size)
              TL.train_model(states, actions, rewards, next_ss, dones, opt, policy_model, target_model, gamma=0.99)

            if t % target_update == 0:
              target_model.load_state_dict(policy_model.state_dict())
        total_reward += r
        s = next_s
        if done:
            break
    if i % 10 == 0:
        print(f"{i}/100")
    return total_reward

env_orig = gym.make('Pogema-8x8-normal-v0').unwrapped
agents_amount = 2

env = OneAgentWrapper(gym.make('Pogema-8x8-normal-v0').unwrapped, agents_amount)

n_actions = env_orig.action_space.n
state_dim = env_orig.observation_space.shape

HIDDEN = 256
policy_model = QModel(state_dim[0] * state_dim[1] * state_dim[2], n_actions, HIDDEN)
target_model = QModel(state_dim[0] * state_dim[1] * state_dim[2], n_actions, HIDDEN)
target_model.load_state_dict(policy_model.state_dict())
target_model.eval()

BATCH_SIZE = 512
GAMMA = 0.99
TARGET_UPDATE = 10
T_MAX = 1000 #5000
EPSILON = 0.5
EPSILON_DECAY = 0.99
LOGGER_SIZE = 2048

logger = Logger(LOGGER_SIZE)
opt = torch.optim.Adam(policy_model.parameters(), lr=1e-4)

for i in range(150):
    session_rewards = [generate_session(env, opt, logger, BATCH_SIZE, policy_model, target_model, TARGET_UPDATE, agents_amount, i, t_max = T_MAX, epsilon=EPSILON, train=True) for i in range(100)]
    print("Epoch: #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), EPSILON))

    EPSILON *= EPSILON_DECAY
    if EPSILON <= 1e-4:
      EPSILON = 0.5 

    if np.mean(session_rewards) >= 0.970 or i > 120:
        print("Принято!")
        break

torch.save(policy_model.state_dict(), "/home/huawei/NonDisPogema/DQN/model.pth")

env = gym.make("Pogema-8x8-normal-v0")
env = AnimationMonitor(env)
env = OneAgentWrapper(env, agents_amount)

generate_session(env, opt, logger, BATCH_SIZE, policy_model, target_model, TARGET_UPDATE, agents_amount, 1, train=False)

env.save_animation("render.svg", egocentric_idx=None)
display(SVG('render.svg'))
env.close()
