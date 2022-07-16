import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def select_action_eps_greedy(network, state, epsilon, agents_amount):

    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    Q_s = network(state).detach().numpy()

    if np.random.rand(1)[0] <= epsilon:
      action = np.random.randint(0, len(Q_s))
    else:   
      action = np.argmax(Q_s)

    return int(action)

def train_model(states, actions, rewards, next_states, done, optimizer, policy_model, target_model, gamma=0.99):

  states_t = torch.tensor(states, dtype=torch.float32)  # shape: [batch_size, state_size]
  actions_t = torch.tensor(actions, dtype=torch.long)            # shape: [batch_size]
  rewards_t = torch.tensor(rewards, dtype=torch.float32)         # shape: [batch_size]
  next_states_t = torch.tensor(next_states, dtype=torch.float32) #shape: [batch_size, state_size]
  done_t = torch.tensor(done, dtype=torch.bool)               # shape: [batch_size]

  predicted_qvalues = policy_model(states_t)
  predicted_qvalues_for_actions = predicted_qvalues[range(states_t.shape[0]), actions_t]
  predicted_next_qvalues = target_model(next_states_t)

  next_state_values = torch.max(predicted_next_qvalues, 1)[0]
  target_qvalues_for_actions = gamma * next_state_values + rewards_t
  target_qvalues_for_actions = torch.where(done_t, rewards_t, target_qvalues_for_actions)

  loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)
  # добавляем регуляризацию на значения Q 
  loss += 0.1 * predicted_qvalues_for_actions.mean()

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
