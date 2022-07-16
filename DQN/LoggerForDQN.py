import random
import numpy as np
import random

class Logger():
  
  def __init__(self, size):
    self._len = 0
    self._logs = []
    self._maxlen = size
  
  def add_log(self, log):
    if (self._len + 1 > self._maxlen):
      self._logs = self._logs[int(self._len / 2):]
      self._len = len(self._logs)
    self._logs.append(log)
    self._len += 1
  
  def sample_logs(self, batch_size):
    if (batch_size > self._len):
      raise ValueError
    rand_log_inds = random.sample(range(0, self._len), batch_size)
    temp_np_logs = np.array(self._logs)
    samples = temp_np_logs[rand_log_inds]
    return np.array([i[0] for i in samples]), np.array([i[1] for i in samples]), np.array([i[2] for i in samples]), np.array([i[3] for i in samples]), np.array([i[4] for i in samples])
  
  def is_ready(self, batch_size):
    return self._len >= batch_size