# From https://github.com/openai/baselines/blob/b5be53dc928bc19c39bce2a3f8a4e7dd0374f1dd/baselines/common/running_mean_std.py

import numpy as np

class RunningMean(object):
    def __init__(self, _sum=0, _count=0):
        self.sum = _sum
        self.count = _count
        self.mean = 0.0 if _count == 0 else float(_sum) / _count

    def update(self,x):
        self.sum += x
        self.count += 1
        self.mean = float(self.sum) / self.count

    def fork(self,list_of_x):
        return [RunningMean(self.sum+x, self.count+1) for x in list_of_x]

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count        
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / (count + batch_count)
    new_var = M2 / (count + batch_count)
    new_count = batch_count + count
    
    return new_mean, new_var, new_count

