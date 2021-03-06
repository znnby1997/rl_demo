import numpy as np
import pandas as pd
import tensorboard_easy as te
import os
import sys
import datetime

sys.path.append('..')
import toy_env.grid_world as gw

class Buffer:
    def __init__(self, limit_len=1000):
        self.buffer = []
        self.limit_len = limit_len

    def put(self, transition):
        if len(self.buffer) >= self.limit_len:
            self.buffer.pop(0)

        self.buffer.append(transition)
        
    def sample(self):
        if not self.buffer:
            return
        return self.buffer[0]


class QL:
    def __init__(self, n_actions, ini_q_table, lr=0.001, gamma=0.99):
        self.n_actions = n_actions
        self.q_table = q_table
        self.lr = lr
        self.gamma = gamma
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def infer_action(self, state, epsilon):
        if np.random.uniform() <= epsilon:
            action = np.random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(self.q_table[state[0]][state[1]])

        return action
    
    def learn(self, transition):
        state, action, reward, next_state, done = transition
        td_error = reward + self.gamma * np.max(self.q_table[next_state[0]][next_state[1]]) * (1 - done) - self.q_table[state[0]][state[1]][action]
        self.q_table[state[0]][state[1]][action] += (self.lr * td_error)

    def save_model(self, url):
        np.save(url + '/ql_model_' + self.timestamp + '.npy', self.q_table)


if __name__ == '__main__':
    env = gw.GridWorld()
    env_info = env.get_env_info()
    map_size = env.map_size
    n_actions = env_info['n_actions']

    model_url = '../model/ql'
    log_url = '../log/ql_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    if not os.path.exists(log_url):
        os.makedirs(log_url)

    logger = te.Logger(log_url)

    q_table = np.zeros((map_size, map_size, n_actions), dtype=float)

    ql = QL(env_info['n_actions'], q_table)
    buff = Buffer()

    epsilon=0.8
    epsilon_delta=0.01
    epsilon_low_bound=0.01

    for eps in range(200000):
        epsilon = 0.8 if eps < 5000 else max(epsilon_low_bound, epsilon - epsilon_delta / 1e-3 * (eps - 3000))
        # epsilon = max(epsilon_low_bound, epsilon - epsilon_delta)
        print('episode {} start ...'.format(eps))
        done = False
        total_reward = 0
        state = env.reset()

        while not done:
            action = ql.infer_action(state, epsilon)
            next_state, reward, done = env.step(action)

            # print('action: {}, reward: {}'.format(action, reward))
            
            total_reward += reward
            buff.put((state, action, reward, next_state, done))

            ql.learn(buff.sample())
        
            state = next_state
        
        if eps % 1000 == 0:
            print('episode: {} \ttotal_step: {} \ttotal_reward: {} \tdone: {}'.format(eps, env.step_num, total_reward, done))

        logger.log_scalar(tag='total_reward', value=total_reward, step=eps)
    
    print('training is over. model saving ...')
    ql.save_model(model_url)
    print('model saved')








        
        

