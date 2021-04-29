import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Categorical
import numpy as np
import random
import tensorboard_easy as te
import datetime
import os

import sys
sys.path.append('..')
import toy_env.grid_world as gw


'''
    与dqn基本一致，唯一不同的在于buffer中data的设计
    对于每个transition data需要对应一个td-error，作为priority
'''

class PrioritiedBuffer():
    def __init__(self, limit_len, beta=1):
        self.limit_len = limit_len
        self.buffer = []
        self.priority_lst = []
        self.beta = beta

    def put(self, transition):
        if len(self.buffer) >= self.limit_len:
            self.buffer.pop(0)
            self.priority_lst.pop(0)
        self.buffer.append(transition[:-1])
        self.priority_lst.append(transition[-1])

    def sample(self, batch_size):
        pri_dis = f.softmax(torch.tensor(self.priority_lst, dtype=torch.float), dim=0).numpy()
        # pri_dis = np.array(self.priority_lst) / np.sum(self.priority_lst)
        is_weights = np.exp(-self.beta * np.log(self.limit_len * pri_dis))
        sample_idx = np.random.choice(np.arange(0, len(self.priority_lst)), size=(batch_size, ), replace=False, p=pri_dis) # batch_size

        s_lst, a_lst, r_lst, next_s_lst, d_lst, is_w_lst = [], [], [], [], [], []
        for idx in sample_idx:
            s, a, r, next_s, d = self.buffer[idx]
            is_w = is_weights[idx]

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            next_s_lst.append(next_s)
            d_lst.append([d])
            is_w_lst.append([is_w])

        '''
            state shape: batch_size, state_dim
            action shape: batch_size, 1
            reward shape: batch_size, 1
            next_state shape: batch_size, state_dim
            done shape: batch_size, 1
            is_w shape: batch_size, 1
        '''
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst, dtype=torch.float), \
                torch.tensor(next_s_lst, dtype=torch.float), torch.tensor(d_lst, dtype=torch.float), torch.tensor(is_w_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super(QNet, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.q_network = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, state):
        return self.q_network(state)


class DQN():
    def __init__(self, state_dim, n_actions, hidden_dim=64, gamma=0.98, lr=1e-4):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr

        self.cur_q = QNet(state_dim, n_actions, hidden_dim)
        self.tar_q = QNet(state_dim, n_actions, hidden_dim)

        self.cur_opt = optim.Adam(self.cur_q.parameters(), lr)
        self.tar_q.load_state_dict(self.cur_q.state_dict())

    def infer_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            return np.random.choice(np.arange(0, n_actions, dtype=int))
        else:
            return self.cur_q(state).max(0)[1].item()

    def get_td_error(self, transition):
        s, a, r, next_s, d = transition
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        next_s = torch.tensor(next_s, dtype=torch.float)
        d = torch.tensor(d, dtype=torch.float)

        qa_value = self.cur_q(s).gather(0, a)
        target_q = r + self.gamma * self.tar_q(next_s).max(0)[0] * (1 - d)
        # print(qa_value, ' jjjj ', target_q)
        return abs(target_q.item() - qa_value.item())

    
    def learn(self, minibatch):
        batch_s, batch_a, batch_r, batch_next_s, batch_d, batch_w = minibatch

        q_values = self.cur_q(batch_s) # shape: batch, n_actions
        qa_values = q_values.gather(1, batch_a) * batch_w # shape: batch, 1

        target_q = (batch_r + self.gamma * self.tar_q(batch_next_s).max(1, keepdim=True)[0] * (1 - batch_d)) * batch_w # shape: batch, 1

        # print(target_q.shape, ' ', qa_values.shape)

        one_step_loss = loss = f.smooth_l1_loss(qa_values, target_q.detach())

        self.cur_opt.zero_grad()
        loss.backward()
        self.cur_opt.step()
        return one_step_loss

    def save(self, url):
        torch.save(self.tar_q, url + '/dqn_model_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt')

    def update_target(self):
        self.tar_q.load_state_dict(self.cur_q.state_dict())


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    dqn = DQN(state_dim, n_actions)
    buffer = PrioritiedBuffer(50000)

    model_url = '../model/dqn'
    log_url = '../log/dqn_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    if not os.path.exists(log_url):
        os.makedirs(log_url)

    logger = te.Logger(log_url)

    max_epsilon=0.8
    epsilon_delta=0.01
    epsilon_low_bound=0.01

    count = 0

    for eps in range(100000):
        # print('episode {} start ... '.format(eps))

        epsilon = max(epsilon_low_bound, max_epsilon - epsilon_delta * (eps / 200))
        # epsilon = 0.8 if eps < 5000 else max(epsilon_low_bound, epsilon - epsilon_delta / 1e-3 * (eps - 3000))
        logger.log_scalar(tag='epsilon', value=epsilon, step=eps)

        total_reward = 0

        state = env.reset()
        done = False
        while not done:
            action = dqn.infer_action(torch.from_numpy(state).float(), epsilon)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            td_error = dqn.get_td_error((state, action, reward/100.0, next_state, done))
            buffer.put((state, action, reward/100.0, next_state, done, td_error)) # 实验证明，reward缩减很重要

            state = next_state

        logger.log_scalar(tag='total_reward', value=total_reward, step=eps)
        logger.log_scalar(tag='max_td_error', value=max(buffer.priority_lst), step=eps)

        if eps % 1000 == 0:
            print('episode {}: total reward={}'.format(eps, total_reward))
        
        if buffer.size() > 2000:
            if eps % 5000 == 0:
                print('episode {} dqn training ... '.format(eps))
            loss = dqn.learn(buffer.sample(64))
            logger.log_scalar(tag='loss', value=loss.item(), step=count)
            count += 1
        
        if eps > 0 and eps % 20 == 0:
            # print('episode {} target network updating ... '.format(eps))
            dqn.update_target()

    env.close()
    print('model saving ...')
    dqn.save(model_url)

    


    





