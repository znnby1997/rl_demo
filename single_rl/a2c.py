import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical

import gym
import random
import numpy as np
import tensorboard_easy as te
import datetime
import os
    

class AC(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super(AC, self).__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.fc_pi = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        self.fc_v = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def pi(self, state, dim=0):
        return f.softmax(self.fc_pi(state), dim=dim)

    def v(self, state):
        return self.fc_v(state)

# n_step
def computer_target(v_final, r_lst, done_lst, gamma):
    G = v_final # [1]
    td_target = []
    # 循环n_step次
    for r, d in zip(r_lst[::-1], done_lst[::-1]):
        G = r + gamma * G * (1-d)
        td_target.append(G)
    return td_target[::-1] # n_step, 1


class A2C():
    def __init__(self, input_shape, n_actions, hidden_dim=256, lr=2e-4, e_coef=0.02):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.e_coef = e_coef
    
        self.ac = AC(input_shape, n_actions, hidden_dim)

        self.ac_opt = optim.Adam(self.ac.parameters(), lr=lr)


    def infer_action(self, state):
        a_dis = self.ac.pi(state) # n_actions
        return Categorical(a_dis).sample().item()

    def learn(self, n_step_data):
        # n_step_data中记录了连续n步的数据
        n_s, n_a, n_td_target = n_step_data
        
        s_vec = torch.tensor(n_s).float() # n_step, state_dim
        a_vec = torch.tensor(n_a).long() # n_step, 1
        td_target_vec = torch.stack(n_td_target) # n_step, 1
        advantage = td_target_vec - self.ac.v(s_vec) # n_step, 1

        pi = self.ac.pi(s_vec, dim=1) # n_step, n_actions
        pi_a = pi.gather(1, a_vec) # n_step, 1
        actor_loss = (torch.log(pi_a + 1e-8) * advantage.detach()).mean()
        critic_loss = f.smooth_l1_loss(self.ac.v(s_vec), td_target_vec)
        entropy_loss = Categorical(pi).entropy().mean()
        loss = -actor_loss + critic_loss - self.e_coef * entropy_loss

        self.ac_opt.zero_grad()
        loss.backward()
        self.ac_opt.step()

        return actor_loss, critic_loss, entropy_loss

    def save(self, url):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        torch.save(self.ac, url + '/a2c_model_actor_' + timestamp + '.pt')


def test(step_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10
    for t in range(num_test):
        s = env.reset()
        while not done:
            a = model.infer_action(torch.from_numpy(s).float())
            next_s, r, done, _ = env.step(a)

            score += r
            s = next_s
        done = False
    print('Step # :{}, avg score : {}'.format(step_idx, score/num_test))
    env.close()
    return score/num_test

    
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    a2c = A2C(state_dim, n_actions, 64)

    update_interval = 5
    max_train_steps = 200000
    gamma = 0.98

    model_url = '../model/a2c'
    log_url = '../log/a2c_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    if not os.path.exists(log_url):
        os.makedirs(log_url)

    logger = te.Logger(log_url)

    step_idx = 0
    s = env.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, done_lst = [], [], [], []
        print('Step # :{} transitions getting starts'.format(step_idx))
        for i in range(update_interval):
            a = a2c.infer_action(torch.from_numpy(s).float())
            next_s, r, d, _ = env.step(a)

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(r/100.0)
            done_lst.append(d)

            if d:
                s = env.reset()
            else:
                s = next_s

            step_idx += 1
        
        v_final = a2c.ac.v(torch.from_numpy(next_s).float()) # [1]
        n_steps_data = (s_lst, a_lst, computer_target(v_final, r_lst, done_lst, gamma))
        print('Step # : {} training ...'.format(step_idx))
        actor_loss, critic_loss, entropy_loss = a2c.learn(n_steps_data)
        logger.log_scalar(tag='actor_loss', value=actor_loss.item(), step=step_idx)
        logger.log_scalar(tag='critic_loss', value=critic_loss.item(), step=step_idx)
        logger.log_scalar(tag='entropy_loss', value=entropy_loss.item(), step= step_idx)

        score = test(step_idx, a2c)
        logger.log_scalar(tag='avg score', value=score, step=step_idx)

    
    env.close()
    print('model saving ... ')
    a2c.save(model_url)

        



        




