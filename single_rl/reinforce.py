from os import stat
import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Categorical
import torch.optim as optim

import numpy as np
import tensorboard_easy as te

import datetime
import os

class Policy(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=128):
        super(Policy, self).__init__()

        self.p = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x, dim=0):
        return f.softmax(self.p(x), dim)

def computer_r(r_lst, gamma):
    """
        r_lst为长度为steps的list，包含了一个episode中所有step的reward
        mc: G_t = r_t + r_{t+1} + ... + r_{T}
    """
    steps = len(r_lst)
    ret = [0] * steps
    ret[-1] = r_lst[-1]
    for i in range(steps-2, -1, -1):
        ret[i] = r_lst[i] + gamma * ret[i+1]
    return ret

def computer_r_(r_lst, d_lst, gamma):
    """
        该方法能够保证收集多个episode下的每个(state, action)对应的MC reward
    """
    steps = len(r_lst)
    ret = [0] * steps
    ret[-1] = r_lst[-1]
    for i in range(steps-2, -1, -1):
        ret[i] = r_lst[i] + gamma * ret[i+1] * (1-d_lst[i])
    return ret

class REINFORCE():
    def __init__(self, input_shape, n_actions, ):
        self.pi = Policy(input_shape, n_actions)

    def infer_action(self, obs):
        obs = torch.from_numpy(obs).float()
        a_prob = self.pi(obs)
        return Categorical(a_prob).sample().item()

def train(eps_trajs, policy, opt, e_coef):
    s_lst, a_lst, r_lst = eps_trajs # r shape: n_steps,
    
    a_probs = policy(s_lst, 1) # steps, n_actions
    log_as = torch.log(a_probs.gather(1, a_lst)) # steps, 1
    policy_loss = log_as * r_lst.unsqueeze(1) # steps, 1
    entropy_loss = Categorical(a_probs).entropy() # steps

    # loss = -policy_loss.mean() - e_coef * entropy_loss.mean()
    loss = -policy_loss.mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    return policy_loss.mean(), entropy_loss.mean()

def test(epoch_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10
    for t in range(num_test):
        s = env.reset()
        while not done:
            a = model.infer_action(s)
            next_s, r, done, _ = env.step(a)

            score += r
            s = next_s
        done = False
    # print('Step # :{}, avg score : {}'.format(epoch_idx, score/num_test))
    env.close()
    return score/num_test



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    rl_alg = REINFORCE(state_dim, n_actions)
    opt = optim.Adam(rl_alg.pi.parameters(), lr=1e-4)

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    model_url = '../model/reinforce'
    log_url = '../log/reinforce_' + timestamp
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    if not os.path.exists(log_url):
        os.makedirs(log_url)

    logger = te.Logger(log_url)

    total_eps = 100000
    gamma = 0.98
    e_coef = 0.002
    update_interval = 10

    for eps in range(total_eps):
        s_lst, a_lst, r_lst, d_lst = [], [], [], []
        for i in range(update_interval):
            done = False
            s = env.reset()

            while not done:
                a = rl_alg.infer_action(s)
                next_s, r, done, _ = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r/100.0)
                d_lst.append(done)

                s = next_s

        new_r_lst = computer_r_(r_lst, d_lst, gamma)
        traj = (torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.long), torch.tensor(new_r_lst, dtype=torch.float))
        if eps % 1000 == 1:
            print('old r_lst: ', r_lst)
            print('new r_lst: ', new_r_lst)
            print('done_lst: ', d_lst)
            
        
        p_loss, en_loss = train(traj, rl_alg.pi, opt, e_coef)
        logger.log_scalar(tag='pi_loss', value=p_loss.item(), step=eps)
        logger.log_scalar(tag='en_loss', value=en_loss.item(), step=eps)
        avg_score = test(eps, rl_alg)
        logger.log_scalar(tag='avg score', value=avg_score, step=eps)
        if eps % 100 == 1:
            print('episode {} starts ... getting exp ...'.format(eps))
            print('current score: ', avg_score)
            print('Step # : {} training ...'.format(eps))

    env.close()
    print('model saving ... ')
    torch.save(rl_alg, model_url + '/reinforce_model_' + timestamp + '.pt')

        

        



