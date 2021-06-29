import gym

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical

import tensorboard_easy as te
import numpy as np
import datetime
import os

class AC(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dim=128):
        super(AC, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def pi(self, s, dim=0):
        return f.softmax(self.actor(s), dim)

    def v(self, s):
        return self.critic(s)

class PPO():
    def __init__(self, input_shape, n_actions):
        self.ac = AC(input_shape, n_actions)

    def infer_action(self, s):
        s = torch.from_numpy(s).float()
        a_dis = self.ac.pi(s)
        return Categorical(a_dis).sample().item(), a_dis.detach().numpy()


def gae(v, s_tensor, r_tensor, ns_tensor, d_tensor, gamma, lamb):
    """
        v: critic
        s_tensor: [n_steps, s_dim],
        r_tensor: [n_steps, 1],
        ns_tensor: [n_steps, s_dim],
        d_tensor: [n_steps, 1]
    """
    n_steps = r_tensor.shape[0]
    deltas = r_tensor + gamma * v(ns_tensor) * (1 - d_tensor) - v(s_tensor) # n_steps, 1
    # deltas = r_tensor + gamma * ns_tensor * (1 - d_tensor) - s_tensor # n_steps, 1
    advantage = 0.
    advantages = torch.zeros((n_steps, 1), dtype=torch.float)
    for t in range(n_steps-1, -1, -1):
        advantage = deltas[t] + gamma * lamb * advantage * (1 - d_tensor[t])
        advantages[t, :] = advantage
        # print('t: {}, deltas[t]: {}, advantages: {}'.format(t, deltas[t], advantages))
    return advantages # n_steps, 1

# ns_tensor = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float)
# s_tensor = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float)
# r_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float)
# d_tensor = torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.float)
# print(gae(None, s_tensor, r_tensor, ns_tensor, d_tensor, 0.1, 0.01))


def train(mini_batch, model, optimizer, gamma, lamb, clip_eps, k_epoch, e_coef):
    mini_s, mini_a, mini_prob, mini_r, mini_ns, mini_d = mini_batch
    """
        mini_s: n_steps, state_dim
        mini_a: n_steps, 1
        mini_prob: n_steps, n_actions
        mini_r: n_steps, 1
        mini_ns: n_steps, state_dim
        mini_d: n_steps, 1
    """

    for k in range(k_epoch):
        advantages = gae(model.v, mini_s, mini_r, mini_ns, mini_d, gamma, lamb).detach() # n_steps, 1

        new_prob = model.pi(mini_s, 1) # n_steps, n_actions
        new_pi = new_prob.gather(1, mini_a) # n_steps, 1
        old_pi = mini_prob.gather(1, mini_a) # n_steps, 1

        ratios = torch.exp((new_pi + 1e-8).log() - (old_pi + 1e-8).log())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-clip_eps, 1+clip_eps) * advantages

        actor_loss = torch.min(surr1, surr2) # n_steps, 1

        entropy = Categorical(new_prob).entropy() # n_steps
        entropy_loss = e_coef * entropy

        td_targets = mini_r + gamma * model.v(mini_ns) * (1 - mini_d)
        preds = model.v(mini_s)
        
        critic_loss = f.smooth_l1_loss(preds, td_targets.detach())

        loss = -actor_loss.mean() + critic_loss.mean() - entropy_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return actor_loss.mean(), critic_loss.mean(), entropy_loss.mean()

def test(epoch_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10
    for t in range(num_test):
        s = env.reset()
        while not done:
            a, _ = model.infer_action(s)

            # if epoch_idx % 100 == 1 and t == 0:
            #     print('policy distributions: ', _)

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

    rl_alg = PPO(state_dim, n_actions)
    opt = optim.Adam(rl_alg.ac.parameters(), lr=1e-4)

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    model_url = '../model/ppo'
    log_url = '../log/ppo_' + timestamp
    if not os.path.exists(model_url):
        os.makedirs(model_url)
    if not os.path.exists(log_url):
        os.makedirs(log_url)

    logger = te.Logger(log_url)

    total_eps = 50000
    gamma = 0.98
    clip_eps = 0.1
    lamb = 0.95
    k_epoch = 3
    e_coef = 0.000
    update_interval = 10

    for eps in range(total_eps):
        s_lst, a_lst, prob_lst, ns_lst, r_lst, d_lst = [], [], [], [], [], []
        for i in range(update_interval):
            done = False
            s = env.reset()

            while not done:
                a, prob = rl_alg.infer_action(s)
                next_s, r, done, _ = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r/100.0])
                d_lst.append([done])
                ns_lst.append(next_s)
                prob_lst.append(prob)

                s = next_s

        mini_batch = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst, dtype=torch.long),
            torch.tensor(prob_lst, dtype=torch.float),
            torch.tensor(r_lst, dtype=torch.float),
            torch.tensor(ns_lst, dtype=torch.float),
            torch.tensor(d_lst, dtype=torch.float))
            
        
        p_loss, v_loss, e_loss = train(mini_batch, rl_alg.ac, opt, gamma, lamb, clip_eps, k_epoch, e_coef)
        logger.log_scalar(tag='pi_loss', value=p_loss.item(), step=eps)
        logger.log_scalar(tag='v_loss', value=v_loss.item(), step=eps)
        logger.log_scalar(tag='e_loss', value=e_loss.item(), step= eps)
        avg_score = test(eps, rl_alg)
        logger.log_scalar(tag='avg score', value=avg_score, step=eps)
        if eps % 100 == 1:
            print('episode {} starts ... getting exp ...'.format(eps))
            print('current score: ', avg_score)
            print('Step # : {} training ...'.format(eps))

    env.close()
    print('model saving ... ')
    torch.save(rl_alg, model_url + '/ppo_model_' + timestamp + '.pt')

        

        





