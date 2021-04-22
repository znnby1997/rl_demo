import numpy as np
import random

class GridWorld():
    def __init__(self, map_size=10, trap_num=20, seed=0):
        self.map_size = map_size
        self.trap_num = trap_num
        np.random.seed(seed)

        self.trap_pos_lst = np.random.randint(1, map_size, size=(trap_num, 2))
        self.reward_lst = [[-1.0 for i in range(self.map_size)] for j in range(self.map_size)]

        self.target_pos = [self.map_size-1]*2

        self.max_step = 200

        for pos in self.trap_pos_lst:
            self.reward_lst[pos[0]][pos[1]] = -2.0

    def reset(self):
        assert self.trap_num > 0, 'there must be some traps in this map'
        assert self.map_size > 1, 'so much smaller maps'

        self.step_num = 0
        self.cur_pos = [0, 0]
        return self.cur_pos

    def step(self, action):
        # print('step {} execute ... '.format(self.step_num))
        # print(self.cur_pos, '\t', self.target_pos)
        reward = -np.linalg.norm(np.array(self.cur_pos, dtype=float) - np.array(self.target_pos, dtype=float)) / 16
        # print('reward: ', reward)
        next_pos = [0, 0]
        done = False

        if self.step_num >= self.max_step:
            done = True

        if self.cur_pos == self.target_pos:
            print('arrived on target !!!')
            done = True
            reward += 1.0
        elif not self._can_exec(action):
            reward -= 0.5
        else:
            self._exec_action(action)
            reward += self.reward_lst[self.cur_pos[0]][self.cur_pos[1]]
        
        self.step_num += 1

        return self.cur_pos, reward, done

    def _exec_action(self, action):
        if action == 0:
            # up
            self.cur_pos[0] -= 1
        elif action == 1:
            # down
            self.cur_pos[0] += 1
        elif action == 2:
            # left
            self.cur_pos[1] -= 1
        elif action == 3:
            # right
            self.cur_pos[1] += 1

    
    def _can_exec(self, action):
        return not (self.cur_pos[0] <= 0 and action == 0) and not (self.cur_pos[0] >= self.map_size-1 and action == 1) and not (self.cur_pos[1] <= 0 and action == 2) and not (self.cur_pos[1] >= self.map_size-1 and action == 3)


    def get_env_info(self):
        env_info = {}
        env_info['state_dim'] = 2 # 当前位置
        env_info['n_actions'] = 4 # 0:up, 1:down, 2:left, 3:right
        env_info['trap_pos_lst'] = self.trap_pos_lst.tolist()
        return env_info

if __name__ == '__main__':
    env = GridWorld()
    env.reset()
    done = False
    while not done:
        prime_s, reward, done = env.step(np.random.choice([0, 1, 2, 3]))
    print('done')

