import os
import numpy as np
import sys
import codecs

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
import random
import json
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from parl.utils import ReplayMemory # 经验回放
from parl.utils import action_mapping # 将神经网络输出映射到对应的 实际动作取值范围 内
######################################################################
######################################################################
#
# 1. 请设定 learning rate，尝试增减查看效果
#
######################################################################
######################################################################
ACTOR_LR = 0.0001   # Actor网络更新的 learning rate
CRITIC_LR = 0.0005   # Critic网络更新的 learning rate

GAMMA = 0.99        # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001         # target_model 跟 model 同步参数 的 软更新参数
MEMORY_SIZE = 1e4   # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e6      # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.001       # reward 的缩放因子
BATCH_SIZE = 256        # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6   # 总训练步数
TEST_EVERY_STEPS = 1e4    # 每个N步评估一下算法效果，每次评估5个episode求平均reward

class ActorModel(parl.Model):
    def __init__(self, act_dim):
        ######################################################################
        ######################################################################
        #
        # 2. 请配置model结构
        hid1_size = 128
        hid2_size = 128
        
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')
        #
        ######################################################################
        ######################################################################

    def policy(self, obs):
        ######################################################################
        ######################################################################
        #
        # 3. 请组装policy网络
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        #
        ######################################################################
        ######################################################################
        return means

class CriticModel(parl.Model):
    def __init__(self):
        ######################################################################
        ######################################################################
        #
        # 4. 请配置model结构
        hid_size = 128

        self.fc1 = layers.fc(size=hid_size, act='relu')
        self.fc2 = layers.fc(size=1, act=None)
        #
        ######################################################################
        ######################################################################

    def value(self, obs, act):
        # 输入 state, action, 输出对应的Q(s,a)

        ######################################################################
        ######################################################################
        #
        # 5. 请组装Q网络
        concat = layers.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = layers.squeeze(Q, axes=[1])
        #
        ######################################################################
        ######################################################################
        return Q
class QuadrotorModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()


from parl.algorithms import DDPG

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim=4):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        # 注意，在最开始的时候，先完全同步target_model和model的参数
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

env = gym.make('gym_foo:foo-v0')

# replay_memory.py
import random
import collections
import numpy as np





act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0] 
# 创建经验池
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)



# 根据parl框架构建agent
######################################################################
######################################################################
#
# 4. 请参考课堂Demo，嵌套Model, DQN, Agent构建 agent
#
######################################################################
######################################################################
model = QuadrotorModel(act_dim=act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = Agent(algorithm, obs_dim=obs_dim, act_dim=act_dim)
total_reward, steps = 0, 0
EPISODES = 500
TAU = 0.01
NUM_OF_SELLER=5
N_ACTIONS = NUM_OF_SELLER*2
NUM_OF_BUYER=9
result = np.zeros((EPISODES, N_ACTIONS + 1 + 1))
for i in range(EPISODES):
    obs = env.reset()
    np.random.seed(0)
    score_history = []
    print(i)
    print(" episode start with" + str(["%.2f" % val for val in obs[0:NUM_OF_SELLER * 2]]))
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # 给输出动作增加探索扰动，输出限制在 [-1.0, 1.0] 范围内
        action = np.random.normal(action, 1.0)
        action = np.clip(action, -1.0, 1.0)
        action = action_mapping(action,env.action_space.low[0],env.action_space.high[0])
        # 动作映射到对应的 实际动作取值范围 内, action_mapping是从parl.utils那里import进来的函数

        next_obs, reward, done, info = env.step(action)

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
            batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)
        obs = next_obs
        total_reward += reward
        if steps > 300:
            result[i] = np.concatenate((next_obs[0:NUM_OF_SELLER * 2], [reward, np.mean(score_history[-50:])]))
            print("episode end with  " + str(["%.2f" % val for val in next_obs[0:NUM_OF_SELLER * 2]]))
            break

