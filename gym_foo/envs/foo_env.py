import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from os import  path
from  calPrice import  rank

class FooEnv(gym.Env):


    def __init__(self,max_buyer_price=300,max_seller_price=400,min_seller_price=300,min_buyer_price=450,\
                 num_of_seller =3,num_of_buyer = 3):
        self.max_buyer_price = max_buyer_price
        self.min_buyer_price = min_buyer_price
        self.num_of_seller = num_of_seller
        self.num_of_buyer = num_of_buyer
        self.max_seller_price= max_seller_price
        self.min_seller_price = min_seller_price

        self.cost = 300
        self.deltaprice = 0.1
        self.delta_volume = 10
        self.max_seller_voloum = 90000
        self.min_seller_voloum = 20000
        self.seller_volume = np.random.uniform(50000, 80000, [1, num_of_buyer])[0]
        self.buyer_voloum = np.random.uniform(30000, 50000, [1, num_of_buyer])[0]  # 随机设置了买方容量以用测试

        #action space 是每一次价格变量的区间
        #todo: 底下这两行记得改成general的

        low = np.array([-self.deltaprice,-self.delta_volume,-self.deltaprice,-self.delta_volume,-self.deltaprice,-self.delta_volume])
        high = np.array([self.deltaprice,self.delta_volume,self.deltaprice,self.delta_volume,self.deltaprice,self.delta_volume])
        # low = np.array([-self.deltaprice,-self.deltaprice,-self.deltaprice])
        # high = np.array([self.deltaprice,self.deltaprice,self.deltaprice])

        self.action_space = spaces.Box(low=low,high=high,shape=(num_of_seller*2,))
        # self.action_space = spaces.Box(low=low,high=high,shape=(num_of_seller,))

        #todo: observation_space是不是可以去

        high_obs = np.array([self.max_seller_price,self.max_seller_voloum,self.max_seller_price,self.max_seller_voloum,self.max_seller_price,self.max_seller_voloum])
        # high_obs = np.array([self.max_seller_price,self.max_seller_price,self.max_seller_price])

        self.observation_space = spaces.Box(low=np.array([0 for i in range(num_of_seller*2)]),high=high_obs,shape=(num_of_seller*2,))
        # self.observation_space = spaces.Box(low=np.array([0 for i in range(num_of_seller)]),high=high_obs,shape=(num_of_seller,))

        # print("__init__ called, seller volume %s is set up , buyer volume %s is set up "%(str(self.state),str(self.buyer_voloum)))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #todo:给价格和volume加一个上限

        buyer_price = self._get_buyer_random_price() #找到当前状态的买方随机价格
        buyer_random_name = ["b_a","b_b","b_c"] #随机取的名字
        seller_random_name = ["s_a","s_b","s_c"] #随机取的名字

        self.state = [self.state[i]+action[i] for i in range(self.num_of_seller*2)] #update self state with action
        # self.state = [self.state[i]+action[i] for i in range(self.num_of_seller)] #update self state with action

        buyer_num = [[self.buyer_voloum[i],buyer_price[i]] for i in  range(self.num_of_buyer)]
        seller_num =  [[self.seller_volume[i],self.state[i]] for i in  range(self.num_of_buyer)]




        #todo:这一步 每一步的buyer_volume还没有设置随机,只是在env最开始setpup的时候取的随机。

        buyer_num = []#构建买方的价格和容量
        for i in range(self.num_of_buyer):
            buyer_num.append([buyer_price[i],self.buyer_voloum[i]])
        seller_num =[]
        for i in range(0,self.num_of_seller*2,2):
            seller_num.append([self.state[i],self.state[i+1]])

        match_result ,clear_price =rank(buyer_random_name,buyer_num,seller_random_name,seller_num)

        reward = 0
        for i in range(len(match_result)):
            reward+= match_result[i][2]*(clear_price- self.cost)
        volume  = [self.state[i] for i in range(1,self.num_of_seller*2,2)]
        max =  sum(volume) * self.max_seller_price
        return np.array(self.state), reward-max, False, {}

    def reset(self):
        prices = self.np_random.uniform(self.min_seller_price, self.max_seller_price,self.num_of_seller)
        # self.state= prices
        # return self.state

        voloum =  self.np_random.uniform(self.min_seller_voloum, self.max_seller_voloum,self.num_of_seller)
        states = []
        for i in range(self.num_of_seller):
            states.append(prices[i])
            states.append(voloum[i])
        self.state = np.array(states)
        return np.array(self.state)


    def render(self, mode='human'):
        pass


    def _get_buyer_random_price(self):
        return self.np_random.uniform(self.min_buyer_price, self.max_seller_price,self.num_of_buyer)



