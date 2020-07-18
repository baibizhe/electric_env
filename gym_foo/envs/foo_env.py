import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from os import  path
from  calPrice import  rank

class FooEnv(gym.Env):
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 30
    # }

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
        self.delta_volume = 1
        self.max_seller_voloum = 90000
        self.min_seller_voloum = 20000
        # self.seller_volum = np.random.uniform(30000, 80000, [1, num_of_seller])[0]  # 随机设置了卖方容量已用测试
        self.buyer_voloum = np.random.uniform(30000, 80000, [1, num_of_buyer])[0]  # 随机设置了买方容量已用测试

        #action space 是每一次价格变量的区间
        #todo: 底下这两行记得改成general的
        low = np.array([-self.deltaprice,-self.delta_volume,-self.deltaprice,-self.delta_volume,-self.deltaprice,-self.delta_volume])
        high = np.array([self.deltaprice,self.delta_volume,self.deltaprice,self.delta_volume,self.deltaprice,self.delta_volume])
        self.action_space = spaces.Box(low=low,high=high,shape=(num_of_seller*2,))
        #todo: observation_space是不是可以去掉
        high_obs = np.array([self.max_seller_price,self.max_seller_voloum,self.max_seller_price,self.max_seller_voloum,self.max_seller_price,self.max_seller_voloum])
        self.observation_space = spaces.Box(low=np.array([0 for i in range(num_of_seller*2)]),high=high_obs,shape=(num_of_seller*2,))
        # print("__init__ called, seller volume %s is set up , buyer volume %s is set up "%(str(self.state),str(self.buyer_voloum)))

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        prices = self.state
        #todo:给价格和volume加一个上限

        buyer_price = self._get_buyer_random_price() #找到当前状态的买方随机价格
        random_name = ["a","b","c"] #随机取的名字
        self.state = [self.state[i]+action[i] for i in range(self.num_of_seller*2)] #update self state with action
        #todo:这一步 每一步的buyer_volume还没有设置随机,只是在env最开始setpup的时候取的随机。
        buyer_num = [[buyer_price[i],self.buyer_voloum[i]] for i in range(self.num_of_buyer)] #构建买方的价格和容量
        seller_num = [[self.state[i],self.state[i+1]] for i in range(0,self.num_of_seller,2)] #构建卖方的价格和容量
        match_result ,clear_price =rank(random_name,buyer_num,random_name,seller_num)
        # print(match_result)
        reward = 0
        # print(match_result)
        for i in range(len(match_result)):
            reward+= match_result[i][2]*(clear_price- self.cost)
        # print(match_result)

        # reward = reward/sum([i[2] for i in match_result])


        # self.state = np.array([newth, newthdot])
        return np.array(self.state), reward, False, {}

    def reset(self):
        prices = self.np_random.uniform(self.min_seller_price, self.max_seller_price,self.num_of_seller)
        voloum =  self.np_random.uniform(self.min_seller_voloum, self.max_seller_voloum,self.num_of_seller)
        states = []
        for i in range(self.num_of_seller):
            states.append(prices[i])
            states.append(voloum[i])
        self.state = states
        # print("__init__ called, seller price %s is set up  "%(str(self.state)))

        return np.array(self.state)


    def render(self, mode='human'):
        pass
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(500, 500)
    #         self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
    #         rod = rendering.make_capsule(1, .2)
    #         rod.set_color(.8, .3, .3)
    #         self.pole_transform = rendering.Transform()
    #         rod.add_attr(self.pole_transform)
    #         self.viewer.add_geom(rod)
    #         axle = rendering.make_circle(.05)
    #         axle.set_color(0, 0, 0)
    #         self.viewer.add_geom(axle)
    #         fname = path.join(path.dirname(__file__), "assets/clockwise.png")
    #         self.img = rendering.Image(fname, 1., 1.)
    #         self.imgtrans = rendering.Transform()
    #         self.img.add_attr(self.imgtrans)
    #
    #     self.viewer.add_onetime(self.img)
    #     self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
    #     if self.last_u:
    #         self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_buyer_random_price(self):
        return self.np_random.uniform(self.min_buyer_price, self.max_seller_price,self.num_of_buyer)



# def angle_normalize(x):
#     return (((x + np.pi) % (2 * np.pi)) - np.pi)