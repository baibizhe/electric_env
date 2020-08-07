import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from os import path
from utils import rank, generate_volume, select_clear_amount,calculate_total_amount

#这一版本env ， observation space是经过action 更新后的卖方的申报价格和容量，dim = 卖方数量*2，action_space是变化量

class FooEnv(gym.Env):


    def __init__(self, min_buyer_price=300, max_seller_price=400, min_seller_price=300, max_buyer_price=370, \
                 num_of_seller=5, num_of_buyer=9, seller_data="data/seller_data", buyer_data="data/buyer_data"):


        self.max_buyer_price = max_buyer_price  # todo: 需要买房最高价格吗？
        self.min_buyer_price = min_buyer_price  # todo: 需要吗？
        self.buyer_volume = np.loadtxt(buyer_data)
        self.num_of_buyer = len(self.buyer_volume)
        self.buyer_price =   np.random.uniform(self.min_buyer_price, self.max_buyer_price, self.num_of_buyer)# 找到当前状态的买方价格
        self.max_seller_price = max_seller_price
        self.min_seller_price = min_seller_price

        self.set_data_for_seller(seller_data)
        self.num_of_seller = len(self.costfuncton_for_sellers)

        self.min_seller_volume = 0  # max_seller_volume在 set_data_for_seller()定义了
        #
        del_p = 0.5
        del_v = 100
        act_high = np.array([[del_p,del_v]*self.num_of_seller]).flatten()
        act_low= np.array([[-del_p,-del_v]*self.num_of_seller]).flatten()
        self.buyer_name = ["buyer_%d" % i for i in range(self.num_of_buyer)]  # 随机取的名字
        self.seller_name = ["seller_%d" % i for i in range(self.num_of_seller)]  # 随机取的名字
        self.action_space = spaces.Box(low=act_low,
                                       high=act_high, shape=(self.num_of_seller * 2,))

        self.observation_space = spaces.Box(low=0, high=np.max(self.max_seller_volume),
                                            shape=(self.num_of_seller * 2 ,) )

        self.seed()

    def set_data_for_seller(self, path="data/seller_data"):
        """"
        设置卖方的成本函数，
        例子 [[76633,0.0016 , 102 , 4,560,000],[....],[....].. ] :
        第一个卖家 最大申报电量=76633, a_b = 0.0016 , b_g = 102 , c_g =4,560,000
        总共有 num_of_seller个 ， 成本函数默认从data/seller_data.txt中读取
        """
        result = np.loadtxt(path)
        self.max_seller_volume = result[:, 0]  # 设置卖方的最大申报电量
        self.costfuncton_for_sellers = result[:, 1:result.shape[1]]  # 列切片，去掉第一列。
        return result

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _update_state(self, action):
        index = 0  #index是为了在max_seller_volume找到每个用户的最大申报量，以免超过那个量
        for i in range(len(action)):
            new_num = self.state[i] + action[i]
            if i % 2 == 0:
                self.state[i] = min(self.max_seller_price, max(self.min_seller_price, new_num))
            else:
                self.state[i] = min(self.max_seller_volume[index], max(0, new_num))
                index += 1

    def step(self, action: np.ndarray):
        self._update_state(action)  # update self state with action
        match_result, clear_price = self._get_match_result()
        # for i in match_result:
        #     if 0 in i:
        #         for index in range(len(self.seller_name)):
        #             print("factor is ",select_clear_amount(self.seller_name[index],match_result)/self.state[index*2+1])
        #             print(select_clear_amount(self.seller_name[index],match_result),self.state[index*2+1])
        #         print(self.state)
        #         print(action)
        #         print(match_result)

        # print(match_result)
        reported_volume = [self.state[i] for i in range(1, self.num_of_seller * 2, 2)]

        # reported_volume  = [i[2] for i in match_result]
        cost_result = self._calculate_cost(reported_volume)
        reward = 0
        info = []
        for i in range(self.num_of_seller):
            volume = select_clear_amount(self.seller_name[i], match_result)
            single_r = volume * clear_price - cost_result[i]
            reward += single_r
            info.append(single_r)
        info.append(calculate_total_amount(match_result))
        info.append(clear_price)
        return np.array(self.state), reward, False, info

    def reset(self):
        seller_volume  =self.max_seller_volume
        seller_volume = [i+np.random.uniform(-200,200) for i in seller_volume]
        # seller_volume = generate_volume(self.num_of_seller, int(sum(self.max_seller_volume)))  # 产生随机数 固定总量是150%
        #todo :reset的时候随机状态
        seller_prcie = np.array([self.max_seller_price] * self.num_of_seller)
        self.state = np.array([])
        for i in range(0,self.num_of_seller):
            self.state = np.concatenate((self.state,[seller_prcie[i]]))
            self.state = np.concatenate((self.state,[seller_volume[i]]))

        # [seller_price ,seller_volume , .........,buyer_price,buyer_volume,buyer_price,buyer_volume]
        return np.array(self.state)

    def render(self, mode='human'):
        pass


    def _get_match_result(self):
        seller_data, buyer_data = [], []
        for i in range(0, self.num_of_seller * 2, 2):
            seller_data.append([self.state[i], self.state[i + 1]])

        for i in range(0, self.num_of_buyer):
            buyer_data.append([self.buyer_price[i], self.buyer_volume[i]])
        match_result, clear_price = rank(self.buyer_name, buyer_data, self.seller_name, seller_data)
        return match_result, clear_price

    def _calculate_cost(self, reported_volume):
        """
        :param reported_volume: 申报的电量:list ，不是匹配后的电量
        :return: 长度为 num_of_seller的list，里面是float
        """
        result = np.zeros(self.num_of_seller)
        for i in range(self.num_of_seller):
            cost = self.costfuncton_for_sellers[i][0] * reported_volume[i] ** 2 + self.costfuncton_for_sellers[i][1] * \
                   reported_volume[i] + self.costfuncton_for_sellers[i][2]
            result[i]=cost
        return result



