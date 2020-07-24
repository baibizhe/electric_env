import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from os import path
from utils import rank, generate_volume, select_clear_amount,calculate_total_amount


class FooEnv(gym.Env):


    def __init__(self, min_buyer_price=300, max_seller_price=370, min_seller_price=300, max_buyer_price=400, \
                 num_of_seller=5, num_of_buyer=9, seller_data="data/seller_data", buyer_data="data/buyer_data"):


        self.max_buyer_price = max_buyer_price  # todo: 需要买房最高价格吗？
        self.min_buyer_price = min_buyer_price  # todo: 需要吗？
        self.num_of_seller = num_of_seller
        self.num_of_buyer = num_of_buyer
        self.buyer_volume = np.loadtxt(buyer_data)
        self.num_of_buyer = len(self.buyer_volume)
        self.buyer_price =   np.random.uniform(self.min_buyer_price, self.max_buyer_price, self.num_of_buyer).astype(
                np.float16)# 找到当前状态的买方随机价格
        self.max_seller_price = max_seller_price  # 400元
        self.min_seller_price = min_seller_price  # 300元
        self.set_data_for_seller(seller_data)
        self.num_of_seller = len(self.costfuncton_for_sellers)
        # self.deltaprice = 0.1
        # self.delta_volume = 10
        self.min_seller_volume = 0  # max_seller_volume在 set_data_for_seller()定义了
        #
        del_p = 0.5
        del_v = 50
        act_high = np.array([[del_p,del_v]*self.num_of_seller]).flatten()
        act_low= np.array([[-del_p,-del_v]*self.num_of_seller]).flatten()
        self.buyer_name = ["buyer_%d" % i for i in range(self.num_of_buyer)]  # 随机取的名字
        self.seller_name = ["seller_%d" % i for i in range(self.num_of_seller)]  # 随机取的名字
        self.action_space = spaces.Box(low=act_low,
                                       high=act_high, shape=(self.num_of_seller * 2,), dtype=np.float16)

        self.observation_space = spaces.Box(low=0, high=np.max(self.max_seller_volume),
                                            shape=(self.num_of_seller * 2 + self.num_of_buyer * 2,), \
                                            dtype=np.float16)

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
        index = 0
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
        total_match_volume = calculate_total_amount(match_result)
        factor= total_match_volume/sum(self.max_seller_volume)
        reported_volume = [self.state[i]*factor for i in range(1, self.num_of_seller * 2, 2)]

        # reported_volume  = [i[2] for i in match_result]
        cost_result = self._calculate_cost(reported_volume)
        reward = 0
        for i in range(self.num_of_seller):
            volume = select_clear_amount(self.seller_name[i], match_result)
            reward += volume * clear_price - cost_result[i]

        return np.array(self.state), reward, False, {}

    def reset(self):
        seller_volume  =self.max_seller_volume
        seller_volume = [i+np.random.uniform(-200,200) for i in seller_volume]
        # seller_volume = generate_volume(self.num_of_seller, int(sum(self.max_seller_volume)))  # 产生随机数 固定总量是150%
        #todo :reset的时候随机状态
        seller_prcie = np.array([self.max_seller_price] * self.num_of_seller)
        buyer_volume = self.buyer_volume  # 产生随机数 固定总量是100%
        buyer_price = self.buyer_price  # 找到当前状态的买方随机价格
        self.state = np.array([],dtype=np.float32)
        for i in range(0,self.num_of_seller):
            self.state = np.concatenate((self.state,[seller_prcie[i]]))
            self.state = np.concatenate((self.state,[seller_volume[i]]))
        for i in range(0, self.num_of_buyer):
            self.state = np.concatenate((self.state, [buyer_price[i]]))
            self.state = np.concatenate((self.state, [buyer_volume[i]]))
        # [seller_price ,seller_volume , .........,buyer_price,buyer_volume,buyer_price,buyer_volume]
        return np.array(self.state)

    def render(self, mode='human'):
        pass


    def _get_match_result(self):
        seller_data, buyer_data = [], []
        for i in range(0, self.num_of_seller * 2, 2):
            seller_data.append([self.state[i], self.state[i + 1]])
        for i in range(self.num_of_seller * 2, self.num_of_buyer * 2, 2):
            buyer_data.append([self.state[i], self.state[i + 1]])
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



