import gym
import numpy as  np
from ddpf2 import Agent

# from utils import  plotLearning

import  matplotlib.pyplot as plt
if __name__ == '__main__':
    # from gym import envs
    # print(envs.registry.env_specs)
    # print(envs.registry.all())
    env = gym.make('gym_foo:foo-v0')
    agent = Agent(alpha=0.1, beta=0.2, input_dims=[4], tau=0.02, env=env, batch_size=64, layer1_size=400,
                  layer2_size=300, n_actions=4)
    np.random.seed(0)
    score_history = []
    size = 400
    result = np.zeros((size,agent.n_actions))
    for i in range(size):
        obs = env.reset()
        # print("asdasda",obs)
        done = False
        score = 0
        print(i)
        j = 0
        while not done:
            act = agent.choose_action(obs)
            # print(act)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            # print("reward %.2f , obs%s , act %s,new_state %s"%(reward,str(obs),str(act),str(new_state)))
            j+=1
            # print(j)
            # print("seller -price :%s" % (str(new_state)))

            if j==200:
                done = True
                # result[i] = np.concatenate((new_state,[reward]))
                result[i] = new_state
                print("seller -price :%s" % (str(new_state)))
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score, '100 game vag %.2f' % np.mean(score_history[-100:]))
    filename = 'pendulum.png'
    # x = range(len(result))
    # plt.plot(x,result[:,0])
    # plt.plot(x, result[:, 1])
    # plt.plot(x, result[:, 2])
    # plt.plot(x, result[:, 3],label = "reward",color= 'coral')
    # plt.legend(loc='upper right')


    plt.show()
        # plotLearning(score_history,filename,window=100 )
