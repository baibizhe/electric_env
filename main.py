import gym
import numpy as  np
from ddpf2 import Agent
import  matplotlib.pyplot as plt

# from utils import  plotLearning
#todo : 改成p-replay buffer
if __name__ == '__main__':

    env = gym.make('gym_foo:foo-v0')
    agent = Agent(alpha=0.1, beta=0.2, input_dims=[3], tau=0.01, env=env, batch_size=32, layer1_size=500,
                  layer2_size=300, n_actions=3)
    np.random.seed(0)
    score_history = []
    EPISODES = 300
    result = np.zeros((EPISODES,agent.n_actions+1))
    for i in range(EPISODES):
        obs = env.reset()
        print(i)
        print("episode start with %s"%(str(obs)))
        done = False
        j = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            obs = new_state
            # print("reward %.2f , obs%s , act %s,new_state %s"%(reward,str(obs),str(act),str(new_state)))
            j+=1
            if j==400:
                done = True
                result[i] = np.concatenate((new_state,[reward]))
                # result[i] = new_state
        # volume = sum([new_state[1], new_state[3], new_state[5]])
        # print("ending seller price and volume ,average profit  :%s %d" % (str(new_state), reward / volume))
        score_history.append(reward)
        print('episode ', i, 'score %.2f' % reward, '100 game vag %.2f' % np.mean(score_history[-100:]))
        print("-----------------------------------------------")
    filename = 'pendulum.png'
    x = range(len(result))
    plt.subplot(3,1,1)
    plt.title('price line ')
    plt.plot(x,result[:,0])
    plt.plot(x, result[:, 1])
    plt.plot(x, result[:, 2])

    # plt.subplot(3,1,2)
    # plt.title('volume line ')
    # plt.plot(x, result[:, 1])
    # plt.plot(x, result[:, 3])
    # plt.plot(x, result[:, 5])

    plt.subplot(3,1,3)
    plt.title('total profit line')
    plt.plot(x, result[:, 3],label = "reward",color= 'coral')
    plt.legend(loc='upper right')

    plt.savefig("my main only with prices")
    plt.show()
