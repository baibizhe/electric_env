import gym
import numpy as  np
from ddpf2 import Agent

# from utils import  plotLearning
#todo : 改成p-replay buffer
import  matplotlib.pyplot as plt
if __name__ == '__main__':

    env = gym.make('gym_foo:foo-v0')
    #todo : actor learning rate is 1e-4 ,critic learning rate is 1e-3
    agent = Agent(alpha=1e-4, beta=1e-3, input_dims=[6], tau=0.02, env=env, batch_size=64, layer1_size=400,
                  layer2_size=300, n_actions=6)
    np.random.seed(0)
    score_history = []
    size = 100
    result = np.zeros((size,agent.n_actions+1))
    for i in range(size):
        obs = env.reset()
        # print("init seller -price :%s" % (str(obs)))
        #
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

            if j==1000:
                done = True
                result[i] = np.concatenate((new_state,[reward]))
                # result[i] = new_state
                avg = sum([new_state[1],new_state[3],new_state[5]])
                print("ending seller price and volume ,average profit  :%s %d" % (str(new_state),reward/avg))
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score, '100 game vag %.2f' % np.mean(score_history[-100:]))
    filename = 'pendulum.png'
    x = range(len(result))
    plt.subplot(3,1,1)

    plt.title('price line ')
    plt.plot(x,result[:,0])
    plt.plot(x, result[:, 2])
    plt.plot(x, result[:, 4])

    plt.subplot(3,1,2)
    plt.title('volume line ')
    plt.plot(x, result[:, 1])
    plt.plot(x, result[:, 3])
    plt.plot(x, result[:, 5])

    plt.subplot(3,1,3)
    plt.title('total profit line')
    plt.plot(x, result[:, 6],label = "reward",color= 'coral')
    plt.legend(loc='upper right')


    plt.show()
        # plotLearning(score_history,filename,window=100 )
