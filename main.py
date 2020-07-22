import gym
import numpy as  np
from ddpf2 import Agent
import  matplotlib.pyplot as plt
import  tensorflow as tf
import  os
tf.compat.v1.reset_default_graph()
# from utils import  plotLearning
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#todo : 改成p-replay buffer
# todo: 给噪声加衰减函数
TAU = 0.01
BATCH_SIZE = 256
BUFFER_SIZE = 50000
NUM_OF_SELLER=5
N_ACTIONS = NUM_OF_SELLER*2
NUM_OF_BUYER=9
def run():
    alphas = [0.0005,0.001,0.005,0.01,0.05,0.1]
    betas = [0.0007,0.002,0.007,0.02,0.07,0.2]
    for ii in range((len(alphas))):
        env = gym.make('gym_foo:foo-v0')
        agent = Agent(alpha=alphas[ii], beta=betas[ii], input_dims=[N_ACTIONS+NUM_OF_BUYER*2], tau=TAU, env=env, batch_size=BATCH_SIZE, layer1_size=500,
                      layer2_size=300, n_actions=N_ACTIONS,buffer_max_size=BUFFER_SIZE)
        tf.compat.v1.reset_default_graph()
        np.random.seed(0)
        score_history = []
        EPISODES = 500
        result = np.zeros((EPISODES,agent.n_actions+1+1))
        for i in range(EPISODES):
            obs = env.reset()
            print(i)
            print(" episode start with"+str(["%.2f" % val for val in obs[0:NUM_OF_SELLER * 2]]))
            done = False
            j = 0
            while not done:
                act = agent.choose_action(obs).astype(np.float16)
                # print("action is %s"%str(act))
                new_state, reward, done, info = env.step(act)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                obs = new_state
                # if reward>0:
                #     print(j)
                #     print("reward > 0:episode end with  "+str(["%.2f"%val for val in new_state]))
                # print("reward is :%.2f \n act is:%s \n new price volume is :%s"%(reward,str(act),str(new_state)))
                j+=1
                if j==3500:
                    done = True
                    result[i] = np.concatenate((new_state[0:NUM_OF_SELLER*2],[reward,np.mean(score_history[-50:])]))
                    print("episode end with  "+str(["%.2f"%val for val in new_state[0:NUM_OF_SELLER*2]]))
                    # result[i] = new_state
            # volume = sum([new_state[1], new_state[3], new_state[5]])
            # print("ending seller price and volume ,average profit  :%s %d" % (str(new_state), reward / volume))
            score_history.append(reward)
            print('episode ', i, 'score %s' % str(format(reward,",.1f")), '100 game vag %.2f' % np.mean(score_history[-100:]))
            print("-----------------------------------------------")
        filename = 'pendulum.png'
        x = range(len(result))
        plt.figure(figsize=(6,10))
        plt.subplot(4,1,1)
        plt.subplots_adjust(hspace=0.7)
        plt.title('price line ,alpha = %s beta =%s'%(str(alphas[ii]),str(betas[ii])))
        plt.plot(x,result[:,0])
        plt.plot(x, result[:, 2])
        plt.plot(x, result[:, 4])
        plt.plot(x, result[:, 6])
        plt.plot(x, result[:, 8])



        plt.subplot(4,1,2)
        plt.title('volume line ')
        plt.plot(x, result[:, 1])
        plt.plot(x, result[:, 3])
        plt.plot(x, result[:, 5])
        plt.plot(x, result[:, 7])
        plt.plot(x, result[:, 9])



        plt.subplot(4,1,3)
        plt.title('total profit line')
        plt.plot(x, result[:, NUM_OF_SELLER*2],label = "reward",color= 'coral')
        plt.legend(loc='upper right')

        plt.subplot(4, 1, 4)
        plt.title('50 game avg')
        plt.plot(x, result[:, NUM_OF_SELLER * 2+1], label="avg", color='blue')
        plt.legend(loc='upper right')

        plt.savefig("image/my main with round %d"%ii)
        plt.clf()
        plt.close()
        # plt.show()


run()
