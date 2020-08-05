import gym
import numpy as  np
from ddpf2 import Agent
import  matplotlib.pyplot as plt
import  tensorflow as tf
import  numpy as np
import  os
tf.compat.v1.reset_default_graph()
# from utils import  plotLearning
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#todo : 改成p-replay buffer
# todo: 给噪声加衰减函数
TAUS = [0.1,0.01,0.001]
BATCH_SIZES = [16,32,128]
GAMMAS =[0.9,0.999,0.99]
BUFFER_SIZE = 50000
NUM_OF_SELLER=len(np.loadtxt("data/seller_data"))

N_ACTIONS = NUM_OF_SELLER*2
NUM_OF_BUYER=42

def run():
    graph_num = 0
    alphas = [0.001,0.0001,0.0002,0.001]
    betas = [0.002,0.0001,0.002,0.005]
    for TAU in TAUS:
        for GAMMA in GAMMAS:
            for BATCH_SIZE in BATCH_SIZES:
                for ii in range((len(alphas))):
                    graph_num+=1
                    env = gym.make('gym_foo:foo-v0')
                    agent = Agent(alpha=alphas[ii], beta=betas[ii], input_dims=[N_ACTIONS+NUM_OF_BUYER*2], tau=TAU, env=env, batch_size=BATCH_SIZE, layer1_size=128,
                                  layer2_size=128, n_actions=N_ACTIONS,buffer_max_size=BUFFER_SIZE)
                    tf.compat.v1.reset_default_graph()
                    np.random.seed(0)
                    score_history = []
                    EPISODES =1000
                    result = np.zeros((EPISODES,NUM_OF_SELLER+1+1+1))
                    for i in range(EPISODES):
                        obs = env.reset()
                        print(i)
                        # print(" episode start with"+str(["%.2f" % val for val in obs[0:NUM_OF_SELLER * 2]]))
                        done = False
                        j = 0
                        while not done:
                            act = agent.choose_action(obs).astype(np.float16)
                            # print("action is %s"%str(act))
                            j += 1
                            if j == 300:
                                new_state, reward, done, info = env.step(act)
                                agent.remember(obs, act, reward, new_state, int(done))
                                agent.learn()
                                done = True
                                result[i] = np.concatenate((info, [reward]))
                                break
                            new_state, reward, done, info = env.step(act)
                            agent.remember(obs, act, reward, new_state, int(done))
                            agent.learn()
                            obs = new_state
                            # if reward>0:
                            #     print(j)
                            #     print("reward > 0:episode end with  "+str(["%.2f"%val for val in new_state]))
                            # print("reward is :%.2f \n act is:%s \n new price volume is :%s"%(reward,str(act),str(new_state)))
                            # if reward>0:
                            #     print(reward,new_state)

                                # print("episode end with  "+str(["%.2f"%val for val in new_state[0:NUM_OF_SELLER*2]]))
                                # result[i] = new_state
                        # volume = sum([new_state[1], new_state[3], new_state[5]])
                        score_history.append(reward)
                        print('第%s幕'%str(i), '得分（利润） %s' % str(format(reward,",.1f")), '100 game vag %.2f' % np.mean(score_history[-100:]))
                        # print("-----------------------------------------------")
                    # filename = 'pendulum.png'
                    x = range(len(result))
                    plt.figure(figsize=(8,15))
                    plt.rcParams['font.sans-serif'] = ['SimHei']
                    plt.subplot(5,1,1)
                    plt.subplots_adjust(hspace=0.7)
                    plt.title('随机挑选的大发电商利润 \n,alpha = %s beta =%s ,gamma = %.3f,batch size = %.1f,tau = %.3f'%(str(alphas[ii]),str(betas[ii]),GAMMA,BATCH_SIZE,TAU))
                    plt.plot(x,result[:,0])
                    plt.plot(x, result[:, 4])


                    plt.subplot(5,1,2)
                    plt.title('随机挑选的中等发电商利润 ')
                    plt.plot(x, result[:, 9])
                    plt.plot(x, result[:, 15])

                    plt.subplot(5, 1, 3)
                    plt.title('随机挑选的小发电商利润 ')
                    plt.plot(x, result[:, 23])
                    plt.plot(x, result[:, 25])

                    plt.subplot(5, 1, 4)
                    plt.title('总成交容量')
                    plt.plot(x, result[:,-3], label="总成交容量", color='blue')
                    plt.legend(loc='upper right')

                    plt.subplot(5,1,5)
                    plt.title('总利润')
                    plt.plot(x, result[:,-1],label = "发电商总利润",color= 'coral')
                    plt.legend(loc='upper right')



                    plt.savefig("image/my main with round %d"%graph_num)
                    plt.clf()
                    plt.close()
                    # plt.show()


run()
