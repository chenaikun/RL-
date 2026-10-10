import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self,K):
        self.probs=np.random.uniform(size=K)
        self.best_idx=np.argmax(self.probs)
        self.best_prob=self.probs[self.best_idx]
        self.K=K


    def step(self,K):
        if np.random.rand() < self.probs[K]:
            return 1
        else:
            return 0



#多臂老虎机算法基本框架
class Solver:
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.K)
        self.regret=0. #积累的懊悔
        self.actions=[] #每一步动作的列表
        self.regrets=[] #积累懊悔的列表

    def update_regret(self,k):
        self.regret+=self.bandit.best_prob-self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
         # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
         raise NotImplementedError

    def run(self,num_steps):
        #运行一定次数，num_steps为总运行次数
        for _ in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)


# np.random.seed(1)
# K=10
# bandit_10_arm=BernoulliBandit(K)
#print(f"{type(bandit_10_arm.probs)}")
# print(f"生成{K}臂老虎机\n")
# print(f"获奖概率最大的是{bandit_10_arm.best_idx}号，其获奖概率为{bandit_10_arm.best_prob}")
# print(f"{bandit_10_arm.step(1)}")