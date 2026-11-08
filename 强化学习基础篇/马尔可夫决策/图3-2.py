import numpy as np

np.random.seed(0)

#定义状态转移概率矩阵p

P=[
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

P=np.array(P)
rewards=[-1,-2,-2,10,1,0] #定义奖励函数
gamma=0.5

#从给定的序列计算回报
def compute_return(start_index,chain,gamma):
    G=0
    for i in reversed(range(start_index,len(chain))):
        G=gamma*G+rewards[chain[i]-1]
    return G


def compute(P,rewards,gamma,states_num):
    rewards=np.array(rewards).reshape((-1,1))
    value=np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*P),rewards)

    return value


if __name__ == '__main__':

    chain=[1,2,3,6]
    start_index=0

    G=compute_return(start_index,chain,gamma)
    print(f"根据本序列计算得到的回报为：{G}")

    V=compute(P,rewards,gamma,6)
    print(f"MRP中每个状态价值分别为\n{V}")