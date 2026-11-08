#include <iostream>
#include <Windows.h>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;


int main()
{
    SetConsoleOutputCP(CP_UTF8);

    //3个状态，两个动作(动作0: 向左发力, 动作1: 向右发力)
    double P[3][2][3]={
        //状态0
        {{0.9,0.1,0.0},
             {0.1,0.8,0.1}},
        //状态1
        {{0.8,0.2,0.0},
             {0.0,0.2,0.8}},
        //状态2
        {{0.0,0.2,0.8},
             {0.0,0.1,0.9}}
    };

    //奖励R[s][a]
    double R[3][2]={
        {-1.0,-1.0},
        {-1.0,-1.0},
        {10.0,10.0}
    };

    //--------引入Q表
    //Q[s][a] 代表在状态s下执行动作a的价值
    double Q[3][2]={0.0};

    double alpha=0.1; //学习率
    double gamma=0.9; //折扣因子
    // int current_state=1;
    // int steps=50;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1.0);

    int episodes=1000000; //进行1000局游戏训练

    cout<<"---开始Q-Learning训练---"<<endl;


    for(int ep=0;ep<episodes;++ep)
    {
        int current_state=1; //每一句都从状态1出发
        //每局最多走20步 防止死循环
        for(int step=0;step<20;step++)
        {
            //1.选择动作（epsilon-greedy策略）
            int action;
            if(dis(gen)<0.1)
            {
                action=dis(gen)>0.5?1:0;
            }else 
            {
                //90%的概率选择Q值最大的动作
                action =(Q[current_state][0]>Q[current_state][1]) ? 0:1;
            }
            double reward=R[current_state][action];

            //2.模拟环境转移
            double rand_val=dis(gen);
            double cumulative_prob=0.0;
            int next_state=current_state;
            for(int s_next=0;s_next<3;++s_next)
            {
                cumulative_prob+=P[current_state][action][s_next];
                if (rand_val<=cumulative_prob)
                {
                    next_state=s_next;
                    break;
                }
            }

            if (next_state == 2) {
                reward += 10.0; 
            }
            //3.寻找下一个状态能获得的最大Q值
            double max_next_Q=max(Q[next_state][0],Q[next_state][1]);

            //4. Q-Learning 更新公式 (贝尔曼最优方程的代码实现)
            // Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            Q[current_state][action]+=alpha*(reward+gamma*max_next_Q-Q[current_state][action]);
            current_state=next_state;

            if(current_state==2)
            {
                break;
            }
        }
    }
    // 打印训练好的 Q 表
    cout << "\n--- 训练结束，打印 Q 表 ---" << endl;
    for (int s = 0; s < 3; ++s) {
        cout << "状态 " << s << ": ";
        for (int a = 0; a < 2; ++a) {
            cout << "Q(s" << s << ",a" << a << ")=" << Q[s][a] << "  ";
        }
        cout << endl;
    }

    cout << "\n结论：" << endl;
    cout << "在状态 1 时，向左走(0)的价值是 " << Q[1][0] << "，向右走(1)的价值是 " << Q[1][1] << endl;
    if (Q[1][1] > Q[1][0]) {
        cout << "AI 学会了：在状态 1 应该向右走 (动作 1) 才能最快到达终点！" << endl;
    }

       
    
    return 0;
}