
#include <iostream>
#include <windows.h>
#include <random>
#include <cmath>
#include <winnls.h>


using namespace std;

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    double P[3][3]={
        {0.2,0.5,0.3},
        {0.4,0.1,0.5},
        {0.2,0.2,0.6}
    };

    //状态奖励R(s):进入状态时可以拿到钱
    double R[3]={1.0,-1.0,5.0};
    double gamma=0.9;

    int current_state=0;
    int steps=10;
    double G_t=0.0;//累计回报Return

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1.0);

    cout<<"--- MRD 模拟---  gamma:"<<gamma<<endl;
    for(int step=0;step<steps;++step)
    {
        //拿到当前状态的奖励
        double current_reward=R[current_state];

        //计算折扣后的奖励，累加到G_t中
        double discounted_reward=pow(gamma,step)*current_reward;
        G_t+=discounted_reward;

        cout<<"第"<<step<<"步：状态："<<current_state
            <<",奖励："<<current_reward
            <<",折扣后："<<discounted_reward<<endl;

        //随机转移到下一个状态
        double rand_val=dis(gen);
        double cumulative_prob=0.0;
        for(int s_next=0;s_next<3;++s_next)
        {
            cumulative_prob+=P[current_state][s_next];
            if (rand_val<=cumulative_prob)
            {
                current_state=s_next;
                break;
            }
        }
    }
    cout<<"总累计回报 G_t ="<<G_t<<endl;

    return 0;
}