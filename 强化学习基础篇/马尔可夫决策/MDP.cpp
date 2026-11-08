#include <iostream>
#include <Windows.h>
#include <random>



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

    int current_state=1;
    int steps=50;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1.0);
    cout<<"--- MDP 模拟--- "<<endl;


    for(int step=0;step<steps;++step)
    {
        int action=dis(gen)>0.5?1:0;
        double reward=R[current_state][action];
        cout<<"第"<<step<<"步：状态"<<current_state
            <<"采取动作"<<action
            <<"获取奖励"<<reward;

        //根据 P(s'|s,a) 决定下一个状态
        double rand_val=dis(gen);
        double cumulative_prob=0.0;
        int next_state=current_state;

        for (int s_next = 0; s_next < 3; ++s_next) 
        {
            cumulative_prob += P[current_state][action][s_next];
            if (rand_val <= cumulative_prob) 
            {
                next_state = s_next;
                break;
            }
        }

        cout << " -> 转移到状态 " << next_state << endl;
        current_state=next_state;

        if(current_state==2)
        {
            cout<<"到达状态2，游戏结束！" << endl;
            break;
        }
    }
    return 0;
}