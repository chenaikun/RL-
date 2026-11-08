#include <iostream>
#include <windows.h>
#include <random>

using namespace std;

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    double P[3][3]={
        {0.2,0.5,0.3},
        {0.4,0.1,0.5},
        {0.2,0.2,0.6}
    };
    
    int current_state=0; //最初状态0
    int steps=10;
    //随机数生成器
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0,1.0);

    cout<<"---MP 模拟---"<<endl;
    cout<<"起点：状态"<<current_state<<endl;

    for(int step=1;step<=steps;++step)
    {
        double rand_val=dis(gen);
        double cumulative_prob=0.0;
        int next_state=current_state;

        //根据概率矩阵决定下一个状态
        for(int s_next=0;s_next<3;++s_next)
        {
            cumulative_prob+=P[current_state][s_next];
            if(rand_val<=cumulative_prob)
            {
                next_state=s_next;
                break;
            }
        }
        cout<<"第"<<step<<"步：状态："<<current_state<<"->状态："<<next_state<<endl;
        current_state=next_state;
    }
    return 0;
}

