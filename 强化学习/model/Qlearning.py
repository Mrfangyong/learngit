import numpy as np
import pandas as pd
import time

np.random.seed(2)#伪随机数列

n_states=5   #最开始到保障的步数
actions=["left","right"]    #左右
epsilon=0.9    #选择动作时候  90%选择最优动作，10%选择随机动作
alpha=0.1      #学习效率
LAMBDA=0.9     #衰减度  未来奖励的衰减值
max_episodes=13    #只玩13回合
fresh_time=0.1     #规定走一步花的时间

def build_q_table(n_states,actions):
    table=pd.DataFrame(
        np.zeros((n_states,len(actions))),
        columns=actions
        )
    print(table)
    return table


# build_q_table(n_states, actions)
# 选择动作
def choose_action(state,q_table):
    state_actions=q_table.iloc[state,:]
    if (np.random.uniform()>epsilon) or (state_actions.all()==0):
        action_name=np.random.choice(actions)
    else:
        action_name=state_actions.argmax()
    print(action_name)    
    return action_name

# choose_action(state, q_table)    


# 环境
def get_env_feedback(S,A):
    if A=="right":
        if S==n_states-2:
            S_="terminal"
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R                    
                
def updata_env(S,episode,step_counter):
    env_list=["-"]*(n_states-1)+["T"]
#     print(env_list)
    if S=="terminal":
        interaction="epi%s:total_steps= %s"%(episode+1,step_counter)
        print("\r{}".format(interaction),end="")
        time.sleep(2)
        print("\r                              ",end="")
    else:
        env_list[S]="o"
        interaction="".join(env_list)
        print("\r{}".format(interaction),end="")
        time.sleep(fresh_time)    
        
        
# 组循环        
def rl():       
    q_table=build_q_table(n_states, actions) 
    for episode in range(max_episodes):
        step_counter=0
        S=0
        is_terminated=False
        updata_env(S, episode, step_counter)
        while not is_terminated:
            A=choose_action(S, q_table)
            S_,R=get_env_feedback(S, A)
            q_predict=q_table.ix[S,A]
            if S_ !="terminal":
                q_target=R+LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target=R 
                is_terminated=True
            q_table.ix[S,A]+=alpha*(q_target-q_predict)
            S=S_
            updata_env(S, epsilon, step_counter+1)
            step_counter+=1
    print(q_table)        
    return q_table
if __name__=="__main__":
    q_table=rl()
    print("\r\nQ_table:\n")
    print(q_table)        
                   
        
        
        
        
        
        
        
        
        
        
        
        
        





























