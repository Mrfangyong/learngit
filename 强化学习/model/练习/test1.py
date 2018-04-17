# 1.组循环：（获取初始化q表，遍历训练次数）

import numpy as np
import pandas as pd
import time
np.random.seed(2)
n_states=5
actions=["left","right"]
epsilon=0.9
alpha=0.1
Lambda=0.9
max_episodes=13
fresh_time=0.1
def build_q_table():
    pass
def rl():
    q_table=build_q_table(n_states,actions)
    for epsilon in range(max_episodes):
        step_counter=0
        S=0
    
if __name__=="__main__":
    q_table=rl()
    print(q_table)
