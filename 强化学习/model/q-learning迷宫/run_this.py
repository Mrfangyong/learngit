"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env  环境
            env.render()

            # RL choose action based on observation  基于观测值挑选动作
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward  RL采取行动，获得下一步的观察和奖励
            observation_, reward, done = env.step(action)

            # RL learn from this transition  RL从这个过渡中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation交换的观察
            observation = observation_

            # break while loop when end of this episode在本集结束时，中断while循环
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()