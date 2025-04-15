# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:25:04 2023

@author: BR11WP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:47:21 2023

@author: BR11WP
"""


# Q learning for the 4x4 Gridworld example


from typing import List
import numpy as np


# To generate the dynamics
def n_state(state,action):
    if state == 1:

        if action == 0:
            next_state = 1
            r = -1
        elif action == 1:
            next_state = 5
            r = -1
        elif action == 2:
            next_state = 2
            r = -1
        else:
            next_state = 0
            r = -1
            
    elif state == 2:
        
        if action == 0:
            
            next_state = 2
            
            r = -1
            
        elif action == 1:
            
            next_state = 6
            
            r = -1
            
        elif action == 2:
            
            next_state = 3
            
            r = -1
            
        else:
            
            next_state = 1
            
            r = -1 
            
    elif state == 3:
        
        if action == 0:
            
            next_state = 3
            
            r = -1
            
        elif action == 1:
            
            next_state = 7
            
            r = -1
            
        elif action == 2:
            
            next_state = 3
            
            r = -1
            
        else:
            
            next_state = 2
            
            r = -1 
            
    elif state == 4:
        
        if action == 0:
            
            next_state = 0
            
            r = -1
            
        elif action == 1:
            
            next_state = 8
            
            r = -1
            
        elif action == 2:
            
            next_state = 5
            
            r = -1
            
        else:
            
            next_state = 4
            
            r = -1 
            
    elif state == 5:
        
        if action == 0:
            
            next_state = 1
            
            r = -1
            
        elif action == 1:
            
            next_state = 9
            
            r = -1
            
        elif action == 2:
            
            next_state = 6
            
            r = -1
            
        else:
            
            next_state = 4
            
            r = -1 
            
    elif state == 6:
        
        if action == 0:
            
            next_state = 2
            
            r = -1
            
        elif action == 1:
            
            next_state = 10
            
            r = -1
            
        elif action == 2:
            
            next_state = 7
            
            r = -1
            
        else:
            
            next_state = 5
            
            r = -1 
            
    elif state == 7:
        
        if action == 0:
            
            next_state = 3
            
            r = -1
            
        elif action == 1:
            
            next_state = 11
            
            r = -1
            
        elif action == 2:
            
            next_state = 7
            
            r = -1
            
        else:
            
            next_state = 6
            
            r = -1 
            
    elif state == 8:
        
         
        if action == 0:
             
            next_state = 4
             
            r = -1
             
        elif action == 1:
             
            next_state = 12
             
            r = -1
             
        elif action == 2:
             
            next_state = 9
             
            r = -1
             
        else:
             
            next_state = 8
             
            r = -1 
             
    elif state == 9:
        
            
        if action == 0:
            
            next_state = 5
             
            r = -1
             
        elif action == 1:
             
            next_state = 13
             
            r = -1
             
        elif action == 2:
             
            next_state = 10
             
            r = -1
             
        else:
             
            next_state = 8
             
            r = -1 
            
            
    elif state == 10:
        
            
        if action == 0:
            
            next_state = 6
             
            r = -1
             
        elif action == 1:
             
            next_state = 14
             
            r = -1
             
        elif action == 2:
             
            next_state = 11
             
            r = -1
             
        else:
             
            next_state = 9
             
            r = -1 
            
            
    elif state == 11:
        
            
        if action == 0:
            
            next_state = 7
             
            r = -1
             
        elif action == 1:
             
            next_state = 0
             
            r = -1
             
        elif action == 2:
             
            next_state = 11
             
            r = -1
             
        else:
             
            next_state = 10
             
            r = -1 
            
            
    elif state == 12:
        
        if action == 0:
            
            next_state = 8
             
            r = -1
             
        elif action == 1:
             
            next_state = 12
             
            r = -1
             
        elif action == 2:
             
            next_state = 13
             
            r = -1
             
        else:
             
            next_state = 12
             
            r = -1 
            
            
    elif state == 13:
        
        if action == 0:
            
            next_state = 9
             
            r = -1
             
        elif action == 1:
            next_state = 13
            r = -1
             
        elif action == 2:
            next_state = 14
            r = -1
             
        else:
            next_state = 12
            r = -1 
            
    else:
        if action == 0:
            next_state = 10
            r = -1
             
        elif action == 1:
            next_state = 14
            r = -1
             
        elif action == 2:
            next_state = 0
            r = -1
             
        else:
            next_state = 13
            r = -1 

    return next_state, r
    

class GridWorldEnv:
    def __init__(self):
        self.size = 14
        self.reset()

    def reset(self):
        self.agent_pos = np.random.randint(1, 15)
        return self.agent_pos

    def step(self, action):
        self.agent_pos, reward = n_state(self.agent_pos, action)
        return self.agent_pos, reward, self.agent_pos in [0, 15]
            
# Q-learning algorithm
def q_learning(env: GridWorldEnv, episodes: int, alpha: float, epsilon: float):
    q_values = np.zeros((16, 4))
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(q_values[state])

            next_state, reward, done = env.step(action)
            total_reward += reward

            #best_next_action = np.argmax(q_values[next_state])
            td_target = reward + np.max(q_values[next_state])# best_next_action]
            td_error = td_target - q_values[state, action]
            q_values[state, action] += alpha * td_error

            state = next_state

        rewards.append(total_reward)

    return q_values, rewards

# SARSA algorithm
def sarsa(env: GridWorldEnv, episodes: int, alpha: float, epsilon: float) -> List[float]:
    q_values = np.zeros((16, 4))
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(q_values[state])

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward

            if np.random.rand() < epsilon:
                next_action = np.random.choice(4)
            else:
                next_action = np.argmax(q_values[next_state])

            td_target = reward + q_values[next_state, next_action]
            td_error = td_target - q_values[state, action]
            q_values[state, action] += alpha * td_error

            state = next_state
            action = next_action

        rewards.append(total_reward)

    return q_values, rewards

if __name__ == "__main__":
    env = GridWorldEnv()

    episodes = 10_000 # Number of episodes 
    n_states = 16 # Number of states: 14 nomral states and two terminal states (represented by state 0 & 15)
    n_actions = 4 # Number of total actions
    epsilon = 0.05
    alpha = 0.01 # learning rate
    h = np.zeros(episodes)
    action_state = np.zeros(n_states)

    q_values_from_q_learning, q_learning_rewards = q_learning(env, episodes, alpha, epsilon)
    q_values_from_sarsa, sarsa_rewards = sarsa(env, episodes, alpha, epsilon)
    import matplotlib.pyplot as plt 

    plt.plot(q_learning_rewards, label='Q-learning')
    plt.plot(sarsa_rewards, label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards')
    plt.legend()
    plt.show()


