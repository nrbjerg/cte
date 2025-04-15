# -*- coding: utf-8 -*-
"""
Created on Wed April 2 20:07 2025

@author: Martin

comments: Code was adapted from the file provided by Rahul.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the environment
class CliffWalkingEnv:
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(x - 1, 0)
        elif action == 1:  # right
            y = min(y + 1, self.width - 1)
        elif action == 2:  # down
            x = min(x + 1, self.height - 1)
        elif action == 3:  # left
            y = max(y - 1, 0)

        reward = -1
        if (x, y) in self.cliff:
            reward = -100
            self.agent_pos = self.start
        else:
            self.agent_pos = (x, y)

        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

# Q-learning algorithm
def q_learning(env, episodes, alpha, epsilon):
    q_values = np.zeros(14, 4)
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(q_values[state[0], state[1]])

            next_state, reward, done = env.step(action)
            total_reward += reward

            best_next_action = np.argmax(q_values[next_state[0], next_state[1]])
            td_target = reward + q_values[next_state[0], next_state[1], best_next_action]
            td_error = td_target - q_values[state[0], state[1], action]
            q_values[state[0], state[1], action] += alpha * td_error

            state = next_state

        rewards.append(total_reward)

    return rewards

# SARSA algorithm
def sarsa(env, episodes, alpha, epsilon):
    q_values = np.zeros((env.height, env.width, 4))
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(q_values[state[0], state[1]])

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward

            if np.random.rand() < epsilon:
                next_action = np.random.choice(4)
            else:
                next_action = np.argmax(q_values[next_state[0], next_state[1]])

            td_target = reward + q_values[next_state[0], next_state[1], next_action]
            td_error = td_target - q_values[state[0], state[1], action]
            q_values[state[0], state[1], action] += alpha * td_error

            state = next_state
            action = next_action

        rewards.append(total_reward)

    return rewards

# Compare performance
env = CliffWalkingEnv()
episodes = 1000
alpha = 0.1
epsilon = 0.1

q_learning_rewards = q_learning(env, episodes, alpha, epsilon)
sarsa_rewards = sarsa(env, episodes, alpha, epsilon)

plt.plot(q_learning_rewards, label='Q-learning')
plt.plot(sarsa_rewards, label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards')
plt.legend()
plt.show()

