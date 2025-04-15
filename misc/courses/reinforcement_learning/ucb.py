# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:05:26 2025

@author: BR11WP
"""

#Bandit problem 
import math
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
# %matplotlib inline

#Generate Data
user = list(range(0,10000))
m1 = np.random.normal(100,10,10000)
m2 = np.random.normal(105,5,10000)
m3 = np.random.normal(95,10,10000)
m4 = np.random.normal(100,5,10000)

df = pd.DataFrame({"user":user, "m1":m1,"m2":m2,"m3":m3,"m4":m4})
#df.head()

# Box plot

#sns.boxplot(x="variable", y="value", data=pd.melt(df[['m1','m2','m3','m4']]))
#plt.title("Distribution of Rewards by Message")

# Variable initialization

#Initialize Variables
N = len(df.index)       # the time (or round)
d = 4                   # number of possible messages
Qt_a = 0                # NOTE: Gets recomputed each time.
Nt_a = np.zeros(d)      #number of times action a has been selected prior to T
                        #If Nt(a) = 0, then a is considered to be a maximizing action.
c = 1                   #a number greater than 0 that controls the degree of exploration

sum_rewards = np.zeros(d) #cumulative sum of reward for a particular message

#helper variables to perform analysis
hist_t = [] #holds the natural log of each round
hist_achieved_rewards = [] #holds the history of the UCB CHOSEN cumulative rewards
hist_best_possible_rewards = [] #holds the history of OPTIMAL cumulative rewards
hist_random_choice_rewards = [] #holds the history of RANDONMLY selected actions rewards
###


# Action Selection

#loop through no of rounds #t = time
for t in range(0,10_000): # Loop through the customers 

    UCB_Values = np.zeros(d) # Reset the array holding the ucb values. we pick the max
    for a in range(0, d):
        if (Nt_a[a] > 0):
            ln_t = math.log(t) #natural log of t
            hist_t.append(ln_t) #to plot natural log of t

            #calculate the UCB
            Qt_a = sum_rewards[a]/Nt_a[a]
            ucb_value = Qt_a + c*(ln_t/Nt_a[a])
            UCB_Values[a] = ucb_value

        #if this equals zero, choose as the maximum. Cant divide by 0
        elif (Nt_a[a] == 0):
            UCB_Values[a] = 1e500 #make large value

    #select the max UCB value
    action_selected = np.argmax(UCB_Values)

    #update Values as of round t
    Nt_a[action_selected] += 1
    reward = df.values[t, action_selected+1]
    sum_rewards[action_selected] += reward

        #these are to allow us to perform analysis of our algorithmm
    r_ = df.values[t,[1,2,3,4]]     #get all rewards for time t to a vector
    r_best = r_[np.argmax(r_)]      #select the best action

    pick_random = random.randrange(d) #choose an action randomly
    r_random = r_[pick_random] #np.random.choice(r_) #select reward for random action

    if len(hist_achieved_rewards)>0:
        hist_achieved_rewards.append(hist_achieved_rewards[-1]+reward)
        hist_best_possible_rewards.append(hist_best_possible_rewards[-1]+r_best)
        hist_random_choice_rewards.append(hist_random_choice_rewards[-1]+r_random)

    else:
        hist_achieved_rewards.append(reward)
        hist_best_possible_rewards.append(r_best)
        hist_random_choice_rewards.append(r_random)

print("Reward if we choose randonmly {0}".format(hist_random_choice_rewards[-1]))
print("Reward of our UCB method {0}".format(hist_achieved_rewards[-1]))

#Reward if we choose randonmly 1000464.1984600379
#Reward of our UCB method 1048880.064875564

plt.bar(['m1','m2','m3','m4'],Nt_a)
plt.title("Number of times each Message was Selected")
plt.show()

def simple_bandit_algorithm(epsilon: float):
    hist_t = [] #holds the natural log of each round
    hist_achieved_rewards = [] #holds the history of the UCB CHOSEN cumulative rewards
    hist_best_possible_rewards = [] #holds the history of OPTIMAL cumulative rewards
    hist_random_choice_rewards = [] #holds the history of RANDONMLY selected actions rewards
    Qs = np.zeros(4)
    Ns = np.zeros(4)
    
    for t in range(0,10_000): # Loop through the customers 
        if np.random.uniform(0, 1) > epsilon:
            action_selected = np.argmax(Qs)
        else:
            action_selected = np.random.randint(0, 4)

        #update Values as of round t
        reward = df.values[t, action_selected+1]
        sum_rewards[action_selected] += reward

        Ns[action_selected] += 1
        Qs[action_selected] += 1 / Ns[action_selected] * (reward - Qs[action_selected])

        #these are to allow us to perform analysis of our algorithmm
        r_ = df.values[t,[1,2,3,4]]     #get all rewards for time t to a vector
        r_best = r_[np.argmax(r_)]      #select the best action

        pick_random = random.randrange(d) #choose an action randomly
        r_random = r_[pick_random] #np.random.choice(r_) #select reward for random action

        if len(hist_achieved_rewards)>0:
            hist_achieved_rewards.append(hist_achieved_rewards[-1]+reward)
            hist_best_possible_rewards.append(hist_best_possible_rewards[-1]+r_best)
            hist_random_choice_rewards.append(hist_random_choice_rewards[-1]+r_random)

        else:
            hist_achieved_rewards.append(reward)
            hist_best_possible_rewards.append(r_best)
            hist_random_choice_rewards.append(r_random) 

    print("Reward if we choose randonmly {0}".format(hist_random_choice_rewards[-1]))
    print("Reward of our simple epsilon greedy {0}".format(hist_achieved_rewards[-1]))
    
    #Reward if we choose randonmly 1000464.1984600379
    #Reward of our UCB method 1048880.064875564
    
    plt.bar(['m1','m2','m3','m4'],Nt_a)
    plt.title("Number of times each Message was Selected")
    plt.show()

if __name__ == "__main__":
    simple_bandit_algorithm(0.3)
