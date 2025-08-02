import numpy as np
import random
from numpy.typing import ArrayLike
from itertools import count

connected_states = {0: [0],
                    1: [0, 2, 5],
                    2: [1, 3, 6],
                    3: [2, 7],
                    4: [0, 5, 8],
                    5: [1, 4, 6, 9],
                    6: [2, 5, 7, 10],
                    7: [3, 6, 11],
                    8: [4, 9, 12],
                    9: [5, 8, 10, 13],
                    10: [6, 9, 11, 14],
                    11: [7, 10, 15],
                    12: [8, 13],
                    13: [9, 12, 14],
                    14: [10, 13, 15],
                    15: [15]}

def policy_iteration(gamma: float, theta: float) -> ArrayLike:
    """Performs policy iterations on exercise """
    values = np.zeros(16)
    policy = np.array([random.choice(connected_states[s]) for s in range(16)])
    
    policy_stable = False

    while not policy_stable:
        policy_stable = True 
        # Policy evaluation
        delta = 0 
        while delta < theta:
            delta = 0 
            for s in range(16):
                v = values[s]
                values[s] = (0 if policy[s] in [0, 15] else -1) + gamma * values[policy[s]] # sum([ for ])
                delta = max(delta, abs(v - values[s]))

        # Policy Improvement
        policy_stable = True
        for s in range(16):
            old_action = policy[s]
            policy[s] = max([(state, values[state]) for state in connected_states[s]], key=lambda tup: tup[1])[0]
            if old_action != policy[s]:
                policy_stable = False

    return policy

def random_policy_value_estimation() -> ArrayLike:
    """Random policy value estimation using monte carlo methods."""
    values = np.zeros(16)
    returns = [[] for _ in range(16)]

    for episode in range(10_000):
        # Generate episode
        s = random.choice(list(connected_states.keys())) # Pick a random initial state

        first_visits_indices = dict()
        for i in count():
            if s in [0, 15]:
                first_visits_indices[s] = i
                break

            first_visits_indices[s] = first_visits_indices.get(s, i) 

            s = random.choice(connected_states[s])

        # Update values
        for s, j in first_visits_indices.items():
            returns[s].append(-(i - j))
    
    return np.array([np.mean(returns[s]) for s in range(16)]).reshape((4, 4))
    
if __name__ == "__main__":
    values = random_policy_value_estimation()
    print(values)
    #policy = policy_iteration(0.99, 0.01)
    #import pprint  
    #pprint.pprint({i: int(policy[i]) for i in range(16)})