import numpy as np 
from dataclasses import dataclass, field
from numpy.typing import ArrayLike
from classes.data_types import Matrix

# General hyper parameters for networks
M = 10
INPUT_SHAPE = (2, M)

# hyper parameters for replay buffer
MEM_SIZE = 10

class ReplayBuffer(): 
    """Used to store """
    mem_counter: int = 0
    state: ArrayLike = np.zeros((MEM_SIZE, *INPUT_SHAPE))
    new_states: ArrayLike = np.zeros((MEM_SIZE, *INPUT_SHAPE))
    actions = np.zeros((MEM_SIZE, 1))
    rewards = np.zeros(MEM_SIZE)
    terminals = np.zeros(MEM_SIZE, dtype=np.bool)

    def store_transition(self, state1: Matrix, action: , reward, state2, terminated: bool)
    

    