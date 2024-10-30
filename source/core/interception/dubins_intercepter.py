#!/usr/bin/env python3
from classes.data_types import Velocity, Position, State
from dataclasses import dataclass
import numpy as np
from typing import Callable, List
#from enum import Enum
from scipy.optimize import bisect

def modified_bisection_algorithm(f: Callable[[float], float], a: float, b: float) -> List[float]:
   """Computes all of the zeros of f within the interval [a, b]."""
   # TODO: Compute the zeros of the f'
   ts = [a] + [] + [b]
   
   zeros = []
   for t0, t1 in zip(ts[:-1], ts[1:]):
      if f(t0) == 0:
         zeros.append(t0)
      elif np.sign(f(t0)) != np.sign(f(t1)):
         zeros.append(bisect(f, t0, t1))

   return zeros


