import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
N = 9 
length = lambda r: np.linalg.norm(r[0] - r[-1]) + sum(np.linalg.norm(frm - to) for frm, to in zip(r[1:], r[:-1]))
points = np.random.uniform(0, 1, size = (N, 2))
best_route = np.array(min(permutations(points, N), key=length))
plt.plot(list(best_route[:, 0]) + [best_route[0][0]], list(best_route[:, 1]) + [best_route[0][1]], marker = "o")
plt.show()