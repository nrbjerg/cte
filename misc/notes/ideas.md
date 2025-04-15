# Genetic algorithm
Can the entropy of a population in a GA be used to adabt the mutation rates throughout exectution?

Alternatively is it posible to construct a neural network which addapts the hyperparameters on the fly.

# Incorporating TSP into memetic algorithm for TOP / OP
maybe the TSP could be used to shorten the routes found within a TOP allowing for further insertions of visits ect. Small TSPlib instances n <= 1000 can be solved in about a second using CPLEX, but maybe this will simply be similar to using a Lin-Kernighan or k-opt, however maybe the Held-Karp lower bound (which is obtained by solving the LP relaxation of the integer programming problem of TSP) can be used as an indicator for when it is worth it to use a solver? HK is on average about 0.8% below the optimal lower bound 

# Sweeping:
Bentley-Ottmann sweepline algorithm for decomposing self-intersecting polygons into a set of simple polygons