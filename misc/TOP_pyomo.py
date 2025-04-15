# %%
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:04:42 2023

@author: FJ03XZ
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:58:09 2023

@author: FJ03XZ
"""

### Pyomo implementation of the orienteering problem (based on the traveling slaesman problem template by Inkyung)

from pyomo.environ import *
import networkx as nx
# import matplotlib.pyplot as plt
import math
# import random
import numpy as np
import csv
import matplotlib.pyplot as plt


# Import maps
with open("p7.2smalll.csv", "r") as f:
    csvreader = csv.reader(f)
    rows = []
    for row in csvreader:
        rows.append(row)
    rows
    column_names = rows.pop(0)
    i = 0
    for row in rows:
        rows[i] = [eval(j) for j in row]
        i += 1

T = 50
G = nx.Graph()
for node in rows:
    G.add_node(node[0]-1,pos=(node[1],node[2]), score = node[3])

# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")
print(pos)

dists = np.zeros( (len(rows), len(rows)) )
for i in range(len(rows)):
    for j in range(len(rows)):
          dist = math.hypot(float(rows[i][1]) - float(rows[j][1]), float(rows[i][2]) - float(rows[j][2])) # Euclidean norm  = sqrt(sum(x**2 for x in coordinates))
          if i != j:
              G.add_edge(i, j, weight=dist)
              dists[i,j] = dist

nr_agents = 2

# T = 5
# n = 10
# # G = nx.random_geometric_graph(10, radius=0.4, seed=3, )
# random.seed(100)
# np.random.seed(100)
# G = nx.complete_graph(n)
# # G = nx.random_geometric_graph(200, 0.125, seed=896803) # For a large-scale network

# # position is stored as node attribute data for random_geometric_graph
# # pos = nx.get_node_attributes(G, "pos")
# pos = nx.spring_layout(G)

# # nx.draw_networkx_edges(G, pos, edge_color="blue", width=0.5)

# # Distance (weight) attribute creation
# for i in range(len(pos)):
#     for j in range(len(pos)):
#           dist = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) # Euclidean norm  = sqrt(sum(x**2 for x in coordinates))
#           if i!=j:
#               G.add_edge(i, j, weight=dist)

m = ConcreteModel()
n = len(rows)
m.x = Var(range(n), range(n), range(nr_agents), within = Binary)
m.y = Var(range(n), range(nr_agents), within = Binary)
m.u = Var(range(n), range(nr_agents), within=NonNegativeIntegers,bounds=(0,n-1))

# Add scores to each node
# for i in N:
#     G.add_node(node_for_adding = i, score = 10)
    
score = nx.get_node_attributes(G, "score")

def obj_rule(m):
   return sum(m.y[i,k] * score[i] for k in range(nr_agents) for i in range(1, n - 1))
  # return sum([m.y[i,0] * score[i] for i in V])

m.obj = Objective(rule = obj_rule, sense = maximize)

# Start and end point
def out_going(m,k):
    return sum(m.x[0,j,k] for j in range(1, n)) == 1

m.out_going = Constraint(range(nr_agents), rule=out_going)

def incoming(m,k):
    return sum(m.x[i,n - 1,k] for i in range(0, n - 1)) == 1

m.incoming = Constraint(range(nr_agents), rule=incoming)

def flow(m,v,k):
    return sum(m.x[i,v,k] for i in range(0, n - 1)) == m.y[v,k]

m.flow = Constraint(range(1, n - 1), range(nr_agents), rule=flow)

def flow2(m,v,k):
    return sum(m.x[v,j,k] for j in range(1, n)) == m.y[v,k]

m.flow2 = Constraint(range(1, n - 1), range(nr_agents), rule=flow2)

# Any node can only belong to one agent (value of k)
def one_agent_per_node(m,i):
    return sum(m.y[i,k] for k in range(nr_agents)) <= 1
    # return m.y[i,0] + m.y[i,1] <= 1

m.one_agent = Constraint(range(1, n - 1), rule = one_agent_per_node)

# Make sure that subtours are not allowed for any of the agents.
def sub_tour_elimination11(m,i,k): 
    return 1 <= m.u[i,k]

def sub_tour_elimination12(m,i,k):
    return m.u[i,k] <= n

# NOTE: virker mærkeligt med index??
m.sub_tour11 = Constraint(range(1, n), range(nr_agents), rule=sub_tour_elimination11)

m.sub_tour12 = Constraint(range(1, n), range(nr_agents), rule=sub_tour_elimination12)

def sub_tour_elimination2(m,i,j,k):
    return m.u[i,k] - m.u[j,k] + 1 <= (n-1)*(1-m.x[i,j,k])

# There might be an issue with how it select where U and K go in the different subscripts (maybe U,K,U works better?)
m.sub_tour2 = Constraint(range(1, n - 1), range(1, n - 1), range(nr_agents), rule=sub_tour_elimination2)

# Add constraint for time (distance) budget
def range_constraint(m,k):
    # return sum(m.x[i,j,k] * G[i][j]["weight"] for j in U for i in M if i!=j) <= T
    return sum(m.x[i,j,k] * dists[i][j] for i in range(0, n - 1) for j in range(1, n) if i!=j) <= T

m.rangeconst = Constraint(range(nr_agents), rule = range_constraint)

print("starting solver")

solver = SolverFactory('glpk', executable='/bin/glpsol')
solver.options['tmlim'] = 300
solver.solve(m).write()

l = list(m.x.keys())
used_edges = []
for i in l:
    if m.x[i]() != 0 and m.x[i]() != None:
        #print(i,'--', m.x[i]())
        used_edges.append(i)

edges_used_by_agent = {}
for k in range(nr_agents):
    edges_used_by_agent[k] = []
    for edge in used_edges:
        if edge[-1] == k:
            edges_used_by_agent[k].append(edge)

    print(f"Agent {k} traversed a total distance of {sum(dists[edge[0], edge[1]] for edge in edges_used_by_agent[k]):.1f} via the edges: {edges_used_by_agent[k]}")

# Repeat for each agent
nx.draw_networkx(
    G,
    pos,
    with_labels=False,
    edgelist=edges_used_by_agent[0],
    edge_color="red",
    node_size=50,
    width=3,
)
nx.draw_networkx(
    G,
    pos,
    with_labels=False,
    edgelist=edges_used_by_agent[1],
    edge_color="blue",
    node_size=50,
    width=3,
)
plt.show()

### Results
# It seems like there is a constraint that prevents it from visiting all nodes (even when there is enough range)







