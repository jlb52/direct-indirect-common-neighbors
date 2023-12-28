from dicn import dicn
import networkx as nx

G = nx.erdos_renyi_graph(n=10, p=0.5)
print(f"True Edges: {G.edges()}")

output = dicn(G)
for u, v, score in output:
    print(f"({u}, {v}): {score}")