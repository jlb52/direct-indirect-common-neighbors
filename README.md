# Direct-Indirect Common Neighbors: Python Implementation

I'm sharing a Python implementation of the link prediction algorithm from "Similarity-based link prediction in social networks using latent relationships between the users" by Ahmad Zareie & Rizos Sakellariou. I coded this as a small part of a school project and figured I would share since I was unable to find an existing implemention online and since the results were relatively strong. 

## About the Algorithm
This algorithm goes beyond traditional first-order neighbor measures and digs deeper into the structural similarities by considering latent relationships, thanks to the clever use of second-order neighbors. 

In essence, it:
1. **Builds Neighborhood Vectors**: Each node gets its own vector, showcasing its immediate and extended neighborhood.
2. **Finds Union Neighborhood Sets**: For every pair of nodes, it considers their overlapping neighborhoods.
3. **Measures Similarity**: Using the Pearson correlation coefficient, the algorithm examines how similar the neighborhoods of two nodes are.
4. **Calculates DICN Scores**: Direct-Indirect Common Neighbours (DICN) scores are derived, blending direct and indirect connections to predict potential links.

I found this approach compelling because it mirrors how we often form connections in real life - not just with those we know directly, but also through extended networks.

## Performance
This algorithm was tested on citation network datasets Cora, CiteSeer, and PubMed, all pulled from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid). 50%-90% of randomly selected edges were removed, prediction scores were generated for these edges, and then prediction scores were generated for an equal number of randomly selected negative samples. AUC score and time were recorded for each test run, and then averaged across the five runs at each edge removal level and for each algorithm. 

AUC and time (s) for each dataset are displayed below, and an example notebook has been provided for reproducibility.

### Cora
Number of nodes = 2708 | Number of edges = 5278

![Cora Results](/notebook/images/cora.png)

### CiteSeer
Number of nodes = 3327 | Number of edges = 4552

![CiteSeer Results](/notebook/images/citeseer.png)

### PubMed
Number of nodes = 19,717 | Number of edges = 44,324

![PubMed Results](/notebook/images/pubmed.png)

As can be seen, AUC tends to be very competitve (against other NetworkX link prediction algorithms) while time efficiency sometimes suffers, especially on larger graphs. 

## Getting Started
### Prerequisites
- Python 3.9 or higher
- NetworkX 3.1 or higher
- Numpy 1.1 or higher

### Installation
```bash
pip install dicn
```

### Usage

To use the DICN algorithm in your Python code, first install the package and then follow this example:

```python
from dicn import dicn
import networkx as nx

G = nx.erdos_renyi_graph(n=10, p=0.5)
print(f"True Edges: {G.edges()}")

output = dicn(G)
for u, v, score in output:
    print(f"({u}, {v}): {score}")
```

This will output something like the following (note that actual output will vary due to the random nature of the graph generation):

```
True Edges: [(0, 1), (0, 4), (0, 8), (1, 3), (1, 4), ..., (6, 9)]

(0, 2): 2.0355727751898023
(0, 3): 6.561239813665759
(0, 5): 3.8620436566990364
(0, 6): 3.7799442608229374
...
(8, 9): 4.419480670222152
```

## References

```
Zareie, A., Sakellariou, R. (2020). Similarity-based link prediction 
in social networks using latent relationships between the users. 
Scientific Reports, 10, 20137. DOI:10.1038/s41598-020-76799-4
```