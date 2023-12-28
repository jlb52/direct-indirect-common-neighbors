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
Number of nodes = 2708, Number of edges = 5278

| Edge Split | AUC Mean - DICN | AUC Mean - Resource Allocation | AUC Mean - Jaccard | AUC Mean - Adamic Adar | AUC Mean - Preferential Attachment | AUC Mean - Common Neighbor Centrality | Time Mean (s) - DICN | Time Mean (s) - Resource Allocation | Time Mean (s) - Jaccard | Time Mean (s) - Adamic Adar | Time Mean (s) - Preferential Attachment | Time Mean (s) - Common Neighbor Centrality |
|------:|---------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|---------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|
|   50% | 68.4%                | 59.8%                             | 59.7%                 | 60.0%                     | 60.3%                                     | 67.7%                                       | 0.74                | 0.01                               | 0.02                   | 0.01                       | 0                                         | 0.76                                        |
|   60% | 74.4%                | 63.6%                             | 62.9%                 | 62.9%                     | 62.0%                                     | 71.9%                                       | 0.72                | 0.01                               | 0.02                   | 0.01                       | 0                                         | 0.97                                        |
|   70% | 79.8%                | 66.6%                             | 66.6%                 | 66.8%                     | 62.1%                                     | 76.1%                                       | 0.68                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 1.19                                        |
|   80% | 83.6%                | 70.6%                             | 69.7%                 | 70.1%                     | 64.0%                                     | 79.0%                                       | 0.67                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 1.31                                        |
|   90% | 87.1%                | 73.4%                             | 73.0%                 | 73.2%                     | 64.7%                                     | 83.4%                                       | 0.63                | 0                                  | 0.01                   | 0                           | 0                                         | 1.46                                        |


### CiteSeer
Number of nodes = 3327, Number of edges = 4552

| Edge Split | AUC Mean - DICN | AUC Mean - Resource Allocation | AUC Mean - Jaccard | AUC Mean - Adamic Adar | AUC Mean - Preferential Attachment | AUC Mean - Common Neighbor Centrality | Time Mean (s) - DICN | Time Mean (s) - Resource Allocation | Time Mean (s) - Jaccard | Time Mean (s) - Adamic Adar | Time Mean (s) - Preferential Attachment | Time Mean (s) - Common Neighbor Centrality |
|-----------:|--------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|---------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|
|       50%  | 64.1%               | 57.7%                              | 57.6%                  | 57.8%                      | 59.4%                                     | 67.0%                                       | 1.19                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 0.22                                        |
|       60%  | 70.1%               | 60.5%                              | 60.2%                  | 60.3%                      | 59.4%                                     | 69.9%                                       | 1.13                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 0.4                                         |
|       70%  | 76.0%               | 62.9%                              | 62.7%                  | 62.6%                      | 58.4%                                     | 71.8%                                       | 0.94                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 0.58                                        |
|       80%  | 81.6%               | 65.6%                              | 65.2%                  | 65.3%                      | 58.3%                                     | 74.7%                                       | 0.98                | 0.01                               | 0.01                   | 0.01                       | 0                                         | 0.76                                        |
|       90%  | 85.8%               | 67.9%                              | 67.8%                  | 67.9%                      | 58.0%                                     | 75.5%                                       | 1.04                | 0                                  | 0                      | 0                           | 0                                         | 0.94                                        |


### PubMed
Number of nodes = 19717, Number of edges = 44324

| Edge Split | AUC Mean - DICN | AUC Mean - Resource Allocation | AUC Mean - Jaccard | AUC Mean - Adamic Adar | AUC Mean - Preferential Attachment | AUC Mean - Common Neighbor Centrality | Time Mean (s) - DICN | Time Mean (s) - Resource Allocation | Time Mean (s) - Jaccard | Time Mean (s) - Adamic Adar | Time Mean (s) - Preferential Attachment | Time Mean (s) - Common Neighbor Centrality |
|-----------:|--------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|---------------------:|-----------------------------------:|-----------------------:|---------------------------:|-----------------------------------------:|-------------------------------------------:|
|       50%  | 61.5%               | 57.5%                              | 57.4%                  | 57.3%                      | 71.3%                                     | 72.4%                                       | 117.16               | 0.16                               | 0.21                   | 0.16                       | 0.03                                      | 49.64                                       |
|       60%  | 69.2%               | 59.6%                              | 59.5%                  | 59.5%                      | 71.5%                                     | 73.6%                                       | 117.86               | 0.15                               | 0.19                   | 0.14                       | 0.03                                      | 62.22                                       |
|       70%  | 76.4%               | 61.7%                              | 61.5%                  | 61.5%                      | 71.6%                                     | 74.2%                                       | 116.61               | 0.12                               | 0.16                   | 0.12                       | 0.02                                      | 74.25                                       |
|       80%  | 83.0%               | 63.5%                              | 63.5%                  | 63.6%                      | 71.4%                                     | 74.9%                                       | 117.86               | 0.09                               | 0.11                   | 0.09                       | 0.02                                      | 86.48                                       |
|       90%  | 88.6%               | 65.0%                              | 65.4%                  | 65.3%                      | 72.1%                                     | 75.4%                                       | 117.73               | 0.05                               | 0.06                   | 0.05                       | 0.01                                      | 97.88                                       |

As can be seen, AUC tends to be very competitve (against other NetworkX link prediction algorithms) while time efficiency sometimes suffers, especially on larger graphs. 

## Getting Started
### Prerequisites
- Python 3.9 or higher
- NetworkX 3.1 or higher
- Numpy 1.1 or higher

### Installation
```bash
pip install direct-indirect-common-neighbors
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