# Introduction to Networks/Graphs

## 1. Graph Definitions

A **network** (or **graph**) $G$ is a set of **nodes** (or **vertices**) $V$ connected by a set of **links** (or **edges**) $E$. A set $\{i,j\} \in E$ if the edge is undirected of a tuple $(i,j)\in E$ if the edge is directed from $i$ to $j$ . The network is denoted by $G = (V, E)$. 

Two common representations of a network

* adjacency list

* adjacency matrix:
  $$
  \begin{aligned}
  A_{ij} &= 1 \quad \text { if there exists an edge from } i \text { to } j\\
  A_{ij} &= 0 \quad \text { otherwise}
  \end{aligned}
  $$
  For an undirected graph, adjacency matrix is symmetric; For a directed graph, $A_{ij}$ and $A_{ji}$ are independent.

Quantitative measures to make sense of a network

* connected components 
* edge density
* degree distribution
* diameter and average path length
* clustering
* homophily or assortative mixing

Different parts of graphs

* A (directed) **walk** in a graph is a sequence of (directed) edges such that every pair of edges shares a node, and every pair of nodes are connected by the (directed) edge.
* A (directed) **trail** is a walk where every edge in the sequence is unique.
* A (directed) **path** is a trail where every node in the node sequence is unique.
* A (directed) **cycle** is a (directed) trail that starts and terminates at the same node and such that all other nodes in the node sequence are unique.

Different types of graphs

* **Simple network**: Undirected network with at most one edge between any pair of vertices, and no self-loops.
* **Multigraph**: May contain self-loops or multiple links between vertices.
* **Weighted network**: Edges have weights or vertices have attributes.
* **Tree**: A graph with no cycles.
* **Acyclic network**: Graph with no cycles.
* **Bipartite**: Vertices can be divided into two classes where there are no edges between vertices in the same class (but there can exist edges between vertices in different classes).
* **Hypergraph**: Generalized edges which connect more than two vertices together.

> #### Exercise 25
>
> True or False: $A = A^T$ if and only if the graph is undirected
>
> > **Answer**: False
>
> > **Solution**: 
> >
> > While an undirected graph has the property $A = A^T$, it is not necessary that the all the graphs that hold $A = A^T$ are undirected.
> >
> > Exception 1: 
> >
> > Th directed graph $G=(V=\{ 1,2\} ,E=\{ (1,2),(2,1)\} )$ has adjacency matrix as follows
> > $$
> > \begin{pmatrix}  0& 1\\ 1& 0 \end{pmatrix}
> > $$
> > But $G$ is a cycle with two edges.
> >
> > Exception 2:
> >
> > The direct graph $G'=(V=\{ 1,2\} ,E=\{ \{ 1,2\} \} )$ has the same adjacency matrix. It has no cycle (as the definition of a cycle does not allow repeated edges) and only one edge.

#### Powers of the Adjacency Matrix

For an adjacency matrix $A$, for $l \geq 1$, $A^l$ contains the following elements: $A_{ij}^l$, which is the element in row $i$ and column $j$ of $A^l$, is the number of walks of length $l$ from node $i$ to node $j$.

To inductively prove this, we assume $A^l_{ij}$ equals to the number of walks with length $l$ from node $k$ to node $j$. Then
$$
A_{ij}^{l+1} = [AA^l]_{ij} = \sum_k A_{ik} A_{kj}^l
$$
We see that $A_{ik}A^\ell _{kj}$ will be zero if there is no walk from $i$ to $k$, and it will be equal to $A_{kj}^l$ if there is. Thus $A_{ij}^{l+1}$ equal to the number of walks of length $l+1$ from node $i$ to $j$ and the proof is completed.

## 2. Graph Properties and Metrics - I