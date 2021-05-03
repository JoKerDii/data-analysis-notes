# Introduction to Networks/Graphs

There are 3 topics and 4 exercises.

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

Quantitative measures to make sense of a network

* connected components 

  **Connected components** are sets of nodes that are reachable from one another. Many networks consist of one large component and many small ones.

* edge density

  The **edge density** or **connectence** is defined as
  $$
  \rho = \frac{m}{\begin{pmatrix} n\\2\end{pmatrix}} = \frac{\sum_{i,j}A_{ij}}{n(n-1)}, \quad \text{where |V| = n, |E|=m}
  $$

  * Most networks are **sparse**, i.e. $\rho \xrightarrow{n \rightarrow \infty} 0$ (the number of edges goes not grow proportionally with the square of the number of nodes). E.g. social network
  * Some networks are **dense**, i.e. $\rho \xrightarrow{n \rightarrow \infty} $ const. E.g. food web.

* degree distribution

  The **average degree** is $\frac{1}{n} \sum_i k_i  =\frac{\sum_{i,j}A_{ij}}{n} = \frac{2m}{n}$, where $k_i$ is the degree of node $i$, $|V| = n, |E| = m$.

  **Histogram** of fraction of nodes with degree $k$ reveals more information. (Long tail).

  **Power-law distribution**: $\log p_k = -\alpha \log k +c $ for some  $\alpha, c > 0$. (fat tails i.e. many nodes with high degrees; linear on a log-log plot)

* diameter and average path length

  The **diameter** of a network is the largest length of **geodesic path** $d_{ij}$(or shortest path) between any two nodes $i$ and $j$.
  $$
  \text{diameter }= \max_{i,j \in V}d_{ij}
  $$
  The **average path length** if the average distance between any two nodes 
  $$
  \text{average path length }=\frac{1}{{n \choose 2}} \sum _{i \le j} d_{ij}
  $$
  If network is not connected, one often computes the diameter and the average path length in the largest component.

  Algorithms for finding shortest paths: **Breadth-First Search (BFS)** for unweighted graph, **Dijkstra's Algorithm** for weighted graphs.

* clustering

* homophily or assortative mixing

> #### Exercise 26
>
> Consider a simple, unweighted, undirected, connected tree with $n$ nodes. Let the degree of every node of the graph be at most 2. Compute the average path length of the graph assuming that $n \geq 3$. 
>
> > **Answer**: $(1/3) * n + 1/3$
>
> > **Solution**: The graph is connected and every node has at most a degree of $2$. This is the **line graph** – a graph where all nodes form a straight, connected line with the two nodes on the edges having a degree of $1$ and the $n−2$ interior nodes having a degree of $2$. 
> >
> > 1. The two nodes on the edges (a single pair of nodes) have a path length of $n−1$.
> > 2. There are two pairs of nodes that have a path length of $n−2$.
> > 3. There are three pairs of nodes that have a path length of $n−3$.
> > 4. So on...
> >
> > Therefore, the average path length is
> > $$
> > \begin{aligned}
> > \text{ average path length } &= \frac{1}{{n \choose 2}} \sum _{i \le j} d_{ij}\\
> > &=\frac{1}{{n \choose 2}} \left[(n-1) + 2(n-2) + 3(n-3) + \dots + (n-1)(n-(n-1)) \right]\\
> > &=\frac{1}{{n \choose 2}} \left[n(1+2+\dots (n-1)) - (1+4+9+\dots +(n-1)^2) \right]\\
> > &=\frac{1}{{n \choose 2}} \left[\frac{n (n-1) n}{2} - \frac{n (n-1) (2(n-1) + 1)}{6} \right]\\
> > &=n - \frac{2}{3} \cdot n + \frac{1}{3}\\
> > &= \frac{n + 1}{3} \quad \text{valid when }n \geq 3
> > \end{aligned}
> > $$
> > When $n = 2$, we only have one term overall and the average path length is equal to $1$.

> #### Exercise 27
>
> Consider the sequence of **star graphs**: there is a central node and every other node has only one neighbor and it is the central node. Compute the limit of the average shortest path length of the sequence of star graphs as $n→∞$.
>
> > **Answer**: 2
>
> > **Solution**: First, for a given $n$, there are $n−1$ path lengths of length $1$ and $(n-1) \choose 2$ path lengths of length $2$ (these are all the path lengths and are also the shortest path lengths). Hence
> > $$
> > \frac{1}{{n \choose 2}} \sum _{i \le j} d_{ij} = \frac{2}{n(n-1)} \left[n-1 + 2 \times \frac{(n-1)(n-2)}{2}\right] =  \frac{2 (n-1)}{n}
> > $$
> > Taking the limit and we have average shortest path length of 2. Intuitively, when $n$ is large, only the distance between the outside nodes matter.

## 3. Graph Metrics – A Measure of Clustering and Modularity

* The **triangle density** of a graph is the ratio of number of triangles in the graph to the number of possible triangles:
  $$
  \text {triangle density} \triangleq \frac{\#  \text { of triangles}}{{n \choose 3}}
  $$

	Disadvantages of triangle density:

  * Denominator is much larger since there may be several connected components.
  * Denominator is much larger since in a connected graph there may be three nodes that are not in the same cluster.

* Alternatives: **clustering coefficient**, denoted $C$, which measures the ratio of triangles in the network to the number of connected triples.
  $$
  C = \frac{\#  \text { of closed triplets}}{\#  \text { of closed and open triplets}}= \frac{3 \cdot \#  \text { of triangles}}{\#  \text { of connected triples}} = \frac{\sum _{i,j,k}A_{ij}A_{jk}A_{ki}}{\sum _ i k_ i (k_ i - 1)}
  $$
  The node-wise clustering coefficient is 
  $$
  C = \frac{\#  \text { of triangles at node }i}{\#  \text { of connected triples centered at node }i} = \frac{\sum _{j,k} A_{ij}A_{jk}A_{ki}}{k_ i (k_ i - 1)}
  $$

* 

> #### Exercise 28
>
> A *complete graph* is an undirected graph on n nodes such that every node is connected to every other node. Say you remove an edge from a complete graph on $n$ nodes. What is the new clustering coefficient? Assume that $n≥3$.
>
> > **Answer**: $ \frac{3 \cdot \left({n \choose 3} - (n-2) \right)}{3 \cdot {n \choose 3} - 2 (n-2)}$.
>
> > **Solution**: In a complete graph we have ${n \choose 3}$ triangles and $3 \cdot {n \choose 3}$ connected triples so that the clustering coefficient is equal to $1$. Now, removing an edge results in the loss of $n-2$ triangles and $2(n-2)$ connected triples. Therefore, teh new clustering coefficient is 
> > $$
> > \frac{3 \cdot \left({n \choose 3} - (n-2) \right)}{3 \cdot {n \choose 3} - 2 (n-2)}
> > $$

* **Homophily** (or **assortative mixing**): measures the fraction of edges in the network that run between nodes of the same type to reflect the tendency of nodes that are associated with others that are similar.

  Problem: value is 1 in a network where all nodes are of same type

  Solution: fraction of edges that run between same type of nodes minus fraction of such edges if edges were placed at random.

  * \# edges of same types = $\sum_{(i,j) \in E} \delta(t_i, t_j) = \frac{1}{2} \sum_{i,j} A_{ij} \delta(t_j, t_j)$, where $t_i$ is type of node $i$ and $\delta(a,b) = 1$ if $a = b$ and 0 otherwise.
  * expected # edges of same type  = $\frac{1}{2} \sum_{ij} \frac{k_ik_j}{2m}\delta(t_i,t_j)$ for all pairs of nodes in a random graph with $m$ edges

  **Modularity** of an undirected graph with node types $t_i, i = 1, ...,n$ is defined as
  $$
  \frac{1}{2m} \sum _{i,j} \left(A_{ij} - \frac{k_ i k_ j}{2m}\right) \delta (t_ i,t_ j) \quad \in [-1,1]
  $$
  ​	where $A$ is the adjacency matrix, $m$ is the number of edges, and $δ(t_i,t_j)=1$ if $t_i=t_j$ and is equal to 0 if $t_i≠t_j$. 

  * For a given pair of nodes $i,j$ a positive value of $\left(A_{ij} - \frac{k_ i k_ j}{2m}\right) \delta (t_ i,t_ j)$ indicates that nodes $i,j$ have an affinity that is more than the expected affinity that they would otherwise have in a truly random graph obtained according to the configuration model with given node types  and $m$ edges.
  * Likewise, a negative value of $\left(A_{ij} - \frac{k_ i k_ j}{2m}\right) \delta (t_ i,t_ j)$ indicates that nodes $i,j$ have lesser than expected affinity when compared to a random graph with the same characteristics.

> Python code for calculating modularity given matrix $A$, and node index with different types
>
> ```python
> def modularity_partition(A, part):
> 	m = A.sum()/2
> 	ks = A.sum(axis=0)
> 	return ( A[part].T[part].T - ks[part][:,None]*ks[part][None,:]/(2*m) ).sum()/(2*m)
> 
> def modularity(A, parts):
> 	return sum([ modularity_partition(A, p) for p in parts ])
> 
> modularity(mat, [[0,2,4,6,8], [1,3,5,7,9]])
> ```

