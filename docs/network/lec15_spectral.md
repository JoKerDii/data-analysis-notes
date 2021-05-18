# Spectral Clustering

There are 4 topics.

## 1. Co-offending networks

**Co-offending networks**: Nodes are the offenders, and two offenders share a (possibly weighted) edge whenever they are arrested from the same crime event.

* the co-offending network has (weighted) adjacency matrix $AA^T$ and also $A^TA$.

* we can also analyze the bipartite network given by the adjacency matrix
  $$
  \begin{bmatrix} 0 & A \\ A^T & 0\end{bmatrix}
  $$

**Caviar network**: Drug trafficking network investigated over time. New criminals were added to the network by wire-tapping phones. Unique opportunity to analyze how a network reorganizes itself when subjected to stress.

Related scenario:

* Given a social network and $k$ criminal suspects, how to determine other suspects?
* Same question is extremely important in biology: given certain genes that are known to cause a certain disease, determine other candidate genes (e.g. based on protein-protein interaction network for determining autism genes http://dx.doi.org/10.1101/057828)
* How do we identify nodes that are "between" a given set of seed nodes?

## 2. Steiner trees

Determine a smallest subnetwork that contains the given suspects / genes and connects these nodes. When we define the **"smallest" to the sum of all edge weights in the sub-network**, then the problem becomes Steiner tree problem in graphs.

**Steiner tree**: 

* shortest  subnetwork that contains a given set of nodes
* NP-complete problem
* polynomial time approximations

Use collection of **approximate** Steiner trees for further analysis: **autism interactome / criminal interactome**.

* Is interactome indeed more tightly connected than at random?
  * Assume interactome was built with $k$ seed nodes, choose $k$ nodes at random and compute resulting interactome, perform **hypothesis test** based on diameter / average geodesic
  * Compute nodes with **high betweenness centrality** in interactome to obtain candidate genes / suspects.

Genomics application: http://fraenkel-nsf.csbi.mit.edu/steinernet/tutorial.html

## 3. A Few Practical Datasets

**Community detection**:

* detect subsets of nodes that are more densely connected between each other in the network than outside the community.

**Clustering**:

* determine subsets of points that are 'close' to each other given a pairwise distance or similarity measure.
* can be used also for community detection by defining a vertex similarity measure (e.g., geodesic distance, number of different neighbors, correlation between adjacency matrix columns, etc.)
* examples: hierarchical clustering, k-means

**Divisive algorithm using betweenness**:

* intercommunity edges have a large value of edge betweenness, because many shortest paths connecting vertices of different communities will pass through them.
* can dene betweenness using geodesic, flow or random walk.

**Modularity maximization**:

* **quality function**: function that assigns a number (quality measure) to each partition of a graph.

* most popular quality function: **modularity**
  $$
  Q = \frac{1}{2m} \sum_{i,j} (A_{ij} - P_{ij}) \delta (C_i, C_j),
  $$
  where $P_{ij}$ is expected number of edges between $i$ and $j$ in null model.

  * compares actual edge density to expected edge density in null model. It tells you "*how much more is in the given network are the communities connected to each other than what you would expect just by chance?*"
  * for Erdos-Renyi model $P_{ij} = \frac{2m}{n(n-1)}$
  * for configuration model $P_{ij} = \frac{k_i k_j}{2m-1}$
  * for bipartite graphs: $Q=\frac{1}{2m} \sum_i \sum_j (A_{ij} - \frac{k_i^{(1)} k_j^{(2)}}{2m}) \delta(C_i, C_j)$

* To find a community is to find a configuration or a partition of the graph such that the modularity score is as large as possible

**Louvain Method**: 

* Modularity optimization is NP-complete. 
* Louvain method is very very fast heuristic $O(m)$.
  * put each vertex in its own community
  * put vertex $i$ into community $j$ that yields biggest increase in modularity
  * replace communities by super-vertices, where edge weight between super-vertices is sum of edge weights between corresponding nodes 
  * iterate process until $Q$ cannot be improved.

## 4. Clustering

**Graph Laplacian matrix** of a graph $L$ is defined as 
$$
L = D - A
$$
where $A$ is the adjacency matrix and $D$ is the degree matrix. The **degree matrix** for an undirected, unweighted graph is a matrix whose off-diagonal elements are equal to 0 and whose diagonal elements are given by
$$
D_{ij} = \sum_j A_{ij}
$$
In other words, the degree matrix of an undirected, unweighted, simple graph is simply a diagonal matrix whose diagonal entries are the degrees of the nodes. In the case of a weighted, undirected, simple graph the definition is the same but the interpretation no longer concerns the degree of the nodes, but rather the sum weight of the edges emanating from the nodes.

The adjacency matrix and the degree matrix are both symmetric – hence the Laplacian is also **symmetric**.

