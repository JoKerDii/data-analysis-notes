# Graph Centrality Measures

There are 4 topics.

## 1. Centrality Measures

Choice of centrality measure depends on application:

* **Degree centrality**

  * Undirected graphs: the degree $k_i$ of node $i$ is the number of edges connected to $i$, i.e. $k_i = \sum_i A_{ij}$.
  * Directed graphs: the indegree of node $i$ is $k_i^{in} = \sum_jA_{ji}$ and the outdegree is $k_i^{out} = \sum_jA_{ij}$.
  * Intuitively: individuals with more connections have more influence and more access to information. [Measure by number]
  * Drawback: does not capture "cascade of effects": importance better captured by having connections to important nodes.

* **Eigenvector centrality**

  * Give each node a score that is proportional to the sum of the scores of all its neighbors.

  * Steps:

    1. Start with equal centrality: $x_i^{(0)}$ = 1 for all nodes $i = 1, ..., n$.

    2. Update each centrality by the centrality of the neighbors.

    $$
    x_i^{(1)} = \sum^n_{j=1}A_{ij}x_j^{(0)}
    $$

    3. Iterate this process: 

    $$
    x^{(k)} = A^kx^{(0)}
    $$

  * If there exists $m > 0$ such that $A^m > 0$, then one can show that
    $$
    x^{(k)} \xrightarrow{k \rightarrow \infty} \alpha \lambda_{max}^k v
    $$
    where $\lambda_{max}$ is the largest eigenvalue and $v \geq 0$ the corresponding eigenvector, $\alpha$ depends on choice of $x^{(0)}$ - **Perron-Frobenius theorem**.

    The alternative update rule could be $x^{(k)} = \frac{1}{\lambda_{max}} Ax^{(k-1)}$, so that $x^{(k)}$ converges to a constant $x^{(k)} \xrightarrow{k \rightarrow \infty} \alpha v$.

    * Proof: 
      $$
      \begin{aligned}
      x^{(k)} &= A^kx^{(0)}\\
      &= \sum^n_{i=1} \lambda_i^k v_iv_i^T x^{(0)} \\
      &= \lambda_1^k v_1v_1^Tx^{(0)} + \lambda_2^k v_2v_2^Tx^{(0)} + ...\\
      &= \lambda_1^k(\alpha_1v_1 + \frac{\lambda_2^k}{\lambda_1^k}\alpha_1v_2 + ...) \quad \text{ since } v_i^Tx^{(0)} \text{ is a constant.}\\
      &\xrightarrow{k \rightarrow \infty} \lambda_1^k \alpha_1 v_1 \quad \text{ if } \lambda_1 \geq \lambda_2 \geq ...
      \end{aligned}
      $$

  * Interpretation: 

    * The ranking of a particular node $i$ satisfies $v_i = \frac{1}{\lambda_{max}} \sum^n_{j=1} A_{ij} v_j$.
    * The importance of a node depends on the importance of its neighbors and the number of neighbors.
    * Eigenvector corresponding to largest eigenvalue of $A$ provides a ranking of all nodes.
    * In the case of directed graph $G$
      * right eigenvector: $v_i = \frac{1}{\lambda_{max}} \sum^n_{j=1} A_{ij}v_j$ 
        * Importance comes from nodes $i$ pointing to $j$
        * E.g. determining malfunctioning genes
      * left eigenvector: $w_i = \frac{1}{\lambda_{max}} \sum^n_{j=1} w_jA_{ji}$
        * Importance comes from nodes pointing to $i$
        * E.g. ranking websites

* **Closeness centrality**: 

  * The closeness centrality of a node $i$ is the reciprocal of average distance of the node to every other node.
    $$
    C_i = \begin{pmatrix} \frac{1}{n-1} \sum_{j \neq i} d_{ij} \end{pmatrix} ^ {-1}
    $$
    where $d_{ij}$ is the distance between nodes $i$ and $j$. 

  * Closeness centrality $C$ is large if the $\sum d_{ij}$ is small (that node $i$ is relatively close to all the other nodes).  If there is a large distance, small distances become trivial.

  * In disconnected networks: average over nodes in same component as $i$ or use **harmonic centrality** (weight more small distances): 
    $$
    H_i = \frac{1}{n-1} \sum_{j \neq i} \frac{1}{d_{ij}}
    $$

* **Betweenness centrality**： 

  * Measure the extent to which a node lies on paths between other nodes.

  $$
  B_i = \frac{1}{n^2} \sum_{s,t} \frac{n_{st}^i}{g_{st}}
  $$
  * where $n_{st}^i$ is the number of shortest paths between $s$ and $t$ that pass through $i$, and $g_{st}$ is total number of shortest paths between $s$ and $t$.
  
  * Note that this considers both orderings of each pair of nodes, so for undirected graphs, a path counts twice (as it counts both for $n^i_{st}$ and for $n^i_{ts}$).

## 2. Katz Centrality

**Problem**: Eigenvector centrality cannot be applied to any directed network. The issue with eigenvector centrality in the case of **directed, acyclic graphs (DAGs)** is that the adjacency matrix $A$ of a DAG has the property that $A^ℓ$ contains all entries equal to $0$ for some value $ℓ$ (and hence for all values greater than $ℓ$). This leads to an issue in the application of the Perron-Frobenius theorem that we alluded to in the definition of eigenvector centrality – there is no convergence to a non-zero vector in the series of updates starting with an initial centrality vector. In particular, $(x^k)→0$ as $k→∞$.

**Solution**: Given every node some fixed (but small) centrality for free:
$$
x_i^{(k+1)} = \alpha \sum^n_{j=1} A_{ij}x_j^{(k)} + \beta_i
$$
or equivalently,
$$
x^{(k+1)} = \alpha Ax^{(k)} + \beta
$$
If $\alpha$ is chosen in the interval $(0, 1/\lambda_{max}(A))$, then one can show that 
$$
x^{(k)} \xrightarrow{k \rightarrow \infty} v
$$
where $v = (I - \alpha A)^{-1} \beta \geq 0$ ( for example: for DAGs it holds that $\lambda_{max} = 0$, hence no constraints on $\alpha$, take e.g. $\alpha = 1$).

## 3. Page-Rank Centrality

**Drawback** of Katz centrality and eigenvector centrality: They assign a relatively high importance value to a node $i$ that has an incoming edge from a node $j$ that is of high importance and has no other incoming edges. If node $j$ has a very high out-degree then node $i$ is just one of the many neighbors that node $j$ points to. In some applications, we may require that such a node $i$ not have very high importance simply because it has an incoming edge from a node of very high importance.

**Solution**: **Page-Rank centrality** weighs the contributions of all neighbors of a node by their respective out-degree values:
$$
x_j^{(k+1)} = \alpha \sum^n_{i=1} A_{ij} \frac{x_i^{(k)}}{k_i^{out}} + \beta_j
$$
or equivalently, 
$$
x^{(k+1)} = \alpha D^{-1} Ax^{(k)} + \beta, \quad \text{ where } D = \text{diag}(k_1^{out}, ..., k_n^{out})
$$
If $\alpha$ is chosen in the interval $(0, 1/\lambda_{max}(A))$, then one can show that 
$$
x^{(k)} \xrightarrow{k \rightarrow \infty} v
$$
where $v = (I - \alpha A)^{-1} \beta \geq 0$.

## 4. Hubs and Authorities

An important **hub** is a node that points to many important **authorities** . An important authority is one that is pointed to by many important hubs.                                             

This method combines hub and authorities in a mutual recursion, beginning with an initial assignment of hub and authority scores or every node $\mathbf{x}^0$ and $(\mathbf{y}^0)^T$, respectively. The updates are as follows
$$
\begin{aligned}
\mathbf{x}_i^{(k+1)} &= \alpha \sum^n_{j=1}A_{ij}y_j^{(k)}, \quad i.e., \quad \mathbf{x}^{(k+1)} = \alpha A \mathbf{y}^ {(k)}\\
\mathbf{y}_i^{(k)} &= \beta \sum^n_{j=1} A_{ji}x_j^{(k)}, \quad i.e., \quad
\mathbf{y}^{(k)} = \beta A^T x^{(k)}
\end{aligned}
$$
So that the updated values are
$$
x^{(k+1)} = \alpha \beta AA^T x^{(k)}\\
y^{(k+1)} = \alpha \beta A^TA y^{(k)}\\
$$
Choosing $\alpha \beta = 1/\lambda _{\text {max}}(AA^ T)$, we can show that $\mathbf{x}^ k \to \mathbf{v}$ and $\mathbf{y}^ k \to \mathbf{w}$, where $AA^ T \mathbf{v} = \lambda _{\text {max}}(AA^ T) \mathbf{v}$ and $A^ T A \mathbf{w} = \lambda _{\text {max}}(A^ T A) \mathbf{w}$. In fact, the non-zero eigenvalues of $AA^ T$ and $A^ T A$ are the same and eigenvector of $A^TA$ is $\mathbf{w} = A^ T \mathbf{v}$.

