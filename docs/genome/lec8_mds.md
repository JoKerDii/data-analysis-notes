# Multidimensional Scaling (MDS)

There are 3 topics and 0 exercise.

## 1.  Distance and Dissimilarity

$D \in \R^{n \times n}$ is a **distance matrix** if
$$
D_{ii} = 0, D_{ij} \geq 0, D_{ij} = D_{ji}, D_{ij} \leq D_{ik} + D_{jk}  \text{ for all } i,j,k
$$

* E.g. Euclidean distance, Manhattan distance, maximum distance, ...

$D \in \R^{n \times n}$ is a **dissimilarity matrix** if
$$
D_{ii} = 0, D_{ij} \geq 0, D_{ij} = D_{ji} \text{ for all } i,j,k
$$

* More flexible than distances, works e.g. for rankings

## 2. Introduction to MDS

The goal of MDS is to preserve the relative distance between high-dimensional vectors in a low-dimensional space (usually 2 or 3). MDS can take the distance matrix $\mathbf{D} \in \R^{n \times n}$ as input without knowing the original dataset. MDS is used to find points $\mathbf{y}_1, \dots , \mathbf{y}_ n \in \mathbb {R}^ q$ to minimize an objective, where $\mathbf{y_i}$ and $\mathbf{y_j}$ are the projections of $\mathbf{x_i}$ and $\mathbf{x_j}$ in a low-dimensional space, respectively.

Variation of MDS is that the objective function can be different. Given a matrix $D \in \R^{n \times n}$, determine points $y_1, ..., y_n \in \R^q$ such that

* Classical MDS: 

  minimize 
  $$
  \sum^n_{i=1} \sum^n_{i=1} (D_{ij} - \|y_i - y_j\|_2)^2
  $$
  assuming $D$ is a Euclidean distance matrix

* Weighted MDS:

  minimize 
  $$
  \sum^n_{i=1} \sum^n_{i=1} w_{ij}(D_{ij} - \|y_i - y_j\|_2)^2
  $$
  assuming $D$ is a distance matrix and $w_{ij}$ are non-negative weights

  * solved iteratively using stress majorization

* Non-metric MDS

  minimize 
  $$
  \sum^n_{i=1} \sum^n_{i=1} (\theta(D_{ij}) - \|y_i - y_j\|_2)^2
  $$
  assuming $D$ is a dissimilarity matrix

  * also optimize over increasing function $\theta$
  * finds low-dimensional embedding that respects ranking of dissimilarities
  * solved numerically (isotonic regression); very time-consuming

## 3. Classical MDS

First convert a distance matrix $D$, with $D_{ij} = \|x_i - x_j\|_2$ into a positive semidefinite matrix $XX^T$, namely
$$
XX^T = -\frac{1}{2} (I - \frac{1}{n}ee^t) D^2 (I-\frac{1}{n}ee^t)
$$
where $e$ is a vector of ones.

Note that $XX^T$ is a **doubly centered** distance matrix
$$
(XX^T)_{ij} = -\frac{1}{2}(D_{ij}^2 - D_{i.}^2 - D_{.j}^2 + D_{..}^2)
$$
$D_{ij}^2$ can be represented by $XX^T$
$$
\begin{aligned}
D_{ij}^2 &=\|x_i - x_j\|_2^2\\
&= \|x_i\|^2 + \|x_j\|^2 - 2 x_i^Tx_j \\
 &= (XX^T)_{ii} + (XX^T)_{jj} - 2(XX^T)_{ij}
\end{aligned}
$$
$D_{ij}$ contains all the distances between every two vectors. Whereas, $XX^T_{ij}$ represents the angle between two vectors and the diagonal of it represents the lengths of vectors. By $XX^T$, the data configuration is represented by the angle between two vectors and the length of each vector. $D_{ij}$ and $XX^T$ are just two ways to represent distances.

The **Classical MDS** is
$$
\min_Y \sum^n_{i=1} (D_{ij}^2 - \|y_i - y_j\|^2_2)^2
$$
which is equivalent to 
$$
\min_{Y} \text{ trace}(XX^T - YY^T)^2
$$
where $YY^T$ is the doubly centered distance (or the length of each point).

Assume the dimension $X_{n \times p}$ and $Y_{n \times q}$, to find the best $Y$, or equivalently, to find the best rank $q$ approximation of $XX^T$ that minimize the objective, we apply **eigenvalue decomposition**:
$$
XX^T = V\Lambda V^T
$$
where columns of $V$ are eigenvectors of $XX^T$, $\Lambda$ is diagonal containing eigenvalues of $XX^T$.

Best rank $q$ approximation of $XX^T$ is given by choosing $q$ largest eigenvalues and corresponding eigenvectors, i.e. $YY^T = V_1 \Lambda_1 V_1^T$, or equivalently, $Y=V_1 \Lambda_1^{1/2}$.

Classical MDS is like PCA on $B = XX^T_{n \times n}$, we want to find the largest eigenvalues and their corresponding eigenvectors ; whereas in classical PCA, we start off with covariance matrix $X^TX_{p \times p}$ and find the largest eigenvalues with their corresponding eigenvectors.