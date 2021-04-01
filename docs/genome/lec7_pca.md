# Principal Components Analysis

There are 3 topics.

## Definition 1: Minimize projection residuals:

* PC1 : straight line with smallest orthogonal distance to all points
* PC1 & PC2: Plane with smallest orthogonal distance to all points

**Proof:**

Let $\mathbf{x}^{(1)},\mathbf{x}^{(2)},\mathbf{x}^{(n)}\in \mathbb {R}^ p$ denote the $n$ data points in $p$ dimensional space. The first principal component is the line spanned by a unit vector $\mathbf{w}\in \mathbb {R}^ p$ such that $\mathbf{w}$ **minimizes the sum of squared residuals of the orthogonal projections of data $\mathbf{x}^{(i)}$ onto $\mathbf{w}$.**
$$
\min _{\mathbf{w}\in \mathbb {R}^ p} \sum _{i=1}^ n \left\|  \mathbf{x}^{(i)}-(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w} \right\| ^2
$$
where $(\mathbf{x}^{(i)}\cdot \mathbf{w})$ or equivalently $\mathbf{w}^T\mathbf{x}^{(i)}$ is the length of projection from $\mathbf{x}^{(i)}$ onto $\mathbf{w}$. $(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w}$ is the projection of $\mathbf{x}^{(i)}$ onto $\mathbf{w}$ in the direction of $\mathbf{w}$.

## Definition 2: Maximize projection variance: 

* the principal components (PCs) are descending-ordered according to the variance in the direction they are pointing to.
* PCs are perpendicular to each other.

**Proof:**

Let $\mathbf{x}^{(1)},\mathbf{x}^{(2)},\mathbf{x}^{(n)}\in \mathbb {R}^ p$ denote the $n$ data points in $p$ dimensional space. The first principal component is the line spanned by a unit vector $\mathbf{w}\in \mathbb {R}^ p$ such that $\mathbf{w}$  **maximizes the sum of squared norms of the orthogonal projections of data  $\mathbf{x}^{(i)}$ onto $\mathbf{w}$.**

We can see this by breaking down the sum of residuals:
$$
\begin{aligned}
\sum _{i=1}^ n \left\|  \mathbf{x}^{(i)}-(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w} \right\| ^2 &= \sum _{i=1}^ n \left(\mathbf{x}^{(i)}-(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w}\right)\cdot \left(\mathbf{x}^{(i)}-(\mathbf{x}^{(i)}\cdot \mathbf{w})\mathbf{w}\right)\\
 &=\sum _{i=1}^ n \left(\left\|  \mathbf{x}^{(i)} \right\| ^2 - 2(\mathbf{x}^{(i)}\cdot \mathbf{w})^2 +(\mathbf{x}^{(i)}\cdot \mathbf{w})^2\left\|  \mathbf{w} \right\| ^2\right)\\
 &= \sum _{i=1}^ n \left\|  \mathbf{x}^{(i)} \right\| ^2-\sum _{i=1}^ n (\mathbf{x}^{(i)}\cdot \mathbf{w})^2 \qquad (\left\|  \mathbf{w} \right\| ^2=1)\\
\end{aligned}
$$
Since the first component $\sum _{i=1}^ n \left\|  \mathbf{x}^{(i)} \right\| ^2$ is constant w.r.t. $\mathbf{w}$, minimizing the expression above over all choices of $\mathbf{w}$ is equivalent to maximizing only the second component, which is the sum of squared norms of the orthogonal projections of data onto $\mathbf{w}$.
$$
\max _{\mathbf{w}\in \mathbb {R}^ p} \sum _{i=1}^ n \left\|  (\mathbf{x}^{(i)}\cdot \mathbf{w}) \right\| ^2
$$

## Definition 3: Spectral decomposition

* Covariance matrix (or correlation matrix) $R = \frac{1}{n} X^TX$ is symmetric and positive semidefinite

* Spectral Decomposition Theorem: Every real symmetric matrix $R$ can be decomposed as
  $$
  R = V\Lambda V^T
  $$
  where $\Lambda$ is diagonal and $V$ is orthogonal.

* Columns of $V$ (= eigenvectors of $R$) are the PCs

* Diagonal entries of $\Lambda$ (= eigenvalues of $R$) are variances along PCs

**Proof:**

Let $\mathbf{x}^{(1)},\mathbf{x}^{(2)},\mathbf{x}^{(n)}\in \mathbb {R}^ p$ denote the $n$ data points in $p$ dimensional space. The first principal component is the line spanned by a unit vector $\mathbf{w}\in \mathbb {R}^ p$ such that **$\mathbf{w}$ is an eigenvector corresponding to the largest eigenvalue of the the sample covariance matrix $S$.**
$$
\mathbf{S} = \frac{1}{n-1}\mathbb {X}^ T \mathbb {X}\qquad \text {where } \mathbb {X}\, =\,  \begin{pmatrix}  \leftarrow & (\mathbf{x}^{(1)})^ T& \rightarrow \\ \leftarrow & (\mathbf{x}^{(2)})^ T& \rightarrow \\ & \vdots & \\ \leftarrow & (\mathbf{x}^{(n)})^ T& \rightarrow \\ \end{pmatrix}
$$
The sum of squared norms of the projected data can be written in the following matrix form:
$$
\sum _{i=1}^ n (\mathbf{x}^{(i)}\cdot \mathbf{w})^2 = \mathbf{w}^ T(\mathbb {X}^ T \mathbb {X}) \mathbf{w}\quad \text {where }\mathbb {X}\, =\, \begin{pmatrix}  \leftarrow & (\mathbf{x}^{(1)})^ T& \rightarrow \\ \leftarrow & (\mathbf{x}^{(2)})^ T& \rightarrow \\ & \vdots & \\ \leftarrow & (\mathbf{x}^{(n)})^ T& \rightarrow \\ \end{pmatrix}
$$
Since $\mathbb {X}^ T \mathbb {X}$ is a real symmetric matrix, it can be diagonalized as
$$
\begin{aligned}
\mathbb {X}^ T \mathbb {X}= \mathbf{V}\Lambda \mathbf{V}^ T \qquad \text{ where } \Lambda  &= \begin{pmatrix}  \lambda _1& & & \\ & \lambda _2& & \\ & & \ddots & \\ & & & \lambda _ p \end{pmatrix}\\ v &= \begin{pmatrix}  |& |& & |\\ \mathbf v_{\lambda _1}& \mathbf v_{\lambda _2}& \cdots & \mathbf v_{\lambda _ p}\\ |& |& & |\\ \end{pmatrix}
\end{aligned}
$$
The eigenvalues can be ordered in decreasing order, i.e,  $\lambda _1\geq \lambda _2 \geq \ldots \lambda _ p$, and $\mathbf v_{\lambda _ i}$ is an eigenvector corresponding to the eigenvalue $\lambda_i$.

Write $\mathbf{w}$ as a linear combination of the orthonormal eigenvectors $\mathbf{w}=c_ i \mathbf v_{\lambda _ i}$. Then the expression we need to maximize is
$$
\mathbf{w}^ T(\mathbb {X}^ T \mathbb {X}) \mathbf{w}= \left(\sum _{j=1}^ p c_ j \mathbf v_{\lambda _ j}^ T\right) (\mathbb {X}^ T \mathbb {X}) \left(\sum _{i=1}^ p c_ i \mathbf v_{\lambda _ i}\right) = \sum _{i=1}^{p} c_ i^2 \lambda _ i
$$
Because $\mathbf{w}$ is a unit vector, the quantity above is maximized when most "weight" is put on eigenvector for the largest eigenvalue, i.e. when $c_1 = 1$ and $c_j = 0$ for all $j \neq 1$. Hence, $\mathbf{w} = \mathbf{v}_{\lambda_1}$.

