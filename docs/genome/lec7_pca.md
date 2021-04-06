# Principal Components Analysis

There are 3 topics and 3 exercises.

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

> #### Exercise 16
>
> Consider $n \times n$ matrix
> $$
> \mathbf{H}= \mathbf{I}_ n - \frac{1}{n} \begin{bmatrix}  1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}_{n \times 1}\begin{bmatrix}  1 &  1 &  \cdots &  1 \end{bmatrix}_{1 \times n}
> $$
> Compute $\mathbf{H}^2$. 
>
> Let $n=2$. What is the subspace that $\mathbf{H}$ projects any vector $\mathbf{x} \in \R^2$ onto?
>
> > **Answer**: 
> >
> > $\mathbf{H}^2= \mathbf{H}\\ \left\{  \mathbf y: \frac{y^{(1)} + y^{(2)}}{2} = 0 \right\}$
>
> > **Solution**: 
> >
> > By definition of projection matrix $\mathbf{H}^2= \mathbf{H}$.
> >
> > By looking at the columns of $\mathbf{H}$.
> > $$
> > H = \begin{pmatrix}  \frac{1}{2} &  -\frac{1}{2} \\ -\frac{1}{2} &  \frac{1}{2} \end{pmatrix}
> > $$
> > Notice that the two columns are **linearly dependent** (through a sign change). The column space of $\mathbf{H} $ is all vectors that are scalar multiples of 
> > $$
> > \begin{bmatrix}  \frac{1}{2} \\ -\frac{1}{2} \end{bmatrix}
> > $$
> > which is the line $y^{(1)} = -y^{(2)}$.

> #### Exercise 17
>
> Given two data points in 2 dimensions
> $$
> \mathbf{x}^{(1)} = (x^{(1)}, y^{(1)})= (0,1)\\
> \mathbf{x}^{(2)} = (x^{(2)}, y^{(2)})= (0,-1)\\
> $$
> This sample is already centered, with sample mean 0 in both $x$ and $y$ coordinates.
>
> Find the direction of largest variance (PC1) without computation and then compute PC1.
>
> > **Answer**: The direction of PC1 is $(0,1)$.
>
> > **Solution**: 
> >
> > The direction the PC1 should be $y$ direction (since there are only two points, the direction of the largest variance is any scaler multiple of their difference $\mathbf{x}^{(1)}-\mathbf{x}^{(2)}$), since the data points only spread in the $y$ direction. 
> >
> > The unbiased sample covariance matrix is 
> > $$
> > \begin{aligned}
> > \mathbf{S} &= \frac{1}{n-1}\mathbb{X}^T\mathbb{X}\\
> > &=\frac{1}{(2-1)} \mathbb {X}^ T \mathbb {X}\\
> > &= \begin{pmatrix} x^{(1)}& x^{(2)}\\ y^{(1)}& y^{(2)}\end{pmatrix}\begin{pmatrix} x^{(1)}& y^{(1)}\\ x^{(2)}& y^{(2)}\end{pmatrix}\\
> > &= \begin{pmatrix}  0& 0\\ 1& -1\end{pmatrix}\begin{pmatrix}  0& 1\\ 0& -1\end{pmatrix}\,\\ &=\, \begin{pmatrix}  0& 0\\ 0& 2\end{pmatrix}
> > \end{aligned}
> > $$
> > Since in general for diagonal matrices, the eigenvalues are the diagonal entries themselves, $\lambda_1 = 0, \lambda_2 = 2$, and the eigenvectors corresponding to the $i$-th diagonal entry is the $i$-th standard basis vector, i.e. $\mathbf{v}_{\lambda_1} = \begin{pmatrix} {{1}} \\ 0\end{pmatrix}, \mathbf{v}_{\lambda_2} = \begin{pmatrix} {{0}} \\ 1\end{pmatrix}$.
> > $$
> > \begin{pmatrix}  {{0}} & 0\\ 0& 2\end{pmatrix}\begin{pmatrix} {{1}} \\ 0\end{pmatrix} = {{0}}  \begin{pmatrix} {{1}} \\ 0\end{pmatrix}\\
> > \begin{pmatrix}  0& 0\\ 0& {{2}} \end{pmatrix}\begin{pmatrix} 0\\ {{1}} \end{pmatrix} =  {{2}}  \begin{pmatrix} 0\\ {{1}} \end{pmatrix}
> > $$
> > The eigenvector corresponding to the larger eigenvalue $2$ is $\begin{pmatrix} 0& {{1}} \end{pmatrix}^ T$.
> >
> > With this direction of PC1, PCA will project all vectors orthogonally onto the $y$ axis. The **projection matrix** is given by the transpose of the eigenvector $\mathbf v_{\text {PC1}}$.
> > $$
> > \begin{pmatrix}  \leftarrow & {{\mathbf v_{\text {PC1}}}} & \rightarrow \end{pmatrix}\begin{pmatrix}  x^{(i)}\\ y^{(i)} \end{pmatrix}\, =\,  {{\begin{pmatrix}  0& 1 \end{pmatrix}}}  \begin{pmatrix}  x^{(i)}\\ y^{(i)} \end{pmatrix} =  y^{(i)}
> > $$

> #### Exercise 18
>
> We will use PCA to project the following two data points in 2 dimensions into 1 dimension
> $$
> \mathbf{x}^{(1)} = (x^{(1)}, y^{(1)})= (1,1/2)\\
> \mathbf{x}^{(2)} = (x^{(2)}, y^{(2)})= (-1,-1/2)
> $$
> The data is already centered. (Check the sample mean of each coordinate is zero. E.g. $\overline{x_1}=\frac1n \sum _{i=1}^ n x_1^{(i)}= (1+(-1))/2=0$)
>
> Find the direction in which the sample variance is the largest (PC1), and then compute the principal components of this data.
>
> > **Answer**:  PC1 is $(2,1)$. 
>
> > **Solution**: 
> > $$
> > \mathbf{x}^{(1)}-\mathbf{x}^{(2)} = (1,1/2) - (-1,-1/2) = (2,1)
> > $$
> > The **unbiased covariance matrix** $S$ for centered data is 
> > $$
> > \begin{aligned}
> > \mathbf{S}=\frac{1}{2-1} \mathbb {X}^ T \mathbb {X} &= \begin{pmatrix} x^{(1)}& x^{(2)}\\ y^{(1)}& y^{(2)}\end{pmatrix}\begin{pmatrix} x^{(1)}& y^{(1)}\\ x^{(2)}& y^{(2)}\end{pmatrix}\\
> > \\ &=\begin{pmatrix}  1& -1\\ 1/2& -1/2\end{pmatrix}\begin{pmatrix}  1& 1/2\\ -1& -1/2\end{pmatrix}=\frac12 \begin{pmatrix}  4& 2\\ 2& 1\end{pmatrix}.
> > \end{aligned}
> > $$
> > Or using the **biased Sample covariance** definition, we get $\mathbf{S}=\frac14 \begin{pmatrix}  4& 2\\ 2& 1\end{pmatrix}$.
> >
> > Find **eigenvalues** $\lambda _1, \lambda _2$ of $S$ and for each eigenvalue a corresponding eigenvector $\mathbf v_{\lambda _1}$ and $\mathbf v_{\lambda _2}$.
> > $$
> > \begin{aligned}
> > \text {det}\begin{pmatrix}  4-\lambda & 2\\ 2& 1-\lambda \end{pmatrix} &= 0\\
> > (4-\lambda )(1-\lambda )-4\, =\, \lambda ^2-5\lambda &= 0
> > \end{aligned}
> > $$
> > Hence the eigenvalues of the original matrix $\frac12\begin{pmatrix}  4-\lambda & 2\\ 2& 1-\lambda \end{pmatrix}$ are $\lambda _1=5/2$ and $\lambda _2=0$. or $\lambda _1=5/4$ and $\lambda _2=0$ for biased covariance matrix $\mathbf{S}=\frac14 \begin{pmatrix}  4& 2\\ 2& 1\end{pmatrix}$.
> >
> > Find **eigenvectors** of $\lambda _1=5/2$, we solve
> > $$
> > \begin{aligned}
> > (\mathbf{S}-\lambda _1\mathbf{I}_2)\mathbf v_{\lambda _1} &= 0\\
> > \begin{pmatrix}  -1& 2\\ 2& -4 \end{pmatrix}\mathbf v_{\lambda _1}\, &= 0\\
> > \mathbf{v}_{\lambda_1} &= \begin{pmatrix}  2\\ 1\end{pmatrix}
> > \end{aligned}
> > $$
> > This is on the same line of the given data vectors that capture the largest variances of the data.
> >
> > Similarly, the **eigenvectors** of $\lambda_2 = 0$, we solve
> > $$
> > \begin{aligned}
> > (\mathbf{S}-\lambda _2\mathbf{I}_2)\mathbf v_{\lambda _2} &= 0\\
> > \begin{pmatrix}  4& 2\\ 2& 1 \end{pmatrix}\mathbf v_{\lambda _2}\, &= 0\\
> > \mathbf{v}_{\lambda_2} &= \begin{pmatrix}  1\\ -2\end{pmatrix}
> > \end{aligned}
> > $$
> > The eigenvectors of $\lambda_2$ are all scalar multiples of $\begin{pmatrix}  2\\ 1\end{pmatrix}$.
> >
> > Find the projection $y^{(1)}$ and $y^{(2)}$ of the two given data points onto the first PC.
> >
> > Note that the **projection** of a vector $\mathbf{x}$ onto PC1 (the line spanned by $\mathbf{v}_{\lambda_1}$) is given by
> > $$
> > \text {proj}_{\text {PC1}} \mathbf{x} = \begin{pmatrix}  \leftarrow &  (\hat{\mathbf v}_{\lambda _1})^ T& \rightarrow \\ \end{pmatrix} \mathbf{x}\qquad \text {where }\hat{\mathbf v}_{\lambda _1}^ T\, =\, \frac{\mathbf v_{\lambda _1}^ T}{\left\|  \mathbf v_{\lambda _1} \right\| }\, =\, \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\end{pmatrix}
> > $$
> > Hence, the projection of the data points onto PC1 are
> > $$
> > \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\end{pmatrix} \mathbf{x}^{(1)} = \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\end{pmatrix}\begin{pmatrix} 1\\ 1/2\end{pmatrix}=\frac{\sqrt{5}}{2}\\
> > \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\end{pmatrix} \mathbf{x}^{(2)} = -\frac{\sqrt{5}}{2} \qquad (\mathbf{x}^{(2)}=-\mathbf{x}^{(1)}
> > $$
> > Check the **empirical variance** in PC1, which is indeed $\lambda_1$: 
> > $$
> > (\frac{\sqrt{5}}{2})^2 + (-\frac{\sqrt{5}}{2})^2 = \frac{5}{2}
> > $$
> > Transform data points into PC-axes by using **transformation matrix** $\mathbf{P}^ T$, which is the square matrix with orthogonal unit eigenvectors as its rows. The transformation matrix $\mathbf{P}^T$ for projecting onto first $k$ PCs are:
> > $$
> > \begin{pmatrix}  \leftarrow & \frac{\mathbf v_{\lambda _1}}{\left\|  \mathbf v_{\lambda _1} \right\| }& \rightarrow \\ & \vdots & \\ \leftarrow & \mathbf v_{\lambda _ k}/\left\|  \mathbf v_{\lambda _ k} \right\| & \rightarrow \\ \end{pmatrix}
> > $$
> > In this two points case, with only two eigenvalues, the transformation matrix is 
> > $$
> > \mathbf{P}^ T = \begin{pmatrix}  \leftarrow & \frac{\mathbf v_{\lambda _1}}{\left\|  \mathbf v_{\lambda _1} \right\| }& \rightarrow \\ \leftarrow & \frac{\mathbf v_{\lambda _2}}{\left\|  \mathbf v_{\lambda _2} \right\| }& \rightarrow \\ \end{pmatrix}= \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\\ -1& 2\end{pmatrix}
> > $$
> > (or any reflection of the above, e.g. $\frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\\ 1& -2\end{pmatrix}$, depending on the orientation of the coordinate system.)
> >
> > The transformation of data points to PC1 can be computed by
> > $$
> > \mathbf{P}^ T\mathbf{x}^{(1)}\, =\, \begin{pmatrix}  \leftarrow & \frac{\mathbf v_{\lambda _1}}{\left\|  \mathbf v_{\lambda _1} \right\| }& \rightarrow \end{pmatrix}\begin{pmatrix}  |& |\\ \mathbf{x}^{(1)}& \mathbf{x}^{(2)}\\ |& |\\ \end{pmatrix}
> > $$
> > For example, in the PC-coordinates, the first data point in coordinates
> > $$
> > \frac{1}{\sqrt{5}}\begin{pmatrix}  2& 1\\ -1& 2\end{pmatrix}\begin{pmatrix} 1\\ 1/2\end{pmatrix} = \begin{pmatrix}  \frac{\sqrt{5}}{2} \\ 0 \end{pmatrix}.
> > $$







 



