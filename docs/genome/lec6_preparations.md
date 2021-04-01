# Linear Algebraic Preparations

Some notes for review. There are 4 topics.

## 1. Expectation and Covariance of a Random Vector

1. **Expectation**
   Assume $\mathbf{X} \in \mathbb {R}^n$ denotes a random vector, then $\mathbf E[ \mathbf{X} ]$ is a vector in $\R^n$.

   Assume $\mathbf{X} \in \mathbb {R}^3$ and that $\mathbb {X}$ is Gaussian random vector:
   $$
   \mathbf{X} \sim \mathcal{N}(\mathbf{\mu }, \Sigma )=\mathcal{N}\left( \begin{pmatrix}  -10 \\ 0 \\ 2 \\ \end{pmatrix}, \begin{pmatrix}  1 &  2 &  0 \\ 2 &  2 &  1 \\ 0 &  1 &  1 \\ \end{pmatrix} \right)
   $$
   where $\mu$ is the mean and $\Sigma$ is the covariance matrix of $\mathbb{X}$.

   Then $\mathbf E[ \mathbf{X} ]$ is $\begin{pmatrix}  -10 \\ 0 \\ 2 \\ \end{pmatrix}$ since the definition:
   $$
   \mathbf{E}[ \mathbf{X} ]_ i = \mathbf{E}[ \mathbf{X}_ i ]
   $$
   Note that the diagonal entries of the given covariance matrix denote the variances of $\mathbf{X}_1, \mathbf{X}_2,$ and $\mathbf{X}_3$. Therefore,
   $$
   \mathbf{X}^{(1)} \sim \mathcal{N}(-10, 1), \,  \,  \mathbf{X}^{(2)} \sim \mathcal{N}(0, 2), \,  \,  \mathbf{X}^{(3)} \sim \mathcal{N}(2, 1).
   $$
   It follows that
   $$
   \begin{aligned}
   \mathbf E{[\mathbf{X}]_1} &= \mathbf E[ \mathbf{X}_1 ] = -10\\
   \mathbf E{[\mathbf{X}]_2} &= \mathbf E[ \mathbf{X}_2 ] = 0\\
   \mathbf E{[\mathbf{X}]_3} &= \mathbf E[ \mathbf{X}_3 ] = 2
   \end{aligned}
   $$

2. **Variance** and **covariance**
   $$
   \begin{aligned}
   \textsf{Var}(X) &= \mathbf E[(X - \mathbf E[X])^2]\\  &= \mathbf E[X^2 - 2 X \mathbf E[X] + (\mathbf EX)^2]\\ & = \mathbf E[X]^2 - (\mathbf E[X])^2\\ 
   \textsf{Cov}(X, Y) &= \mathbf E[(X - \mathbf E[X] )(Y - \mathbf EY)]\\ &= \mathbf E\left[XY - X \mathbf EY - Y \mathbf EX + \mathbf E[X] \mathbf E[Y]\right]\\ & = \mathbf E[XY] - \mathbf E[X] \mathbf E[Y]\\
   \end{aligned}
   $$

3. **Covariance matrix**
   
   Let $\mathbf{X} \in \mathbb {R}^ d$ denote a random vector. The covariance matrix is defined as
   $$
   \begin{aligned}
   \Sigma &= \mathbf E\left[ (\mathbf{X} - \mathbf E[\mathbf{X}]) (\mathbf{X} - \mathbf E[\mathbf{X}])^ T \right]\\ &= \mathbf E\left[ \mathbf{X} \mathbf{X}^ T - \mathbf{X} \mathbf E[\mathbf{X}]^ T - \mathbf E[\mathbf{X}] \mathbf{X}^ T + \mathbf E[\mathbf{X}] \mathbf E[\mathbf{X}]^ T \right]\\ &= \mathbf E[ \mathbf{X} \mathbf{X}^ T ] - \mathbf E[ \mathbf{X}] \mathbf E[ \mathbf{X}]^ T
   \end{aligned}
   $$
   
   Then the $\Sigma_{ij}$ is 
   $$
   \begin{aligned}
   \Sigma_{ij} &= (\mathbf E[ \mathbf{X} \mathbf{X}^ T])_{ij} - (\mathbf E[\mathbf{X}] \mathbf E[\mathbf{X}]^ T)_{ij}\\
   &= \mathbf E[ \mathbf{X}_ i \mathbf{X}_ j] - \mathbf E[\mathbf{X}]_ i \mathbf E[\mathbf{X}]_ j\\
   &= \textsf{Cov}(\mathbf{X}_ i, \mathbf{X}_ j)
   \end{aligned}
   $$

## 2. Empirical Mean and Covariance Matrix of a Vector Data Set

1. Sample mean for a dataset of vectors, a.k.a. **empirical mean** $\overline{\mathbf{X}}$.

   Given sample vectors $\mathbf{X}^{(1)} , \mathbf{X}^{(2)}, \mathbf{X}^{(3)}, \mathbf{X}^{(4)}$ in $\R^3$ :
   $$
   \mathbf{X}^{(1)}= \begin{pmatrix}  8 \\ 4 \\ 7 \\ \end{pmatrix},\,  \mathbf{X}^{(2)}= \begin{pmatrix}  2 \\ 8 \\ 1 \\ \end{pmatrix},\,  \mathbf{X}^{(3)}= \begin{pmatrix}  3 \\ 1 \\ 1 \\ \end{pmatrix},\,  \mathbf{X}^{(4)}= \begin{pmatrix}  9 \\ 7 \\ 4 \\ \end{pmatrix}.
   $$
   Empirical mean of the vectors is 
   $$
   \begin{aligned}
   \overline{\mathbb {X}} &= \frac{1}{4} \left( \begin{pmatrix}  8 \\ 4 \\ 7 \\ \end{pmatrix} + \begin{pmatrix}  2 \\ 8 \\ 1 \\ \end{pmatrix} + \begin{pmatrix}  3 \\ 1 \\ 1 \\ \end{pmatrix} + \begin{pmatrix}  9 \\ 7 \\ 4 \\ \end{pmatrix} \right)\\
   &= \begin{pmatrix}  5.5 \\ 5.0 \\ 3.25 \end{pmatrix}
   \end{aligned}
   $$

5. Sample covariance for a data set of  vectors, a.k.a. **empirical covariance matrix** or **sample covariance matrix**.
   $$
   \mathbf{S}=\frac{1}{n} \sum _{i=1}^{n} \left(\mathbf{X}^{(i)} (\mathbf{X}^{(i)})^ T \right) - \overline{\mathbf{X}}~ \overline{\mathbf{X}}^ T
   $$
   where $ \overline{\mathbf{X}}=\frac{1}{n}\sum _{i=1}^ n \mathbf{X}^{(i)}$ is the empirical or sample mean.

   Suppose $I_n$ is the $n \times n$ identity matrix, and $1 \in \R^n$ is the vector with all 1 entries, the formula for sample covariance matrix is
   $$
   S = \frac{1}{n} \mathbb {X}^ T ( I_ n - \frac{1}{n} \mathbf{1} \mathbf{1}^ T ) \mathbb {X}
   $$

6. **Orthogonal Projection**

   Let $\mathrm{{\boldsymbol X}}_1, \ldots , \mathrm{{\boldsymbol X}}_ n \in \mathbb {R}^ d$ denote a data set and let
   $$
   \mathbb {X} = \begin{pmatrix}  \longleftarrow &  \mathbf{X}_1^ T &  \longrightarrow \\ \longleftarrow &  \mathbf{X}_2^ T &  \longrightarrow \\ \vdots &  \vdots &  \vdots \\ \longleftarrow &  \mathbf{X}_ n^ T &  \longrightarrow \\ \end{pmatrix}
   $$
   The empirical covariance matrix $S$ of the data set is
   $$
   S = \frac{1}{n} \mathbb {X}^ T H \mathbb {X}
   $$
   where 
   $$
   H = I_ n - \frac{1}{n} \mathbf{1} \mathbf{1}^ T
   $$
   The matrix $H \in \mathbb {R}^ n$ is an **orthogonal projection**.

   In general, a matrix $M$ is a orthogonal projection onto a subspace $S$ if 

   * $M$ is symmetric
   * $M^2 = M$
   * $S = \{  \mathrm{{\boldsymbol y}} : \,  M \mathbf{x}= y \,  \,  \text {for some} \,  \,  \mathbf{x}\in \mathbb {R}^ n \}$

   Note that

   * For any positive integer $k$ and any vector $\mathbf{x} \in \R^n$, we have $H^ k \mathbf{x}= H \mathbf{x}$.

     Because $H$ is an orthogonal projection, we know $H^2 = H$ and it follows that for any $k \geq 2$, we also have
     $$
     H^ k = H^{k-1} = H^{k-2} = \cdots = H^2 = H
     $$
     Note that $H$ is an **Idempotent matrix**. In linear algebra, an idempotent matrix is a matrix which, when multiplied by itself, yields itself. Idempotent matrices are symmetric matrices that has their value of **trace = rank = sum of diagonal**.

   * The matrix $H$ is a projection onto the subspace of vectors perpendicular to the vector $1 \in \R^n$, which has all of its entries equal to 1.

     If $\mathbf{x}\perp \mathbf{1}$, then we have
     $$
     H \mathbf{x}= \mathbf{x}- \frac{1}{n} \mathbf{1} (\mathbf{1} \cdot \mathbf{x}) = \mathbf{x}
     $$
     Moreover, $H \mathbf{x}\perp \mathbf{1}$ because
     $$
     \begin{aligned}
     H\mathbf{x} \cdot 1 &= (\mathbf{x}- \frac{1}{n} \mathbf{1} (\mathbf{1} \cdot \mathbf{x})) \cdot \mathbf{1}\\
      &= \mathbf{x}\cdot \mathbf{1} - \frac{1}{n} (\mathbf{1} \cdot \mathbf{1}) (\mathbf{1} \cdot \mathbf{x})\\
      &= 0
     \end{aligned}
     $$
     These two facts imply that the outputs of $H$ consist of all vectors that are perpendicular to $\mathbf{1}$.

   * The matrix $H$ is a projection onto the subspace $\{  \mathbf{x}: \frac{1}{n} \sum _{i = 1}^ n \mathbf{x}^ i = 0\}  \subset \mathbb {R}^ n$. ( In other words, this is the set of vectors having coordinate-wise average equal to 0.)

     As mentioned above
     $$
     \mathbf{x}\perp \mathbf{1} \Leftrightarrow \frac{1}{n} \sum _{i = 1}^ n \mathbf{x}^ i = 0
     $$
     Therefore
     $$
     \mathbf{x}\cdot \mathbf{1} = \sum _{i = 1}^ n \mathbf{x}^ i
     $$

## 3. Measuring the Spread of a Point Cloud

1. **Projection** onto a Line

   Let $\mathbf{u}\in \mathbb {R^ d}$ denotes a unit vector (i.e. $\sum _{i = 1}^ d (\mathbf{u}^ i)^2 = 1$). In general, the projection of a vector $\mathbf{x} \in \R^d$ onto a **unit vector** $\mathbf{u}$ is defined to be 
   $$
   \text {proj}_{\mathbf{u}} \mathbf{x}= \left(\mathbf{u}\cdot \mathbf{x}\right) \mathbf{u}
   $$
   Note that if the vector onto which we project is not given as a unit vector but a vector, say $\mathbf{v}$, with length $\|\mathbf{v}\|$, then from the unit vector $\mathbf{u} = \frac{\mathbf{v}}{\|v\|}$ and apply the same formula as above: 
   $$
   \, \displaystyle \text {proj}_{\mathbf v} \mathbf{x}\, = \, \left(\frac{\mathbf v}{\left\|  \mathbf v \right\| } \cdot \mathbf{x}\right) \frac{\mathbf v}{\left\|  \mathbf v \right\| }\, =\, \left(\frac{\mathbf v\cdot \mathbf{x}}{\left\|  \mathbf v \right\| ^2} \right) \mathbf v
   $$

2. **Empirical Variance of a Data Set in a Given Direction**

   The number $\mathbf{u} \cdot X_i$ (where $\mathbf{u}$ is a unit vector ) gives the **signed distance** from the origin to the endpoint of the projection $\text{proj}_u X_i$. By signed distance, we mean that $|\mathbf{u} \cdot X_i|$ is the length of $\text{proj}_u X_i$ and
   $$
   \begin{aligned}
   \mathbf{u}\cdot \mathrm{{\boldsymbol X}}_ i > 0 &\Longrightarrow \mathrm{{\boldsymbol X}}_ i \,  \,  \text {points approximately in the direction of } \,  \mathbf{u}\\
   \mathbf{u}\cdot \mathrm{{\boldsymbol X}}_ i \leq 0 &\Longrightarrow \mathrm{{\boldsymbol X}}_ i \,  \,  \text {points approximately in the opposite direction of } \,  \mathbf{u}\\
   \end{aligned}
   $$
   Thus, the empirical variance of our data set in the direction of $\mathbf{u}$ is
   $$
   \mathbf{u}^ T S \mathbf{u}
   $$
   where $\mathbf{u}$ is a fixed unit vector, and $S$ is the empirical covariance matrix of our data set.
   $$
   S = \frac{1}{n} \mathbb {X}^ T (I_n - \frac{1}{n} \mathbf{1} \mathbf{1}^ T) \mathbb {X}
   $$

3. **Variance of a Random Vector in a Given Direction**

   Since $\mathbf{X}$ is centered with mean $(0,0)^T$, we have that
   $$
   \Sigma = \mathbf E[ \mathrm{{\boldsymbol X}} \mathrm{{\boldsymbol X}}^ T ] = \begin{pmatrix}  \mathbf E[(\mathrm{{\boldsymbol X}}^1)^2] &  \mathbf E[\mathrm{{\boldsymbol X}}^1 \mathrm{{\boldsymbol X}}^2] \\ \mathbf E[\mathrm{{\boldsymbol X}}^1 \mathrm{{\boldsymbol X}}^2] &  \mathbf E[(\mathrm{{\boldsymbol X}}^2)^2] \\ \end{pmatrix}
   $$
   In particular,
   $$
   \Sigma _{11} = \textsf{Var}(\mathrm{{\boldsymbol X}}^1) = \mathbf E[(\mathrm{{\boldsymbol X}}^1 - \mathbf{0})^2]= \mathbf E[(\mathrm{{\boldsymbol X}}^1)^2]\\
   \Sigma _{22} = \textsf{Var}(\mathrm{{\boldsymbol X}}^2) = \mathbf E[(\mathrm{{\boldsymbol X}}^2 - \mathbf{0})^2] = \mathbf E[(\mathrm{{\boldsymbol X}}^2)^2]
   $$
   By inspection, we see that
   $$
   \Sigma _{11} = (1,0) \Sigma (1, 0)^ T\\
   \Sigma _{22} = (0,1) \Sigma (0, 1)^ T
   $$
   Therefore, if we express $\textsf{Var}(\mathrm{{\boldsymbol X}}^1)$ as $\mathbf{u}^ T \Sigma \mathbf{u}$, $\textsf{Var}(\mathrm{{\boldsymbol X}}^2)$ as $\mathrm{{\boldsymbol v}}^ T \Sigma \mathrm{{\boldsymbol v}}$ for some unit vector $\mathbf{u}$ and $\mathbf{v}$, then $\mathbf{u}= (1,0)^ T$ and $\mathrm{{\boldsymbol v}} = (0,1)^ T$.

   We can express $\textsf{Var}( \mathrm{{\boldsymbol X}}^1 + \mathrm{{\boldsymbol X}}^2 )$ as $\mathrm{{\boldsymbol w}}^ T \Sigma \mathrm{{\boldsymbol w}}$ for some vector $\mathrm{{\boldsymbol w}}$. Observe that
   $$
   \begin{aligned}
   \mathrm{{\boldsymbol w}}^ T \Sigma \mathrm{{\boldsymbol w}} &= \mathrm{{\boldsymbol w}}^ T \mathbf E[ \mathrm{{\boldsymbol X}} \mathrm{{\boldsymbol X}}^ T ] \mathrm{{\boldsymbol w}}\\
    &= \mathbf E[ \mathrm{{\boldsymbol w}}^ T \mathrm{{\boldsymbol X}} \mathrm{{\boldsymbol X}}^ T \mathrm{{\boldsymbol w}}]\\
    &= \mathbf E[ (\mathrm{{\boldsymbol w}}^ T \mathrm{{\boldsymbol X}})^2 ]
   \end{aligned}
   $$
   Note that $\mathrm{{\boldsymbol w}}^ T \mathrm{{\boldsymbol X}} = \mathrm{{\boldsymbol w}} \cdot \mathrm{{\boldsymbol X}} \in \mathbb {R}$. Hence,
   $$
   \mathrm{{\boldsymbol w}}^ T \Sigma \mathrm{{\boldsymbol w}} = \mathbf E[ (\mathrm{{\boldsymbol w}} \mathrm{{\boldsymbol X}})^2 ] = \textsf{Var}(\mathrm{{\boldsymbol w}}^ T \mathrm{{\boldsymbol X}})
   $$
   for all vectors $\mathrm{{\boldsymbol w}}$. Noting that $\mathrm{{\boldsymbol X}}^1 + \mathrm{{\boldsymbol X}}^2 = (1,1)^ T \mathrm{{\boldsymbol X}}$, we see that $\mathrm{{\boldsymbol w}} = (1,1)^ T$.

   Note that if we let $\mathrm{{\boldsymbol w}} = \frac{1}{\sqrt{2}} (1,1)^ T$ (a unit vector), then $\mathrm{{\boldsymbol w}}^ T \Sigma \mathrm{{\boldsymbol w}}$ describes the variance of $\mathbf{X}$ in the direction of $\mathrm{{\boldsymbol w}}$. More precisely, it is the variance of the random variable $w^T\mathbf{X} = \frac{1}{\sqrt{2}}(\mathbf{X}^1 + \mathbf{X}^2)$.

4. **Principal Component Analysis (PCA)**

    Let $\mathrm{{\boldsymbol X}}_1, \ldots , \mathrm{{\boldsymbol X}}_ n \in \mathbb {R}^ d$ denote a data set, and let $\mathbb {X}$ denote the matrix whose $i$-th row is $\mathrm{{\boldsymbol X}}_ i^ T$. Let
    $$
    S = \frac{1}{n} \mathbb {X}^ T \left(I_ n - \frac{1}{n} \mathbf{1} \mathbf{1}^ T \right) \mathbb {X}
    $$
    denote the empirical covariance matrix for this data set.

    Consider the optimization problem
    $$
    \text {argmax}_{\mathbf{u}: \left\|  \mathbf{u} \right\| _2^2 =1 } \mathbf{u}^ T S \mathbf{u}
    $$
    Let $\mathbf{u}^*$ denote the unit vector that maximizes $\mathbf{u}^ T S \mathbf{u}$.

    Intuitively, what PCA does basically is to find a direction/ coordinate $\mathbf{u}^*$ which captures most variance of the data set.

    * $\mathbf{u}^*$ is the direction that maximizes the empirical variance of the (projected) data points $\mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_1, \mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_2, \ldots , \mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_ n$.

      For any unit vector $\mathbf{u} \in \R^d$, the empirical variance of the data set $\mathbf{u}^ T \mathrm{{\boldsymbol X}}_1, \ldots , \mathbf{u}^ T \mathrm{{\boldsymbol X}}_ n \in \mathbb {R}$ is given by $\mathbf{u}^ T S \mathbf{u}$. Hence, if we maximize $\mathbf{u}^ T S \mathbf{u}$ over all unit vectors, the maximizer $\mathbf{u}^*$ has the property that the empirical variance of $\mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_1, \mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_2, \ldots , \mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_ n$ is as large as possible.

    * If $\mathrm{{\boldsymbol u^*}}^ T S \mathbf{u}^*$ is very large, then if we project our data set onto the line spanned by $\mathbf{u}^*$, we expect the projected data set to be fairly "spread out" (*i.e.*, the projected data set should have relatively large empirical variance). 

      For any unit vector $\mathbf{u}\in \mathbb {R}^ d$, the data set
      $$
      (\mathbf{u}^ T \mathrm{{\boldsymbol X}}_1) \mathbf{u}, (\mathbf{u}^ T \mathrm{{\boldsymbol X}}_2) \mathbf{u}\ldots , (\mathbf{u}^ T \mathrm{{\boldsymbol X}}_ n) \mathbf{u}
      $$
      is formed by projecting the data points $\mathrm{{\boldsymbol X}}_1, \ldots , \mathrm{{\boldsymbol X}}_ n$ onto the line spanned by the vector $\mathbf{u}$. Note that for all $i$, the number $\mathbf{u}^ T \mathrm{{\boldsymbol X}}_ i$ denotes the distance from the origin of the projection of $\mathrm{{\boldsymbol X}}_ i$ onto the line in the direction of $\mathbf{u}$. 

      Since $\mathbf{u}^*$ maximizes the empirical variance $\mathbf{u}^ T S \mathbf{u}$ of $\mathbf{u}^ T \mathrm{{\boldsymbol X}}_1, \ldots , \mathbf{u}^ T \mathrm{{\boldsymbol X}}_ n$ over all unit vectors $\mathbf{u}$, we expect the points
      $$
      (\mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_1) \mathrm{{\boldsymbol u^*}} , (\mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_2) \mathrm{{\boldsymbol u^*}} \ldots , (\mathrm{{\boldsymbol u^*}}^ T \mathrm{{\boldsymbol X}}_ n) \mathrm{{\boldsymbol u^*}}
      $$
      to be very spread out if the $\mathbf{u}^ T S \mathbf{u}$ is very large.


## 4. The Decomposition Theorem for Symmetric Matrices

1. **Orthogonal matrices**

   A matrix $P \in \mathbb {R}^{d \times d}$ is orthogonal (sometimes referred to as a rotation matrix) if $P P^ T = P^ T P = I_ d$. Suppose that 
   $$
   P = \begin{pmatrix} v_1 &  v_2 &  \cdots &  v_ d \\ \end{pmatrix}
   $$
   where $v_1, v_2, \ldots , v_ d \in \mathbb {R}^ d$ are column vectors.

   The $i$-th row of $P^T$ is $v_ i^ T$, and the $j$-th row of $P$ is $v_ j^ T$. By matrix multiplication and the orthogonal property
   $$
   (P^ T P)_{ij} = v_ i \cdot v_ j = (I_ d)_{ij}
   $$
   In particular, if $i = j$, then
   $$
   (P^ T P)_{ii} = v_ i \cdot v_ i = \sum _{j = 1}^ d (v_ i^ j)^2 = (I_ d)_{ii} = 1.
   $$
   i.e. $ \sum _{i = 1}^ d (v_1^ i)^2 = 1$.

   If $i \neq j$, then
   $$
   (P^ T P)_{ij} = v_ i \cdot v_ j = \sum _{k = 1}^ d v_ i^ k v_ j^ k = (I_ d)_{ij} = 0
   $$
   We can summarize above equation as
   $$
   (P^ T P)_{ij} = v_ i \cdot v_ j = (I_ d)_{ij} = \begin{cases}  0 \quad \text {if} \,  \,  i \neq j\\ 1 \quad \text {if} \,  \,  i = j. \end{cases}
   $$
   showing that **the columns of $P$ are mutually orthogonal unit vectors.**

   This also holds true for the rows of $P$, since we can use $P P^ T = I_ d$ , and follow the same procedure as above. Let $w_1, \ldots , w_ d$ denote the columns of $P^T$, we can get
   $$
   (P P^ T)_{ij} = w_ i \cdot w_ j = (I_ d)_{ij} = \begin{cases}  0 \quad \text {if} \,  \,  i \neq j\\ 1 \quad \text {if} \,  \,  i = j. \end{cases}
   $$
   showing that **the rows of $P$ are also mutually orthogonal unit vectors**.

2. **The Decomposition Theorem of a Symmetric Matrix**

   Suppose that $A \in \mathbb {R}^{d \times d}$ is a **symmetric** matrix. The decomposition theorem states that for all symmetric matrices $A$ with **real** entries
   $$
   A = PDP^T
   $$
   where $P \in \R^{d \times d}$ is an orthogonal matrix (i.e. $P^ T P = P P^ T = I_ d$), 

   * The columns of $P$ are unit vectors
   * The columns of $P$ are eigenvectors of $A$
   * The dot product of any two different columns of $P$ is 0
   * The rows of $P$ are unit vectors

   and $D \in \R^{d \times d}$ is a diagonal matrix.

   * The diagonal entries of $D$ are the eigenvalues of $A$
   * The first diagonal element of $D$ (i.e. in the top left corner) is the eigenvalue corresponding to the eigenvector which is in the first (i.e. leftmost) column of $P$.

   Let $\mathrm{{\boldsymbol v}}_1, \ldots , \mathrm{{\boldsymbol v}}_ d$ denote the columns of $P$, we have
   $$
   A \mathrm{{\boldsymbol v}}_ i = P D P^ T \mathrm{{\boldsymbol v}}_ i = \lambda _ i \mathrm{{\boldsymbol v}}_ i
   $$

3. Properties of **Covariance Matrices**

   Let $\Sigma$ denote a covariance matrix for some random vector $\mathrm{{\boldsymbol X}} \in \mathbb {R}^ d$ (Assume that $\mathbf E[\left\|  \mathrm{{\boldsymbol X}} \right\| _2^2] < \infty$), $\Sigma$ is **symmetric** and **positive semidefinite**.

   * The covariance matrix of a random vector is symmetric because the $i,j$-th entry is given by $\textsf{Cov}(\mathrm{{\boldsymbol X}}^ i, \mathrm{{\boldsymbol X}}^ j)$ and it is true that $\textsf{Cov}(\mathrm{{\boldsymbol X}}^ i, \mathrm{{\boldsymbol X}}^ j) = \textsf{Cov}(\mathrm{{\boldsymbol X}}^ j, \mathrm{{\boldsymbol X}}^ i)$.
   * The covariance matrix of a random vector is positive semidefinite. Because for all $\mathbf{u} \in \R^d$ it is true that $\mathbf{u}^ T \Sigma \mathbf{u}= \textsf{Var}( \mathbf{u}^ T \mathrm{{\boldsymbol X}} ) \geq 0$. The inequality follows because the variance of a random variable is always **non-negative**. Therefore by definition $\Sigma$ is positive semidefinite.

   Note that most positive semidefinite matrices are not orthogonal, and most orthogonal matrices are not positive semidefinite. 

   Also note that in **PCA**, the general strategy is to **diagonalize** the (empirical covariance) matrix and select the eigenvectors whose eigenvalues are the largest as axes on which to visualize our data set or point cloud.