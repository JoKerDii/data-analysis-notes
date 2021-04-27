# Clustering

There are 6 topics and 1 exercise.

## 1. k-means Algorithm

* Suppose we have a fixed cluster number $K$. Clusters are obtained by minimizing k-mean loss function, given by **within-groups sum of squares (WGSS)** :
  $$
  W(C) = \sum^K_{k=1}\sum_{C(x^{(i)}) = k}\sum_{C(x^{(j)}) = k}d(x^{(i)},x^{(j)})^2
  $$
  This is equivalent to maximizing the **between-groups sum of squares (BGSS)**:
  $$
  B(C) = \sum^K_{k=1}\sum_{C(x^{(i)}) = k}\sum_{C(x^{(j)}) \neq k}d(x^{(i)},x^{(j)})^2
  $$
  because the sum of them is a constant
  $$
  \sum^K_{k=1}\sum_{C(x^{(i)}) = k}(\sum_{C(x^{(j)}) = k}d(x^{(i)},x^{(j)})^2+\sum_{C(x^{(j)}) \neq k}d(x^{(i)},x^{(j)})^2) =\sum^K_{k=1}\sum_{C(x^{(i)}) = k}(d(x^{(i)},x^{(j)})^2)
  $$
  where the distance is measured by Euclidean distance
  $$
  d(x^{(i)},x^{(j)})^2 = \|x^{(i)} - x^{(j)}\|^2_2
  $$
  The WGSS can be written as follows in terms of $\mathrm{{\boldsymbol \mu }}_ k=\sum _{\mathbf{x}^{(i)}\in C_ k} \mathbf{x}^{(i)} /n_ k$ of each cluster $C_k$:
  $$
  \begin{aligned}
  W(C) &= \sum _{k=1}^{K} \sum _{\mathbf{x}^{(i)},\mathbf{x}^{(j)}\in C_ k} \left\|  \mathbf{x}^{(i)}-\mathbf{x}^{(j)} \right\| ^2\\
  &= \sum _{k=1}^{K} \sum _{\mathbf{x}^{(i)},\mathbf{x}^{(j)}\in C_ k}\left\|  (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k)-(\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2\\
  &=\sum _{k=1}^{K} \sum _{\mathbf{x}^{(i)},\mathbf{x}^{(j)}\in C_ k}\left(\left\|  (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2+\left\|  \mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2- 2\left(\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k\right)\cdot \left(\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_ k\right) \right)\\
  &=\sum _{k=1}^{K} \left(n_ k\sum _{\mathbf{x}^{(i)}\in C_ k}\left\|  (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2+n_ k \sum _{\mathbf{x}^{(j)}\in C_ k}\left\|  \mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2- 2\left(\sum _{\mathbf{x}^{(i)}\in C_ k}(\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k)\right)\cdot \left(\sum _{\mathbf{x}^{(j)}\in C_ k}(\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_ k)\right) \right)\\
  &=2 \sum _{k=1}^{K} n_ k\sum _{\mathbf{x}^{(i)}\in C_ k}\left\|  (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k) \right\| ^2\qquad \text {since } \sum _{\mathbf{x}^{(i)}\in C_ k}(\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_ k)= \sum _{\mathbf{x}^{(i)}\in C_ k}\mathbf{x}^{(i)} - n_ k \mathrm{{\boldsymbol \mu }}_ k\, =\, \mathbf{0}.\\
  \end{aligned}
  $$
  This algorithm leads to spherical shaped clusters of similar radii.

* Steps of k-means:

  1. Randomly initialize $K$ means $\{ \mathrm{{\boldsymbol \mu }}_ k\} _{k=1,\ldots K}$.
  2. Assign each point to the closest $\mathrm{{\boldsymbol \mu }}_ k$
  3. Update all $\mathrm{{\boldsymbol \mu }}_ k$ to be the new average of all point in each cluster
  4. Repeat 2 and 3 to minimize the k-means loss function until converge.

* Solution to computational infeasibility

  * Greedy algorithm.
  * Random restarts and conduct multiple runs to avoid local optima. (Because k-mean can converge to a local minimum depending on the initialization of cluster centroids)

* Choosing the number of clusters (diagnostics)

  * "elbow" method: Run k-means for increasing number of groups $K$, plot WGSS versus $K$, and choose the $K$ after the last big drop of the curve.
  * Plot Silhouette scores, the data point index on y-axis versus the silhouette coefficient values on x-axis (Positive Silhouette scores mean the point is clustered very well.)

## 2. Partitioning around medoids (PAM)

Partitioning around medoids (PAM) addresses the limitation of k-means that

* k-means is sensitive to outliers
* k-means's cluster centroids are not necessarily data points

## 3. Gaussian Mixture Models (GMM) and Expectation-Maximization (EM) Algorithm

GMM addresses the limitation of K-means that

* cluster assignment by GMM are soft assignment based on probabilities 
* The shape of cluster in GMM can be elliptical rather than spherical

Consider the GMM of $K$ Gaussians
$$
\mathbf{P}(\mathbf{X}) = \sum _{k=1}^ K p_ k \mathbf{P}(\mathbf{X}| \text {cluster } k)\\
\text{ where }\quad \mathbf{X}| \text {cluster } k \sim \mathcal{N}(\mathrm{{\boldsymbol \mu }}_ k, \Sigma _ k).
$$
This mixture has parameters $\theta =\{ p_1 , \ldots p_ k , \mathrm{{\boldsymbol \mu }}_1,\ldots , \mathrm{{\boldsymbol \mu }}_ K, \Sigma _1, \ldots , \Sigma _ K\}$. Given $n$ data points $\mathbf{x}^{(1)}, \dots , \mathbf{x}^{(n)}$ in $\R^d$, our goal is to set our parameters $\theta$ to maximize the data log-likelihood.
$$
 \ell (\mathbf{x}^{(1)},\dots ,\mathbf{x}^{(n)} ; \theta ) =  \log \prod _{i=1}^{n} \mathbf{P}(\mathbf{x}^{(i)};\theta ) \, =\, \sum _{i=1}^{n} \log \left[\sum _{k=1}^ K p_ k \mathbf{P}(\mathbf{x}^{(i)}| \text {cluster } k;\theta ) \right]
$$
Since there is no closed-form solution to finding $\theta$ that maximizes this likelihood, EM algorithm can iteratively find a local optima $\hat{\theta}$.

* **E-step**: find the posterior using **Bayes' rule**
  $$
  p(k \mid i)=\mathbf{P}(\text {cluster } k | \mathbf{x}^{(i)}; \theta ) = \frac{p_ k \, \mathbf{P}(\mathbf{x}^{(i)}|\text {cluster } k; \theta ) }{\mathbf{P}(\mathbf{x}^{(i)} ; \theta )}\, =\, \frac{p_ k \, \mathcal{N}\left(\mathbf{x}^{(i)}; \mu ^{(k)},\Sigma _ k \right)}{\sum _{j=1}^ K p_ k \, \mathcal{N}\left(\mathbf{x}^{(i)}; \mu ^{(j)},\Sigma _ j \right)}
  $$

* **M-step**: maximize the **expected log likelihood** 
  $$
  \begin{aligned}
  \tilde{\ell }(\mathbf{x}^{(1)},\dots ,\mathbf{x}^{(n)} ; \theta ) &=  \sum _{i=1}^{n} \left[\sum _{k = 1}^ K p(k \mid i) \log \left( \frac{\mathbf{P}(\mathbf{x}^{(i)},\text {cluster } k ; \theta )}{p(k \mid i)} \right)\right]\\
  &=\sum _{i=1}^{n} \left[\sum _{k = 1}^ K p(k \mid i)\log \left( \frac{p_ k \, \mathcal{N}\left(\mathbf{x}^{(i)}; \mu ^{(k)},\Sigma _ k \right)}{p(k \mid i)} \right)\right]\\
  &=\sum _{i=1}^{n} \log \left[\sum _{k = 1}^ K \mathbf{P}\left( \mathbf{x}^{(i)},\text {cluster } k ; \theta \right)\right].
  \end{aligned}
  $$

* E and M steps are repeated iteratively until there is no noticeable change in the actual likelihood.

When the covariance matrix is $\Sigma _ k= \sigma _ k^2 \mathbf{I}$, after taking derivative and set to zero, the parameters that maximize the above expected log likelihood function are
$$
\begin{aligned}
\widehat{\mathrm{{\boldsymbol \mu }}^{(k)}} &= \frac{\sum _{i = 1}^ n \mathbf{x}^{(i)} p (k \mid i) }{\sum _{i=1}^ n p(k \mid i)}\\
  \widehat{p_ k} &= \frac{1}{n}\sum _{i = 1}^ n p(k \mid i)\\
  \widehat{\sigma _ k^2} &= \frac{\sum _{i = 1}^ n p(k \mid i) \|  \mathbf{x}^{(i)} - \widehat{\mu ^{(k)}} \| ^2}{d \sum _{i = 1}^ n p(k \mid i)}
  \end{aligned}
$$
Initialization of parameter $\theta$ before EM steps

* random initialization
* apply k-means to find cluster centers, and use global variance of the dataset as the initial variance of all clusters

## 4. Hierarchical Clustering

Approaches: 

* Agglomerative clustering: bottom-up 
* Divisive clustering: top-down (will not cover)

Advantage:

* All numbers of clusters can be solved at once, allowing a desired number of clusters can be chosen later.

Common used distances:

* Euclidean distance (i.e. $l_2$ norm)
  $$
  d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})=\sqrt{(x^{(i)}_1-x^{(j)}_1)^{{\color{blue}{2}} }+ (x^{(i)}_2-x^{(j)}_2)^{{\color{blue}{2}} }+\cdots +(x^{(i)}_ p-x^{(j)}_ p)^{{\color{blue}{2}} }}\qquad (\mathbf{x}^{(i)}\in \mathbb {R}^ p)
  $$

* Manhattan distance (i.e. $l_1$ norm)
  $$
  d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})=\left| x^{(i)}_1-x^{(j)}_1 \right|+ \left| x^{(i)}_2-x^{(j)}_2 \right|+\cdots +\left| x^{(i)}_ p-x^{(j)}_ p \right|\qquad (\mathbf{x}^{(i)}\in \mathbb {R}^ p)
  $$

* Maximum distance (i.e. $l_{\infty}$ norm)
  $$
  d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})=\max _{k=1,\ldots ,p} \left| x^{(i)}_ k-x^{(j)}_ k \right|\qquad (\mathbf{x}^{(i)}\in \mathbb {R}^ p)
  $$

* Dissimilarity measures that satisfy
  $$
  \begin{aligned}
  d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) &\geq 0 \text{ (positivity) },\\ d(\mathbf{x}^{(i)},\mathbf{x}^{(i)}) & = 0, \\ d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) & = d(\mathbf{x}^{(j)},\mathbf{x}^{(i)}) \text{ (symmetry) }
  \end{aligned}
  $$

Dissimilarity measures between clusters:

* Minimum distance (a.k.a. single linkage)
  $$
  d(C_1, C_2) =  \min _{\mathbf{x}^{(i)}\in C_1, \mathbf{x}^{(j)}\in C_2} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})
  $$

  * Intuitively, we can easily observe a long narrow cluster, because as long as there is one point in one cluster that is close enough to another point in another cluster, then the two clusters can be merged (regardless of how far away the other points in the two clusters are from each other)

* Maximum distance (a.k.a. complete linkage)
  $$
  d(C_1, C_2) = \max _{\mathbf{x}^{(i)}\in C_1, \mathbf{x}^{(j)}\in C_2} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}).
  $$

  * Intuitively, we can observe small clusters formed, because the distance between points in other clusters to all the point in this small cluster has to be bounded.

* Average distance (a.k.a. average linkage)
  $$
  d(C_1, C_2) =\frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})
  $$

  * Intuitively, merging depends on the distances between all points in the two clusters, but the pairs of points that are close together balance out the pairs of points that are far apart.


> #### Exercise 24
>
> Is the average squared distance between two clusters equal to the squared distance of the difference in the two centroids? Hint: Simplify $ \frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2} \left\|  \mathbf{x}^{(i)}-\mathbf{x}^{(j)} \right\| _2^2$ where $C_1$ and $C_2$ are two different clusters
>
> > **Answer**: No
>
> > **Solution**: 
> > $$
> > \begin{aligned}
> > &\frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2} \left\|  \mathbf{x}^{(i)}-\mathbf{x}^{(j)} \right\| ^2 \\
> > &= \frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2} \left\|  (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1)+(\mathrm{{\boldsymbol \mu }}_1-\mathrm{{\boldsymbol \mu }}_2)-(\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2) \right\| ^2\\
> > &=\frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2}\left( \left\|  \mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1 \right\| ^2+\left\|  \mathrm{{\boldsymbol \mu }}_1-\mathrm{{\boldsymbol \mu }}_2 \right\| ^2+\left\|  \mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2 \right\| ^2+ 2(\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1)(\mathrm{{\boldsymbol \mu }}_1-\mathrm{{\boldsymbol \mu }}_2) -2(\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2)(\mathrm{{\boldsymbol \mu }}_1-\mathrm{{\boldsymbol \mu }}_2)- 2(\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1)\cdot (\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2)\right)\\
> > &= \frac{1}{n_1 n_2}\sum _{\mathbf{x}^{(i)}\in C_1}\sum _{\mathbf{x}^{(j)}\in C_2}\left( \left\|  \mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1 \right\| ^2+\left\|  \mathrm{{\boldsymbol \mu }}_1-\mathrm{{\boldsymbol \mu }}_2 \right\| ^2+\left\|  \mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2) \right\| ^2\right)
> >  \end{aligned}
> > $$
> > Since the cross terms all vanish $\sum _{\mathbf{x}^{(i)}\in C_1} (\mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1)=\sum _{\mathbf{x}^{(j)}\in C_2} (\mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2)=\mathbf{0}$
> >
> > Hence, the average squared distance is greater than the squared distance of the difference in centroids, and the extra terms $\sum _{\mathbf{x}^{(i)}\in C_1} \left\|  \mathbf{x}^{(i)}-\mathrm{{\boldsymbol \mu }}_1 \right\| ^2$ and $\sum _{\mathbf{x}^{(j)}\in C_2} \left\|  \mathbf{x}^{(j)}-\mathrm{{\boldsymbol \mu }}_2 \right\| ^2$ take into account the spread of the two clusters.

## 5. Density-based spatial clustering of applications with noise (DBSCAN)

**DBSCAN** aims to cluster together points that are close to each another in a dense region, and leave out points that are in low density regions.

We need to choose two parameters:

* $\epsilon$: distance between connected points, used to define **connected** points which are within a distance $\epsilon$.
* $k$: core strength, used to define the minimum number of points that the **core point** connects to.

Hence, DBSCAN gives clusters with core points connected at least $k$ other points. Points with no connection with others are not in any cluster. Points with no connection with core points are **outliers**.

## 6. Silhouette Plot

**Silhouette plot** measures the quality of cluster assignments, with Silhouette scores of each data point.

The **Silhouette score** of a data point $\mathbf{x}^{(i)}$ is defined to be 
$$
S(\mathbf{x}^{(i)}) = \frac{b(\mathbf{x}^{(i)})-a(\mathbf{x}^{(i)})}{\max \left(b(\mathbf{x}^{(i)}),a(\mathbf{x}^{(i)})\right)}
$$
where $a(\mathbf{x}^{(i)})$ is the **average "within group" distance or dissimilarity** from $\mathbf{x}^{(i)}$.
$$
a(\mathbf{x}^{(i)}) = \frac{1}{n_ i-1}\sum _{\mathbf{x}^{(j)}\in C_ i, j\neq i} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)}) \qquad (\mathbf{x}^{(i)}\in C_ i)
$$
and $b(\mathbf{x}^{(i)})$ is the **average distance or dissimilarity** from $\mathbf{x}^{(i)}$ to the closest other cluster.
$$
b(\mathbf{x}^{(i)}) = \min _{C_ k\neq C_ i} \frac{1}{n_ k}\sum _{\mathbf{x}^{(j)}\in C_ k} d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\qquad (\mathbf{x}^{(i)}\in C_ i)
$$
