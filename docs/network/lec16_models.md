# Graphical Models

There are topics.

## 1. Erdos-Renyi model

$G(n,p)$, where each edge between the $n$ nodes is formed with probability $p \in [0,1]$ independently of every other edge.

* Expected number of edges: $\mathbb{E}[\text{number of edges}] = {n \choose 2} p$

* Expected degree of a node: $\mathbb{E}[k_i] = (n-1)p$

* Degree distribution is $\text{Binomial} (n-1,p)$

  $\mathbb{P}(k) = {n-1 \choose k} p^k (1-p)^{n-1-k}$

* Degree distribution does not follow a power law

  approximation of binomial distribution by **Poisson distribution**
  $$
  \mathbb{P}(k) = \frac{e^{-\lambda}\lambda^k}{k!} \quad \text{where } \lambda = (n-1)p\\
  \log \mathbb{P}(k) = -\lambda + k \log(\lambda) - \log(k!) \approx - k \log (k) - k + k \log(\lambda)  \approx - k \log(k)
  $$
  **Power law degree distribution**
  $$
  \log \mathbb{P}(k) \approx - \lambda \log(k)
  $$
  The **exponent** of the power law degree distribution is always between 2 and 3 no matter what kind of network it is.

* The average **geodesic** does scale logarithmically with the number of nodes (not linearly).
* **Small-world** property, for a constant degree $c$, one can show that **diameter** is $\log(c)$.
* **Locally tree-like**, i.e. few triangles

* Problems:

  * In binomial distribution $\log \mathbb{P}(k) \approx - k \log(k)$ just dies off very quickly compared to power law degree distribution, so we have too few high-degree nodes, which is in fact not the case in reality, so this model does not fit the data well.

    ![binomial-model](../assets/images/binomial-model.png)

  * Clustering coefficient (which is the fraction of triangles among all connected triples) is 
    $$
    p \approx {c \over n} \quad \text{where c is mean degree, n is |nodes|}
    $$
    which is small and sparse, but in reality the coefficient is usually quite larger.

  

