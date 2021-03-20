# Likelihood Ratio Test and Multiple Hypothesis Testing

There are topics and 1 exercise.

## 1. Likelihood Ratio Test

**Model**: $X \sim p(x, \theta)$, parametric model with parameter $\theta$.

**Test**: $H_0: \theta \in \Theta_0$ versus $H_A: \theta \in \Theta_A$, where $\Theta_0$ and $\Theta_A$ are disjoint subsets: $\Theta_0 \cap \Theta_A = \emptyset$ of a parameter space $\Theta = \Theta_0 \cup \Theta_A$.

**Likelihood ratio**: $L(x) = \frac{max_{\theta \in \Theta_0} p(x;\theta)}{max_{\theta \in \Theta} p(x;\theta)}$, where $\Theta = \Theta_0 \cup \Theta_A$.

* $p(x;\theta)$ is the probability / density of observing the data $x$.
* the parameter $\hat{\theta}$ that maximizes $p(x;\theta)$ is called the **maximum likelihood estimator (MLE)**.
* $0 \leq L(x) \leq 1$ since
  * the numerator: MLE of $\theta$ is selected from a range of $\theta$ from null model
  * the denominator: MLE of $\theta$ is selected from a wider range of $\theta$ from the union of null model and alternative model.
* The likelihood ratio test statistic is a valid statistic for the hypothesis test. It can distinguish between the null hypothesis and the alternative.
  * $L(x)<< 1$ if $\theta \in \Theta_A$ [ so when the likelihood is too small, we would like to reject the null hypothesis ]
  * $L(x) \approx 1$ if $\theta \in \Theta_0$

**Likelihood ratio test**: Reject $H_0$ if $L(x) < \eta$, where $\eta$ is chosen such that $P_{H_0}(L(x) \leq \eta) = \alpha$. This means that we choose the cutoff value $\eta$ so that the probability of rejecting the null when the null is true (Type I error) is equal to $\alpha$.

* **Neyman-Pearson Lemma**: Likelihood ratio test is the most powerful among all level $\alpha$ tests for testing $H_0: \theta = \theta_0$ versus $H_A: \theta = \theta_A$. Among all tests testing the same simple hypothesis and at the same significance level, the likelihood ratio test gives the largest probability of rejecting the null when indeed the alternate is true.

## 2. Asymptotic Likelihood Ratio Test

In general $L(x)$ does not have an easily computable null distribution, i.e. it is difficult to determine $\eta$. So we defined $\Lambda(x)$ as (negative twice) the logarithm of the likelihood ratio $L(x)$.

**Likelihood ratio statistic**:
$$
\Lambda(x) = -2log(L(x)) = -2log\frac{\max_{\theta \in \Theta_0}p(x;\theta)}{\max_{\theta \in \Theta}p(x;\theta)}
$$

* $0 \leq \Lambda(x) \leq \infty$

* reject $H_0$ if $\Lambda(x)$ is too large

Equivalently, in **MLE**
$$
L(x) = \frac{ p\left(x;\hat{\theta }_{\text {MLE}}^{\text {constrained}}\right)}{p\left(x;\hat{\theta }_{\text {MLE}}\right)}
$$
where $\hat{\theta}_{MLE}$ is the maximum likelihood estimator of $\theta$ and $\hat{\theta}^{constrained}_{MLE}$ is the constrained maximum likelihood estimator of $\theta$ within $\Theta_0$.

## 3. Distribution of Likelihood Ratio Test Statistics

**Wilks Theorem**:  when the sample size is large, the distribution of $\Lambda$ under $H_0$ approaches a $\chi^2$ distribution with $d$ degrees of freedom.
$$
\Lambda(x) \overset {n\to \infty }{\longrightarrow } \chi _ d^2, \text{ where } d = dim(\Theta) - dim(\Theta_0) > 0
$$
> #### Exercise 2:
>
> Determine the dimension $d$ for the following models:
> $$
> H_0: \pi_{treatment} = \pi_{control};\\ H_A: \pi_{treatment} \neq \pi_{control}
> $$
>
> > **Answer**:  $d = 1$
>
> > **Solution**: The $dim(H_0) = 1, dim(H_A) = 2$, since in a plane where $y = \pi_{treatment}, x = \pi_{control}$, $H_0$ is a line ($y = x$), while $H_A$ is anywhere else.







