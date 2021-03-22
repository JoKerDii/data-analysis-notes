# Hypothesis Testing

There are 7 topics and 0 exercise.

## 1. Hypergeometric probability distribution

The **hypergeometric distribution** is a discrete distribution based on the following probability problem:

“Suppose there are $N$ balls in a bowl, $K$ of which are red and the remaining $N−K$ of which are blue. From the bowl, $n$ balls are drawn without replacement. What is the probability that among the $n$ balls drawn, exactly $x$ are red?"

The solution to this problem is given by the following PMF:
$$
\begin{aligned}
\mathbb {P}(X = x)& = \frac{\left(\text {Number of ways to choose } x \text { out of } K \text { red balls} \right) \cdot \left(\text {Number of ways to choose } n-x \text { out of } N-K \text { blue balls } \right)}{\text {Number of ways to choose } n \text { balls out of} N}\\
& = \frac{\dbinom {K}{x}\dbinom {N-K}{n-x}}{\dbinom {N}{n}}.
\end{aligned}
$$
This PMF defines the hypergeometric distribution **Hypergeometric** $(N,K,n)$ with the three parameters:

- $N$, size of population (number of balls in bowl)
- $K$, size of sub-population of interest (number of red balls in bowl)
- $n$, the number of targeted outcomes (total number of balls drawn). [specially needed for hypergeometric model]

## 2. Fisher's Exact Test p-value

**Fisher's Exact Test** provides a method based on the hypergeometric distribution to test hypotheses of the form:

- H0: $π_{treatment}$= $π_{control}$, i.e. treatment has no effect on the rate of occurrence of a targeted outcome
- HA: $π_{treatment}$< $π_{control}$, i.e. treatment lowers (or raises, or changes) the rate of occurrence of a targeted outcome.

We define the test statistic $T$ to be the **number of targeted outcomes in the treatment group**. Under the null hypothesis that the treatment has no effect, $T$ follows a **hypergeometric distribution** **hypergeometric**$(N,K,n)$

Recall the **p-value** is defined to be the probability that we obtain an observation as extreme or more extreme than the one observed,in the direction of the alternative hypothesis, under the null hypothesis. In a **Fisher's exact test**, this corresponds to the **probability under a tail of the hypergeometric PMF**.

**Advantage**: Fisher's exact test does not assume knowledge about the true probability of the targeted outcomes (e.g. death due to breast cancer) in the control population.

## 3. Paired tests and continuous data

A **paired test design** involves taking multiple samples from an individual, one corresponding to a control situation and the other to a treatment situation. In a paired test, it is the difference between the observed values in the treatment and the control situations, i.e. $Y_i:=X_{i,treatment}−X_{i, control}$ that will be considered. A null hypothesis that states that the treatment has no effect is equivalent to claiming that $E[Y_i]=0$. 

**z-test** is a popular approach to hypothesis testing on continuous data. 

## 4. Central limit theorem (CLT) and the z-test statistic

Suppose that we have observations $X_1,…,X_n$, which are **independent and identically distributed** based on a probability model. Under a few regularity assumptions (such as the model having a finite second moment), the distribution of the sample mean $\bar{X}$ will approximate a **normal distribution** when sample size becomes sufficiently large (typically $n≥30$).

The **central limit theorem (CLT)** states that: When sampling random variables $X_1,…,X_n$ from a population with mean $μ$ and variance $σ^2$, $\bar{X}$ is approximately normally distributed with mean $μ$ and variance $σ^2/n$ when $n$ is large:
$$
\overline{X} = \frac{X_1 + X_2 + \ldots + X_ n}{n} \sim \mathcal{N}\left(\mu , \frac{\sigma ^2}{n}\right) \qquad \text {for } n \text { large} .
$$
Hence we can define a test statistic $z = \frac{\overline{X} - \mu }{\sigma /\sqrt{n}}$, which approximately follows a **standard normal distribution** when $n$ is large.
$$
z = \frac{\bar{X} - \mu }{\sigma /\sqrt{n}} \sim \mathcal{N}(0,1).
$$
The test statistic $z$ is called an (approximate) **pivotal quantity**, since its (approximate) distribution does not depend on the parameters $μ$ or $σ$. We can use the **CDF** of a pivotal quantity to compute the p-value, and compare the p-value with with $α$ the significance level to decide whether to reject the null hypothesis $H_0$.

## 5. t-statistic and t-distribution

The **t-test** is a statistical method to test a hypothesis without knowing the population standard deviation $σ$. Instead, we can estimate $σ$ using the **sample standard deviation formula**, based on the observations $X_1,X_2,…,X_n$, where $\bar{X}$ is the sample mean. 


Under the assumption that $X_1,X_2,…,X_n ∼ \text{ i.i.d } N(μ,σ2)$ for any pair of parameters $(μ,σ^2)$. The **t-statistics** is defined as
$$
T = \frac{\bar{X_n} - \mu}{\hat{\sigma}/\sqrt{n}}
$$
where we have
$$
\overline{X_ n} = \frac{1}{n}\sum _{i=1}^ n X_ i,\\
\hat{\sigma } = \sqrt{ \frac{1}{n-1}\sum _{i = 1}^{n} (X_ i-\overline{X})^2}.
$$
$T$ is a ***pivotal* statistic**. Its distribution is called a **$t$-distribution** and is parameterized by the number of ***degrees of freedom***. In this case, $T∼t_{n−1}$, the $t$ distribution with $n−1$ degrees of freedom.
$$
\frac{\bar{X_n}}{\hat{\sigma}/\sqrt{n}} \sim t_{n-1}
$$
Some properties of **t-distribution**: Let $T \sim t_n$. Then

* The $t$-distribution with $n$ degree of freedom:   $\sum^n_{i=1}X_i^2 \sim X_n^2$; $t_n \sim \frac{\mathcal{N}(0,1)}{\sqrt{X_n^2/n}}$ where $X_1, ..., X_n \sim \mathcal{N}(0,1)$. $X_n^2 \sim \chi^2$.

* $t_n (n \rightarrow \infty) \rightarrow \mathcal{N}(0,1)$

* $E(T) = 0$, $Var(T) = \frac{n}{n-2} > 1$

  $\rightarrow$ estimating $\sigma$ introduces uncertainty; more weight in tails.

### Proof: t-statistic follows t-distribution with n-1 degree of freedom

We first specify $Y$ and $Z$ such that 
$$
T = \frac{Y}{\sqrt{Z/n}}
$$
where $Y \sim \mathcal{N}(0,1)$ has a **standard normal** distribution and has $Y = \frac{\overline{X_ n} - \mu }{\sigma / \sqrt{n}}$. $Z \sim \chi_n^2$ is a **chi-squared** distribution with $n$ degrees of freedom.

We can solve for $Z$ by equating the expressions for the $t$-statistic and the $t_{n-1}$ distribution:
$$
T = \frac{Y}{\sqrt{Z/(n-1)}} = \frac{\overline{X_ n} - \mu }{\sqrt{\hat{\sigma }^2 / n}}.
$$
Hence, we derive the corresponding $Z$ as: 
$$
\sqrt{\frac{Z}{n-1}} = \frac{Y\sqrt{\hat{\sigma }^2 / n}}{\overline{X_ n} - \mu } = \frac{\sqrt{\hat{\sigma }^2 / n}}{\sigma / \sqrt{n}} \Longrightarrow Z = (n-1)\frac{\hat{\sigma }^2}{\sigma ^2} = \frac{1}{\sigma ^2}\sum _{i=1}^ n (X_ i - \overline{X_ n})^2.
$$
Note that $Y$ only depends on $\overline{X_ n}$. Hence, it suffices to show that

* $\frac{1}{\sigma ^2}\sum _{i=1}^ n (X_ i - \overline{X_ n})^2$ has a $\chi_{n-1}^2$ distribution
* $\overline{X_ n}$ and $\sum _{i=1}^ n (X_ i - \overline{X_ n})^2$ are independent



To show that $\frac{1}{\sigma ^2}\sum _{i=1}^ n (X_ i - \overline{X_ n})^2$ has a $\chi_{n-1}^2$ distribution, we construct a $\chi_n^2$ from $\frac{X_ i - \mu }{\sigma }$ as it has a $\mathcal{N}(0,1)$ distribution.
$$
W = \sum _{i=1}^ n \left(\frac{X_ i - \mu }{\sigma }\right)^2 \sim \chi ^2_ n.
$$
By some algebra manipulation we got
$$
W = \sum _{i=1}^ n \left(\frac{X_ i - \mu }{\sigma }\right)^2 = \frac{1}{\sigma ^2} \sum _{i=1}^ n (X_ i - \overline{X_ n})^2 + \frac{n}{\sigma ^2} (\overline{X_ n} - \mu )^2.
$$
We now reason using **multivariate Gaussians**, as $X_1,…,X_n$ are i.i.d. Gaussians. Therefore
$$
\overline{X_n}∼\mathcal{N}(μ,σ^2/n)\\
\frac{n}{σ^2}(\overline{X_n}−μ)^2∼\chi_1^2\\
$$
More generally, we can construct variables out of linear combinations of $X_1,…,X_n$. If we have a pair of such variables, they will be **jointly Gaussian** so they are **independent** iff they have **zero covariance**.

To show that $X_ i - \overline{X_ n}$ and $\overline{X_ n}$ are **independent**:
$$
\textsf{Cov}(X_ i, \overline{X_ n}) = \textsf{Cov}(X_ i, \frac{1}{n}X_ i) = \frac{1}{n}\sigma ^2\\
\textsf{Cov}(\overline{X_ n}, \overline{X_ n}) = \sum _{i=1}^ n \textsf{Cov}(\frac{1}{n} X_ i, \frac{1}{n} X_ i) = n\left(\frac{1}{n^2} \sigma ^2 \right) = \frac{1}{n}\sigma ^2
$$
Hence we got $\textsf{Cov}(X_ i - \overline{X_ n}, \overline{X_ n}) = 0$, and so  $X_ i - \overline{X_ n}$ and $\overline{X_ n}$ are independent.

Since $\chi_n^2 = \chi_1^2 + \chi_{n-1}^2$, we have
$$
\frac{1}{\sigma ^2} \sum _{i=1}^ n (X_ i - \overline{X_ n})^2 \sim \chi_{n-1}^2
$$

## 6. Testing the observations are from $\mathcal{N}(\mu,\sigma^2)$

* **qq-plot** (quantile-quantile plot): qualitative method
* **Kolmogorov-Smirnov**: quantitative method

Alternative if your sample does not come from normal distribution: **Wilcoxon signed rank test**. This test does not make an assumption of normality.

* Model: $X_1, ..., X_n \sim F$ symmetric around a mean $\mu$

* Test statistic: $W = \sum^n_{i=1}sgn(X_i - \mu)R_i$, where $R_i$ is rank of $|X_i - \mu|$

* One can show that this test statistic is asymptotically $n \rightarrow \infty$ normally distributed

  $\implies$ build hypothesis test based on **asymptotic distribution**

Note that all hypothesis tests have unpaired version, but unpaired tests are usually less powerful.

## 7. Confidence Interval

The confidence interval (CI) at level $1-\alpha$ is defined as
$$
I(X) = \{\theta | H_0: \mu = \theta \text{ can not be rejected at significance level } \alpha \text{ , given the data } X\}
$$
Where $I(X)$ is a **random quantity**, which depends on the observations, and is often computed based on **2-sided testing** and **normal approximation**.

Therefore, the CI is an interval that is a function of the observations in that

- it is **centered around the sample mean**
- its width is **proportional to the standard error**.

An alternative interpretation of confidence interval : **Confidence interval contains true parameter $\mu$ with probability** $1-\alpha$, i.e.
$$
P_\mu(\mu \in I(X)) = 1-\alpha
$$
Therefore, it is also parameterized by the significance level $\alpha$ . This “**probability**" means that if we sample the dataset numerous times and calculate intervals for each time, the probability that $μ$ is in the proposed range (resulting intervals) is $1-\alpha$.

The probability of **z-test statistic** can take between $−Φ^{−1}(1−α/2)$ and $Φ^{−1}(1−α/2)$, where $Φ$ is the **Cumulative Distribution Function** of the standard normal distribution, $α$ is the significant level.
$$
P ({-\Phi ^{-1}_{(1-\alpha /2)}} \leq {\frac{\bar{X} - \mu }{\sigma /\sqrt{n}}} \leq {\Phi ^{-1}_{(1-\alpha /2)}})= 1 - \alpha\\
P (\bar{X} -\frac{\sigma }{\sqrt{n}}\Phi ^{-1}_{(1-\alpha /2)} \leq \mu \leq \bar{X} +\frac{\sigma }{\sqrt{n}}\Phi ^{-1}_{(1-\alpha /2)}) = 1 - \alpha\\
$$
Hence the CI for true parameter $mu$ with probability $1-\alpha$ is 
$$
\bar{X} \pm \frac{\sigma }{\sqrt{n}}\Phi ^{-1}_{(1-\alpha /2)}
$$
Note that if the sample size increases, the length of the confidence interval will decrease.

-----

### A summary of the tests

![img](../assets/images/lec2Sumary.png)