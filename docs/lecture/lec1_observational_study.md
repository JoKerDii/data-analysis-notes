# Observational Studies and Experiments

There are 5 topics and 1 exercise.

## 1. Basics

**Randomized Controlled Trial (RCT)**: is an experimental design in which treatments are assigned at random.

* In an example of drug trial, patients for a particular disease are randomly assigned to the treatment or the control group. The patients in a treatment group receive the new drug being studied, while those in the control group receive a placebo drug.
* Modification of RCT: **stratification** allowing **subgroup analysis**, which is analyzing the treatment effect within a particular group.

**Confounder / confounding variables**: A confounder is a variable that influences both dependent and independent variables.

* Note that when estimating the direct relation between independent ($X$) and dependent ($Y$) variable from data, we have to block all other variables that associated $X$ and $Y$. However, if a variable associated between $X$ and $Y$ while it is a **descendent** of them, we should not condition on it.
* **Stratification** could be a remedy to confounding. A example is the confounder of demographic category.

**Control variables**: we select controls that will capture all possible sources of bias, factors that lead to both the treatment and outcome values.

* To account for the effect of controls, a common technique is **multivariate regression**.

## 2. Statistical Model - Bernoulli Model

**Bernoulli random variables** are used to model random experiments with only two possible outcomes. A Bernoulli random variable with parameter $p$ is a random variable that takes the value 1 with probability $p$ and the value 0 with probability $1−p$. The **Bernoulli distribution** is the discrete probability distribution of a Bernoulli random variable. Hence, we can write the Bernoulli **probability mass function (PMF)** as
$$
\begin{aligned}
f(x) & = \begin{cases}  p &  \text{if } x = 1 \\ 1-p &  \text{if } x = 0 \end{cases}\\
& = p^ x(1-p)^{1-x}\\
& = px + (1-p)(1-x)
\end{aligned}
$$
The Expectation and Variance is 
$$
\begin{aligned}
E(x) & = p\\
Var(x) & = p(1-p)\\
\end{aligned}
$$
In many applications of the **Bernoulli model** with multiple indicator variables, they are **independent and identically distributed (i.i.d.)**, meaning that the indicator variables are mutually independent and that they are all Bernoulli with the same parameter $p$. 

## 3. Statistical Model - Binomial Model

**Binomial random variable** with parameters $n$ (trials) and $p$ (probability) is defined as the sum of $n$ independent Bernoulli random variables, all with parameter $p$, where each Bernoulli trial represents a single indicator variable. The **Binomial distribution** models the special case of multiple Bernoulli random variables.

From the definition, we can compute the **probability mass function (PMF)** of a binomial random variable $Y$ with parameters $n$ and $p$ to be.
$$
f(Y = k) = \binom {n}{k} \cdot p^ k \cdot (1-p)^{n-k}.
$$

## 4. Statistical Model - Poisson Model

The **Poisson random variable** is based on taking the limit of a **binomial distribution** with a fixed mean $np$. As we take $n→∞$, the distribution converges to a fixed discrete PMF which we parameterize by $λ=np$. Indeed, we can compute the probability of $k$ successes, substitute $p=λ/n$, and then take the limit.
$$
\begin{aligned}
\lim _{n\to \infty } \binom {n}{k} p^ k (1-p)^{n-k} & = \lim _{n\to \infty } \frac{n!}{k!(n-k)!} p^ k (1-p)^{n-k}\\
& = \lim _{n\to \infty } \frac{n(n-1)\ldots (n-k+1)}{k!} \left(\frac{\lambda }{n}\right)^ k \left(1-\left(\frac{\lambda }{n}\right)\right)^{n-k}\\
& = \lim _{n\to \infty } \frac{n(n-1)\ldots (n-k+1)}{n^ k} \frac{n^ k}{k!} \frac{\lambda ^ k}{n^ k} \left(1-\left(\frac{\lambda }{n}\right)\right)^{n-k}\\
& = \lim _{n\to \infty } \left( \frac{n(n-1)\ldots (n-k+1)}{n^ k} \right) \frac{\lambda ^ k}{k!} \left(1-\left(\frac{\lambda }{n}\right)\right)^{n-k}\\
& = (1) \frac{\lambda ^ k}{k!} e^{-\lambda }\\
& = \frac{e^{-\lambda } \lambda ^ k}{k!}.
\end{aligned}
$$
Therefore, when data follows **binomial distribution** with large $n$ (number of trials) and small $p$ (probability of success), $Poisson(np)$ is a good approximation to $Binomial(n,p)$.

Another interpretation of the Poisson random variable is in terms of a random process called the **Poisson process**. This is defined as a process where events can occur at any time in continuous time, with an average rate given by the parameter $λ$ and satisfying the following conditions:

- Events occur independently of each other.
- The probability that an event occurs in a given length of time is constant.

## 5. Hypothesis Testing

1. **Determine a model**: $X \sim Bernoulli(\pi)$ or $Y \sim Poisson(\lambda)$.

2. **Determine a (mutually exclusive) null hypothesis and alternative**:

   Null Hypothesis ($H_0$) : claim that the treatment *does not* have a significant effect on the outcome, also known as the *status-quo*; 

   Alternative ($H_A$) claim that the treatment *does* have a significant effect on the outcome. It describes what we are interested in.

3. **Determine a test statistics** (quantity that can differentiate between $H_0$ and $H_A$, and whose distribution under $H_0$ you can compute.)

4. **Determine a significance level** ($\alpha$) . Significance level is the probability of rejecting $H_0$ when $H_0$ is true. (how much error that you are comfortable with.)

5. **Reject or not reject the Null Hypothesis.** The decision is based on the test statistics. We reject the null hypothesis if we deem it relatively unlikely for the null hypothesis to be true, given the observations. We fail to reject the null hypothesis if we do not have sufficient evidence from the observation to discredit the null hypothesis.

-----

### Key Points

* Why hypothesis testing?

A high-level summary of **hypothesis testing** is that it involves calculating the *probability*, under a given model, that an observation equal to or more extreme than what is observed in the treatment group is obtained, conditioned on the treatment having no effect

* Why statistical model?

The role of a **statistical model** is in calculating this probability. Without a model and its corresponding assumptions, we cannot determine how likely a particular observation in the treatment group is.

* Why p-value?

**p-value** is the probability under $H_0$ to obtain the observed value or a more extreme value of the test statistic. In other words, the p-value measures the "compatibility" of the observed data with the null hypothesis. It is the smallest significance level for which $H_0$ just gets rejected. It can be used for hypothesis testing: Reject $H_0$ if p-value $\leq \alpha$. It can also be used to quantify significance of alternative.

### Error and Power of a Test

Hypothesis testing is an uncertain process due to inherent variation in the observations.  **Error** is the possibility of having the wrong conclusion.

- **Type I error (false positive)**: We reject $H_0$ (equivalently, find the result *significant*) when $H_0$  is actually true. 
- **Type II error (false negative)** : We do not reject $H_0$ (equivalently, find the result *not significant*) when $H_A$ is actually true. 

![type_error](../assets/images/type_error.png)

The **power** of a test is defined as the probability of rejecting $H_0$ when $H_A$ is true, i.e. the probability of correctly rejecting the null hypothesis.

 **Power = 1 - $P$(Type II error) = $P_{H_A}$(reject $H_0$)**.

![power](../assets/images/power.png)

> #### Exercise 1: 
>
> Which of the following statement is / are true?
>
> A. Type I error + Type II error = 1
>
> B. P(Type I error) + P(Type II error) = 1
>
> C. P(Type I error) = $\alpha$ (significance level) = $P_{H_0}$(Reject $H_0$) and is set by us
>
> D. P(Type I error) = $\alpha$ (significance level) = $P_{H_0}$(Reject $H_0$) and depends on the data
>
> > **Answer**: C
>
> > **Solution**: 
> >
> > A. type I error and type II error refer to events, not probabilities
> >
> > B. P(Type I error) = $P_{H0}$(reject null), P(Type II error) = $P_{HA}$(fail to reject null). They are conditional probabilities that are conditioned on different events, so they do not in general sum up to 1.
> >
> > C. The type I error is entirely controllable because we decide on the boundary for the test statistic as to whether we will reject the null hypothesis.

### Trade-offs

There is a direct tradeoff between reducing the type I and type II errors. With a fixed test statistic, reducing the type I error results into increasing the type II error, and vice versa.

Keeping the significance level constant, a **one-sided hypothesis** test has a higher power than the corresponding **two-sided hypothesis test**. An exception is when the distribution of the observation under the alternative hypothesis is bimodal.





# Additional Readings

Introduction to Modern Statistics

https://openintro-ims.netlify.app/