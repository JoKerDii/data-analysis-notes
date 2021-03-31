# Correlation and Least Squares Regression

There are 6 topics and 4 exercises.

##  1. Correlation

Mean:
$$
\bar{X} = \frac{1}{N} \sum _{i=1}^ N X_ i
$$
Population standard deviation:
$$
s_ x = \sqrt{N^{-1} \sum _{i=1}^ N (x_ i - \bar{x})^2}
$$
Sample standard deviation:
$$
\sigma_x = \sqrt{(N-1)^{-1} \sum _{i=1}^ N (x_ i - \bar{x})^2}
$$
Sample covariance between $X$ and $Y$:  
$$
s^2_{X,Y} = \frac{1}{N-1} \sum _{i=1}^ N (X_ i - \bar{X}) (Y_ i - \bar{Y})
$$
Correlation coefficient:
$$
r = \frac{1}{N} \sum^N_{i=1} (\frac{x_i - \bar{x}}{s_x}) (\frac{y_i - \bar{y}}{s_y}) = \frac{cov(x,y)}{s_xs_y}
$$
Note that correlation coefficient only measures linear relationship, so be careful with **nonlinearities** and **outliers**.

> #### Exercise 7
>
> Suppose $Y=X^2$. What is the correlation between random variables $X$ and $Y$?
>
> A. 0, because correlation only measures variables' relations up to linear relations
>
> B. 1
>
> C. 1/4
>
> D. We do not know, it depends on $X$'s distribution
>
> > **Answer**: D
>
> > **Solution**: It is true "correlation only measures variables' relations up to linear relations", but there is a chance that $X$ and $Y$ have no linear relationship but they actually have nonzero correlation: 
> >
> > Suppose $X = 0$ with probability $1/2$, and $X = 2$ otherwise. Then 
> > $$
> > Cov(X,Y) = E(XY) - E(X)E(Y) = 4-1 \times 2 = 2\\
> > r(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(x)Var(Y)}} = \frac{2}{\sqrt{1 \cdot 4}} = 1
> > $$

> #### Exercise 8
>
> Suppose $X \sim \mathcal{N}(0,1), Y = X^2$. What is the correlation between random variables $X$ and $Y$.
>
> > **Answer**: 1
>
> > **Solution**: 
> > $$
> > \begin{aligned}
> > Cov(X,Y) & = E((X - EX)(Y - EY)) \\
> > & = E[XY] - E[X]E[Y]\\
> > & = E[X^3] - E[X]E[X^2] \\
> > & = 0 - (0 \cdot E[X^2])\\
> > & = 0
> > \end{aligned}
> > $$
> > Thus $r(X,Y) = 0$ as well.

## 2. Regression

####  Regression line for y on x 

1. Related to **Correlation Coefficient**: **Increase of 1 std dev in $x$ associated with increase of $r$ std dev in $y$.**

   * The slope of the linear model is governed by the correlation coefficient: $\hat{\beta_1} = r\frac{s_y}{s_x}$.

2. Interpolating conditional averages of $y$ given $x$.

3. Solution to **least squares** problem.

   * Model: 

   $$
   \hat{y_i} = \hat{\beta_1}x_i + \hat{\beta_0}
   $$

   * Fit to minimize **RMS error (Gauss)**:

   $$
   \sqrt{\frac{1}{N}\sum^N_{i=1} (\hat{y_i} - y_i)^2} = \sqrt{\frac{1}{N}\sum^N_{i=1} (\beta_0 + \beta_1x_i - y_i)^2}
   $$

   * The least square solution: 
     $$
     \begin{aligned}
     \hat{\beta_1} & = \frac{\sum _ i^ N (X_ i - \bar{X}) (Y_ i - \bar{Y})}{\sum _ i^ N (X_ i - \bar{X})^2} \\
     & = \frac{s^2_{X,Y}}{s_ X^2} \\
     & = \frac{s^2_{X,Y}}{s_ X s_ Y} \frac{s_ Y}{s_ X} \\
     & = r_{X,Y} \frac{s_ Y}{s_ X}.
     \end{aligned}
     $$

   * RMS error is $\sqrt{1-r^2}s_y$.

4. Caution: **Ecological Correlations** tend to overstate the strength of an association for individuals. 
   
   * Reason: when different groups are centered or each group is represented as a mean point, the correlation turns to be higher than the correlation of original distributed points.
5. Caution: regression line implies linear relationship, but not implies causality.

> #### Exercise 9
>
> Fit this linear model below through least squares, what is the least square solution?
> $$
> \hat{X}(Y) = \hat{\alpha }_1 Y + \hat{\alpha }_0,
> $$
>
> > **Answer**: $\hat{\alpha }_1 = \hat{\beta }_1 s^2_ X/s^2_ Y$
>
> > **Solution**: 
> >
> > The best fit line of regressing $Y$ on $X$ and regressing $X$ on $Y$ are different. The model of the first case is $Y = \beta _1 X + \beta _0 + \epsilon$, while the model of the second case is $X = \alpha _1 Y + \alpha _0 + \epsilon$. 
> >
> > Regressing $X$ on $Y$ should result in slope $\hat{\alpha }_1 = r_{X,Y} \frac{s_ X}{s_ Y}$. Note that the correlation coefficient is symmetric, $r_{X,Y} = r_{Y,X}$.

#### Goodness of fit

The least squared regression problem has a goodness of fit metric called the **coefficient of determination**, $R^2$. 
$$
R^2 = 1 - \frac{SumSq_{\text {res}}}{SumSq_{\text {tot}}}
$$
where the sum of the squared residuals, and the total sum of the squares (for $Y$) are:
$$
SumSq_{res} = \sum _ i^ N \left( Y_ i - \hat{Y}(X_ i) \right)^2\\
SumSq_{tot} = \sum _ i^ N \left( Y_ i - \bar{Y} \right)^2
$$
**Alternatives** of calculating the goodness of fit of the least square regression line
$$
\begin{aligned}
R^2 & = 1 - \frac{\sum _ i^ N \left(Y_ i - \hat{\beta }_1 X_ i - \hat{\beta _0}\right)^2}{\sum _ i^ N (Y_ i - \bar{Y})^2 }\\
& = \frac{\sum _ i^ N \left[ (Y_ i - \bar{Y})^2 - \left(Y_ i - \frac{r_{X,Y} s_ Y}{s_ X} X_ i - \left[\bar{Y} - \frac{r_{X,Y} s_ Y}{s_ X} \bar{X} \right]\right)^2 \right]}{\sum _ i^ N (Y_ i - \bar{Y})^2 } \\ 
& = \frac{\sum _ i^ N \left[ (Y_ i - \bar{Y})^2 - \left( (Y_ i - \bar{Y}) - \frac{r_{X,Y} s_ Y}{s_ X} ( X_ i - \bar{X} ) \right)^2 \right]}{\sum _ i^ N (Y_ i - \bar{Y})^2 }\\
& = \frac{\sum _ i^ N \left[ 2 \frac{r_{X,Y} s_ Y}{s_ X} (Y_ i - \bar{Y}) ( X_ i - \bar{X} ) - \frac{r_{X,Y}^2 s_ Y^2}{s_ X^2} ( X_ i - \bar{X} )^2 \right]}{\sum _ i^ N (Y_ i - \bar{Y})^2 } \\
& = \frac{2 \frac{r_{X,Y} s_ Y}{s_ X} (N-1) s_{X,Y}^2 - \frac{r_{X,Y}^2 s_ Y^2}{s_ X^2} (N-1) s_ X^2 }{ (N-1) s_ Y^2 } \\
& = 2 r_{X,Y} \frac{ s_{X,Y}^2 }{s_ Y s_ X} - r_{X,Y}^2\\
& = 2 r_{X,Y}^2 - r_{X,Y}^2\\
& = r_{X,Y}^2
\end{aligned}
$$
Thus,
$$
R^2 = r_{X,Y}^2
$$

## 3. Correcting simple non-linear relationships

#### Evaluate the model fit: Residuals.

Assumption: 

* Linear relationship: $Y = \beta_1 X + \beta_0 + \epsilon$
* errors $\epsilon$ are mean zero, independent, and Gaussian

**Residuals**: $e_i = y_i - \hat{y_i}$, where $\hat{y_i} = \hat{\beta_1}x_i + \hat{\beta_0}$. Plot $e_i \sim x_i$, or $e_i \sim \hat{y_i}$.

* Should show no pattern, points regularly scattered around 0.
* If there is any pattern: Variables transformations $log(y), \sqrt{y}, \sqrt{x}, \log(x), x^2$.

> #### Exercise 10
>
> How to fit linear models for the following relationships? ( $\alpha$ and $\beta$ are parameters.)
>
> 1. $Y = \alpha e^{\beta X}$
> 2. $Y = \beta \ln {X} + \alpha$
> 3. $Y^{\gamma } = \alpha X^{\beta }$
>
> > **Answer**: 
> >
> > 1. Take the log of both sides
> > 2. Take the log of $X$
> > 3. Take the log of both sides
>
> > **Solution**: 
> >
> > 1. Taking the log of both sides 
> >    $$
> >    \ln {Y} = \beta X + \ln {\alpha }
> >    $$
> >    Thus, $\ln {Y}$ and $X$ have a linear relationship with coefficient $\beta$ and intercept $\ln{\alpha}$.
> >
> > 2. Taking the log of $X$
> >    $$
> >    Y = \beta \ln {X} + \alpha
> >    $$
> >    Thus, $Y$ and $\ln{X}$ have a linear relationship with coefficient $\beta$ and intercept $\alpha$.
> >
> > 3. Taking the log of both sides
> >    $$
> >    \ln {Y} = \frac{\beta \ln {X} + \ln {\alpha }}{\gamma }
> >    $$
> >    Thus, $\ln{Y}$ and $\ln{X}$ have a linear relationship with coefficient $\beta/\gamma$ and intercept $\ln{\alpha}/\gamma$.

#### Effect of transformations on noise

Assume $\epsilon$ is some multiplicative noise, we have
$$
Y = \alpha e^{\beta X} \epsilon
$$
After log transformation we have
$$
\ln {Y} = \beta X + \ln {\alpha } + \ln {\epsilon }
$$
If $\epsilon$ are **log-normally distributed**, then $\ln {\epsilon}$ should be normally distribution.

However, if we have some addictive noise,
$$
Y = \alpha \left( e^{\beta X} + \epsilon \right)
$$
Then after log transformation we have
$$
\ln {Y} = \ln {\left(e^{\beta X} + \epsilon \right)} + \ln {\alpha }
$$
If $\epsilon$ is small, the relationship will be approximately linear, but if $\epsilon$ is large, there will be no linear relation and we should resort to non-linear model.

## 4. Multiple linear regression

**Model**: $y_i = \beta_0 + x_{i1}\beta_1 + x_{i2}\beta_2 + \epsilon$

* **Vector** form: $y_i  =x_i \beta + \epsilon_i$
* **Matrix-vector** form: $y = X\beta + \epsilon$
  * Dependent variable: $y: N \times 1$
  * Design matrix: $X: N \times p$
  * Parameters: $\beta: p \times 1$
  * Random error/ disturbance $\epsilon_i$ are iid, $E[\epsilon_i] = 0, Var[\epsilon_i] = \sigma^2$. 

$$
\begin{bmatrix}y_1\\y_2\\ y_3\\y_4 \end{bmatrix} = \begin{bmatrix}1&x_{11}&x_{12}\\1&x_{21}&x_{22}\\ 1&x_{31}&x_{32}\\1&x_{41}&x_{42} \end{bmatrix} \begin{bmatrix}\beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix} + \begin{bmatrix} \epsilon_1 \\\epsilon_2\\ \epsilon_3\\\epsilon_4 \end{bmatrix}
$$

#### Ordinary Least Squares Estimator (OLS)

**Model**: $y_i = \beta_0 + x_{i1}\beta_1 + x_{i2}\beta_2 + \epsilon$

**Fitted values**: $\hat{y_i} = \hat{\beta_0} + x_{i1}\hat{\beta_1} + x_{i2}\hat{\beta_2}$ or $\hat{y_i} = x_i \hat{\beta}$.

Least squares:
$$
\hat{\beta} = arg \min_{\beta} \sum^N_{i=1}(y_i - x_i \beta)^2 = arg \min_{\beta} \|y - X\beta\|^2
$$
The Least squares objective
$$
f(\beta) = \sum^N_{i=1}(y_i - x_i \beta)^2 = (y - X\beta)^T(y - X\beta)
$$
Setting gradient to zero 
$$
\nabla_{\beta}f(\beta) = \begin{bmatrix} \frac{\partial f}{\partial \beta_0} \\\frac{\partial f}{\partial \beta_1} \\ \vdots \\ \frac{\partial f}{\partial \beta_{p-1}} \end{bmatrix} = 0
$$
If $\beta$ is $p \times 1$, then $\nabla_{\beta}f(\beta)$ is $p \times 1$.

Recall the vector product rule is 
$$
\nabla (\mathrm{{\boldsymbol u}}^{\intercal }\mathrm{{\boldsymbol v}}) = \mathrm{{\boldsymbol u}}^{\intercal }\nabla \mathrm{{\boldsymbol v}} + \mathrm{{\boldsymbol v}}^{\intercal }\nabla \mathrm{{\boldsymbol u}}
$$
Therefore, 
$$
\nabla_{\beta}f(\beta) = -2\mathrm{{\boldsymbol y}}^{\intercal }{\boldsymbol X} + 2\mathrm{{\boldsymbol \beta }}^{\intercal }{\boldsymbol X}^{\intercal }{\boldsymbol X} = 0
$$
This gives **normal equations**
$$
{\color{blue}X^TX \hat{\beta} = X^Ty}
$$
If $X^TX$ is **invertible**, then $\hat{\beta} = (X^TX)^{-1}X^Ty$.

Plug it in $\hat{y} = X \hat{\beta}$ we have fitted values:
$$
\hat{y} = X \hat{\beta} = X (X^TX)^{-1}X^T y
$$
where $X (X^TX)^{-1}X^T$ is called **hat matrix**.

## 5. Regularization

When $X^TX$ is **invertible / has full rank**

* $N \geq p$, where $N$ is the number of data, $p$ is the number of feature columns.
* All columns linearly independent.

If $p > N$, we use **regularization**

* **$l_2$ penalty**: minimize
  $$
  \sum^N_{i=1} (y_i - \hat{y_i})^2 + \lambda\|\beta\|^2_2
  $$
  where $\|\beta\|^2_2 = \sum^{p-1}_{j=0}\beta_j^2$.

* **$l_1$ penalty (Lasso)**: minimize
  $$
  \sum_{i=1}^N(y_i - \hat{y_i})^2 + \lambda \|\beta\|_1
  $$
  where $\|\beta\|_1 = \sum^{p-1}_{j=0}|\beta_j|$

  prefer sparse $\beta$ (few nonzero coordinates)

## 6. Model Selection (t-test)

Set $\beta_j = 0$ to exclude variable $j$ from the prediction

* Idea: $\hat{\beta_j}$ is a random variable. Do a **t-test**.

* Recall: model and estimator
  $$
  y_i = x_i \beta + \epsilon_i\\ E[(\epsilon_i)^2] = \sigma^2\\ \hat{\beta} = (X^TX)^{-1}X^Ty
  $$

* **OLS is (conditionally) unbiased**: 
  $$
  E[\hat{\beta}|X] = \beta
  $$
  This means if the data is enough, the expectation of estimated $\hat{\beta}$ is close to parameter $\beta$. (Now if we make a few more assumptions about $\epsilon$, we can make it a test.)

* **Gaussianity**: If $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, model correct and $X$ fixed, then $y_i$ is a linear function of Gaussian random variables $\epsilon$ or the shifted ( by $x_i \beta$) version of $\epsilon$. And since $\hat{\beta}= (X^TX)^{-1}X^Ty$, each of $\hat{\beta_i}$ is a linear combination of each $y_i$ and each $y_i$ is shifted Gaussian, $\beta$ are also Gaussian random variables. Therefore, $\hat{\beta} \sim \mathcal{N}(\beta, \sigma^2(X^TX)^{-1})$.

* **t-test** to check $\beta_j = 0$ vs. $\beta_j \neq 0$: estimate $\sigma^2$ as 
  $$
  \hat{\sigma^2} = \frac{1}{N-p-1} \sum^N_{i-1}(y_i - \hat{y_i})^2
  $$
  Then
  $$
  (N-p-1)\hat{\sigma^2} \sim \sigma^2\chi^2_{N-p}
  $$

#### Derive the statistical distribution for this t-test

As the least squares estimator is conditionally unbiased $\mathbb {E}[\hat{\mathrm{{\boldsymbol \beta }}}|{\boldsymbol X}] = \mathrm{{\boldsymbol \beta }}$, we first compute the **deviance** from mean.
$$
\begin{aligned}
\hat{\mathrm{{\boldsymbol \beta }}} - \mathbb {E}[\hat{\mathrm{{\boldsymbol \beta }}}|{\boldsymbol X}] & = \hat{\mathrm{{\boldsymbol \beta }}} - \mathrm{{\boldsymbol \beta }}\\
&= ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol y}} - \mathrm{{\boldsymbol \beta }}\\
&= ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\left( {\boldsymbol X} \mathrm{{\boldsymbol \beta }} + \mathrm{{\boldsymbol \epsilon }} \right) - \mathrm{{\boldsymbol \beta }}\\
&= ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} ({\boldsymbol X}^{\intercal }{\boldsymbol X}) \mathrm{{\boldsymbol \beta }} - \mathrm{{\boldsymbol \beta }} + ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol \epsilon }}\\
&= ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol \epsilon }}
\end{aligned}
$$
We then compute the **covariance matrix** of $\hat{\beta}$. Note that the covariance matrix for the noise is $\mathbb {E}[\mathrm{{\boldsymbol \epsilon }}\mathrm{{\boldsymbol \epsilon }}^{\intercal }] = \sigma ^2 I$
$$
\begin{aligned}
\mathbb {E}[(({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol \epsilon }})(({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol \epsilon }})^{\intercal }|{\boldsymbol X}] & = ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathbb {E}[\mathrm{{\boldsymbol \epsilon }}\mathrm{{\boldsymbol \epsilon }}^{\intercal }|{\boldsymbol X}] {\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} \\
& = \sigma ^2 ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }{\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}\\
& = \sigma ^2 ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}
\end{aligned}
$$
Now we have to find $\sigma$, we cannot compute the variance $\sigma$ directly from $\beta$ since it requires the mean and we don't have access to $\beta$ directly. Instead, we find $\sigma$ through the variance of $\mathrm{{\boldsymbol y}}$.
$$
\begin{aligned}
S & = \sum _ i^ N (Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\hat{\mathrm{{\boldsymbol \beta }}})^2\\
& = (\mathrm{{\boldsymbol y}} - {\boldsymbol X} \hat{\mathrm{{\boldsymbol \beta }}})^{\intercal }(\mathrm{{\boldsymbol y}} - {\boldsymbol X} \hat{\mathrm{{\boldsymbol \beta }}})\\
&= (\mathrm{{\boldsymbol y}} - {\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol y}})^{\intercal }(\mathrm{{\boldsymbol y}} - {\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }\mathrm{{\boldsymbol y}})\\
&= ({\boldsymbol X}\mathrm{{\boldsymbol \beta }} + \mathrm{{\boldsymbol \epsilon }})^{\intercal }(I - {\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal })^2 ({\boldsymbol X}\mathrm{{\boldsymbol \beta }} + \mathrm{{\boldsymbol \epsilon }})\\
& = \mathrm{{\boldsymbol \epsilon }}^{\intercal }(I - {\boldsymbol H})^2 \mathrm{{\boldsymbol \epsilon }}
\end{aligned}
$$
where ${\boldsymbol H} = {\boldsymbol X} ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1} {\boldsymbol X}^{\intercal }$, $\boldsymbol H$ is idempotent, so as $(I - \boldsymbol H)$. Note that **the rank of an idempotent matrix equals to the trace of the matrix**, and the rank of $\boldsymbol H$ equals to the number of columns in $X$, which is $p$.  So that $\text{rank} (I - \boldsymbol H) = N - \text{rank}(\boldsymbol H) = N - p$. 

The reason why we use the variance of $\mathrm{{\boldsymbol y}}$ is that with $\mathbb {E}[\mathrm{{\boldsymbol \epsilon }}] = 0,~
\mathbb {E}[\mathrm{{\boldsymbol y}}|{\boldsymbol X}] = \mathbb {E}[{\boldsymbol X} \mathrm{{\boldsymbol \beta }} | {\boldsymbol X}] = {\boldsymbol X} \mathrm{{\boldsymbol \beta }}$
$$
\mathbb {E}[(\mathrm{{\boldsymbol y}} - \mathbb {E}[\mathrm{{\boldsymbol y}}|{\boldsymbol X}])(\mathrm{{\boldsymbol y}} - \mathbb {E}[\mathrm{{\boldsymbol y}}|{\boldsymbol X}])^{\intercal }|{\boldsymbol X}] = \mathbb {E}[\mathrm{{\boldsymbol \epsilon }}\mathrm{{\boldsymbol \epsilon }} | ^{\intercal }{\boldsymbol X}] = \sigma^2I
$$
Since $\mathbb {E}[(I - {\boldsymbol H}) \mathrm{{\boldsymbol \epsilon }} | {\boldsymbol X}] = 0$, $S$ is a sum of squares each with mean 0.

As the variance of $\epsilon$ is $\sigma^2$ and the mean of $\epsilon$ is 0, $\epsilon/\sigma$ is a **standard normally distributed random variable**. From **Cochran's theorem**, we have
$$
\frac{S}{\sigma^2}= \frac{\mathrm{{\boldsymbol \epsilon }}^{\intercal }}{\sigma } (I - {\boldsymbol H}) \frac{\mathrm{{\boldsymbol \epsilon }}}{\sigma }
$$
is $\chi^2$ distributed with number of degrees of freedom equals to $\text{rank}(I - H) = N-p$. Therefore
$$
\mathbb {E}[\frac{S}{\sigma ^2}|{\boldsymbol X}] = N - p
$$
We can use this as an estimator of $\sigma^2$:
$$
\hat{\sigma^2} = \frac{S}{N-p}
$$
The normally distributed variable $Z$ that can be used in a z-test to test the null hypothesis $\beta_j = 0$:
$$
Z = \frac{\hat{\beta }_ j - 0}{\sigma \Sigma _ j} \sim \mathcal{N}(0,1)
$$
where $\Sigma _ j^2 = \left({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}\right)_{jj}$, the $j$th element on the diagonal, $\sigma \Sigma_j = \sqrt{\sigma ^2 ({\boldsymbol X}^{\intercal }{\boldsymbol X})^{-1}}$, the squared root of covariance matrix.

Since an estimator of $\sigma^2$ is 
$$
\hat{\sigma^2} = \frac{S}{N-p} = \frac{{| | \mathrm{{\boldsymbol y}} - {\boldsymbol X} \hat{\mathrm{{\boldsymbol \beta }}} | |}^2}{N-p}
$$
Thus $w$ has a $\chi^2$ distribution with $N-p$ degrees of freedom.
$$
(N-p)\frac{\hat{\sigma }^2}{\sigma ^2} \sim \chi^2_{N-p}
$$
Now if $Z \sim \mathcal{N}(0,1)$ and $w \sim \chi_n^2$ then
$$
\frac{Z}{\sqrt{\frac{\omega }{n}}} \sim t_n
$$
is t distributed with $n$ degrees of freedom. Therefore,
$$
T_j = \frac{ \frac{\hat{\beta }_ j - 0}{\sigma \Sigma _ j} }{ \sqrt{\frac{(N-p)\frac{\hat{\sigma }^2}{\sigma ^2}}{N-p}} } = \frac{\hat{\beta_j}}{\hat{\sigma \Sigma_j}}
$$
is t distributed with $N-p$ degrees of freedom and can be used as a t-test to test the hypothesis that $\beta_j = 0$.