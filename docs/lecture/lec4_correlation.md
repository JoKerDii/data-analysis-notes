# Correlation and Least Squares Regression

There are topics and exercises.

##  Correlation

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

## Regression

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

## Correcting simple non-linear relationships



