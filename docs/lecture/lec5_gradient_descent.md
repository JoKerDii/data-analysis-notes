# Gradient Descent 

There are topics and exercises.

**Tips**: Gradient descent is usually discussed in the context of optimization, where $\hat{w}$ notation is commonly used to denote the estimator for the model parameters instead of $\hat{\beta}$ commonly used in statistics. 

## 1. Convexity

#### Three criteria:

* **Lower bound convexity**

  Function $f$ is convex if at each point, the gradient gives a linear lower bound, i.e. for all $u,w$:
  $$
  f(u) \geq f(w) + \langle \nabla f(w), u-w \rangle.
  $$
  For a convex function, if $\nabla f(w) = 0$ , then $w$ is a global minimum: $f(u) \geq f(w) + \langle 0, u-w \rangle$ for all $u$. 

  * **Proof**: In 1D case, $f(u) \geq f(w_0) + f'(w_0)(w - w_0)$

    The **Taylor expansion** for the loss function $f(u)$ is
    $$
    f(w) = f(w_0) + f'(w_0)(w - w_0) + R_1(w;w_0)
    $$
    where $R_1(w;w_0)$ is the **first order remainder function for the Taylor series**. This remainder encapsulates all the higher order derivatives
    $$
    R_1(w;w_0) = \frac{1}{2} f^{\prime \prime }(w_0) (w - w_0)^2 + \frac{1}{6} f"'(w_0) (w - w_0)^3 + \mathcal{O}(|w-w_0|^4).
    $$
    We can examine this remainder by use of the **Lagrange** form of the remainder function. There exists some $w_*$ where $w_*$ lies in the interval between $w_0$ and $w$ such that
    $$
    R_1(w;w_0)= \frac{f^{\prime \prime }(w_{*})}{2} (w - w_0)^2
    $$
    Note that $(w - w_0)^2 \geq 0$ and for a convex function $f^{\prime \prime }(w_{*}) \geq 0$. Thus, $R_1(w;w_0) \geq 0$. From this we can conclude that
    $$
    \begin{aligned}
    f(w) & = f(w_0) + f'(w_0)(w - w_0) + R_1(w;w_0)\\
    & \geq f(w_0) + f'(w_0)(w - w_0) \\
    & = f'(w_0) w + (f(w_0) - f'(w_0) w_0)
    \end{aligned}
    $$

* **Chord Convexity**

  Function $f$ is convex if for all $u, w$ and $0 \leq \lambda \leq 1$: 
  $$
  f(\lambda w + (1-\lambda)u) \leq \lambda f(w) + (1-\lambda) f(u)
  $$

  * **Proof**: 

    We construct a point $(w_\alpha , f_\alpha )$ on the chord and a corresponding point $(w_\alpha , f(w_\alpha ))$ on the curve.
    $$
    \begin{aligned}
    w_{\alpha} & = \alpha w_ a + (1-\alpha ) w_ b\\
    f_{\alpha} & = \alpha f(w_ a) + (1-\alpha ) f(w_ b)
    \end{aligned}
    $$
    where $0< \alpha < 1$. 

    The tangent lower bounds at $w_{\alpha}, w_a, w_b$ are
    $$
    f(w_{\alpha}) \geq f(w_\alpha ) + f'(w_\alpha ) (w - w_\alpha )\\
    f(w_a) \geq f(w_\alpha ) + f'(w_\alpha ) (w_ a - w_\alpha )\\
    f(w_b) \geq f(w_\alpha ) + f'(w_\alpha ) (w_ b - w_\alpha )
    $$
    Now, we can find a lower bound on $f_{\alpha}$
    $$
    \begin{aligned}
    f_{\alpha} & = \alpha f(w_ a) + (1-\alpha ) f(w_ b)\\
    & \geq \left(\alpha f(w_\alpha ) + (1-\alpha ) f(w_\alpha )\right) + f'(w_\alpha ) \left[ \alpha (w_ a - w_\alpha ) + (1-\alpha )(w_ b - w_\alpha ) \right]\\
    &= f(w_\alpha ) + f'(w_\alpha ) \left[ \alpha \left((1-\alpha ) w_ a - (1-\alpha ) w_ b\right) + (1-\alpha )\left(\alpha w_ b - \alpha w_ a\right) \right]\\
    &= f(w_\alpha ) + f'(w_\alpha ) \left[ \alpha (1-\alpha ) (w_ a - w_ b) - (1-\alpha ) \alpha (w_ a - w_ b) \right]\\
    & = f(w_{\alpha})
    \end{aligned}
    $$
    Therefore, $f_\alpha \geq f(w_\alpha )$ for all $\alpha \in (0,1)$ .                                                   

* If $f$ is **twice differentiable**

  $f$ is convex if for all $w$,  $\nabla^2 f(w)$ is positive semidefinite ( In 1D: $f''(w) \geq 0$ ).

  $\nabla^2 f(w)$ is Hessian Matrix.  Matrix $A$ is psd if $v^TAv \geq 0$ for all $v$ or only has non-negative eigenvalues.

> #### Exercise 11
>
> Recall that for the **Ordinary Least Squares (OLS) problem**
> $$
> \hat{\mathrm{{\boldsymbol \omega }}} = \arg \min _{\mathrm{{\boldsymbol w}}} \sum _{i=1}^ N \left(Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\mathrm{{\boldsymbol w}}\right)^2,
> $$
> The loss /objective function that we want to minimize can be written as
> $$
> f(\mathrm{{\boldsymbol w}}) = \sum _{i=1}^ N \left(Y_ i - \mathrm{{\boldsymbol x}}_ i^{\intercal }\mathrm{{\boldsymbol w}}\right)^2.
> $$
> We have found a critical point $w′$ such that $f′′(w′)>0$, so that we know that it is a minimum.  Is $f′′(w′)>0$ a sufficient condition or necessary condition or neither both for $w'$ to be global minimum?
>
> > **Answer**: $f′′(w′)>0$ is neither a sufficient condition for $w′$ to be global minimum, nor is it a necessary condition.
>
> > **Solution**: 
> >
> > * Not a sufficient condition: 
> >
> >   Suppose a critical point B (local minimum) has $f'(B) = 0$, and also has $f^{\prime \prime }(w') > 0$, it is not global minimum.
> >
> > * Not a necessary condition: 
> >
> >   Consider a loss function $f(w) = w^4$. The point $w' = 0$ is a critical point. and also a global minimum. However $f''(w) = 0$, so $f''(w) > 0$ is not a necessary condition for $w'$ to be a global minimum, or even a local minimum.

Note that $y=x$ is a convex function, because the definitions of convexity use "$\geq$" relations. The second derivative of a line may be zero, but this is technically non-negative as so the line is convex.

## 2. Multidimensional convexity and local optimization

Suppose a column vector $\mathrm{{\boldsymbol w}}$ contains multiple weights, the loss function $f(\mathrm{{\boldsymbol w}})$ can be expanded around a point in parameter space $\mathrm{{\boldsymbol w_0}}$, through **multidimensional Taylor expansion**.
$$
f(\mathrm{{\boldsymbol w}})= f(\mathrm{{\boldsymbol w}}_0) + (\nabla f)(\mathrm{{\boldsymbol w}}_0) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0) + \frac{1}{2} (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0)^{\intercal }(\nabla \nabla f) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0) + \mathcal{O}(|\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0|^3)
$$
where $(\nabla f)(\mathrm{{\boldsymbol w}}_0)$ is the gradient of $f(\mathrm{{\boldsymbol w}})$ evaluated at $\mathrm{{\boldsymbol w}}_0$, and $\nabla \nabla f$ is the Hessian matrix which contains all the second derivatives of $f$.

The critical point $\mathrm{{\boldsymbol w}}'$ of $f(\mathrm{{\boldsymbol w}})$ can be found by
$$
(\nabla f)(\mathrm{{\boldsymbol w}}') = 0
$$
At a critical point, the Taylor expansion is 
$$
f(\mathrm{{\boldsymbol w}}) = f(\mathrm{{\boldsymbol w}}_0) + \frac{1}{2} (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}')^{\intercal }(\nabla \nabla f) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}') + \mathcal{O}(|\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0|^3)
$$
In the immediate vicinity of the critical point, the behavior of the function is governed by the Hessian term $(\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}')^{\intercal }(\nabla \nabla f) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}')$, Let $\mathrm{{\boldsymbol v}} = \mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}'$ and ${\boldsymbol H} = \nabla \nabla f$, this term can be written as $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}}$.

Now if $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}} > 0$ for all non-zero vectors $\mathrm{{\boldsymbol v}} \neq 0$, then

* the critical point is a **minimum**.
* the matrix ${\boldsymbol H}$ is positive definite
* as ${\boldsymbol H}$ is real-valued and symmetric, the eigenvalues and determinants of it are all positive. (Recall that the determinant is the product of all the eigenvalues of the matrix)

Now if $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}} \geq 0$ for all non-zero vectors $\mathrm{{\boldsymbol v}} \neq 0$, then

* the matrix ${\boldsymbol H}$ is semi definite. ( If a real-valued symmetric matrix has all positive eigenvalues except for one or more zero eigenvalues, or all of the eigenvalues are zero, then the matrix is positive semi-definite but not positive definite)
* when $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}} = 0$ at the critical point, it can be a minimum, maximum, or a saddle point. 
  * Interpret each eigenvalue-eigenvector pair as follows: 
    * **Eigenvector** is a vector that points towards a direction away from the critical point
    * **Eigenvalue** shows if the **curvature** of $f(\mathrm{{\boldsymbol w}})$ is positive, negative, or zero in that direction. (curvature means the rate of change of the gradient (i.e. the sign and magnitude of the second derivative))
  * Cases:
    * If ${\boldsymbol H}$ has at least one positive eigenvalue, then we know that there is a direction away from the critical point where the loss function curves **upwards**. 
    * If the same ${\boldsymbol H}$ also has at least one negative eigenvalue, then we know that there is a direction away from the critical point where the loss function curves **downwards**. 
    * A mixture of curving upwards and downwards is the definition of a saddle point, so we now know that the critical point associated with this ${\boldsymbol H}$ is a **saddle point**.
  * In general, a real-valued symmetric matrix with both positive and negative eigenvalues is called an **indefinite matrix**, and the product $\mathrm{{\boldsymbol v}}^{\intercal }{\boldsymbol H} \mathrm{{\boldsymbol v}}$ for any specific $v$ may be positive, negative, or zero.

> #### Exercise 12
>
> Can we determine if a real-valued symmetric matrix (of any general size) is indefinite or not using **only** the determinant of the matrix?
>
> > **Answer**: No
>
> > **Solution**: 
> >
> > An indefinite matrix can have positive, negative or a zero determinant. If any eigenvalues are zero, the determinant will be zero. **Otherwise, if there are an even number of negative eigenvalues, the determinant will be positive. Similarly, if there are an odd number of negative eigenvalues, the determinant will be negative**.

Therefore, one condition for convexity in multiple dimensions: **$f$ is convex if and only if the Hessian is positive semi-definite everywhere**.

**Proof**: 

The Taylor expansion for the loss function $f(\mathrm{{\boldsymbol w}}) $ is
$$
f(\mathrm{{\boldsymbol w}}) = f(\mathrm{{\boldsymbol w}}_0) + (\nabla f)(\mathrm{{\boldsymbol w}}_0) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0) + R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0)
$$
The remainder function is $R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0)$
$$
R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0) = \frac{1}{2} (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0)^{\intercal }(\nabla \nabla f)(\mathrm{{\boldsymbol w}}_{*}) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0)
$$
Suppose the loss function is convex and so the Hessian matrix is positive semi-definite, $R_1(\mathrm{{\boldsymbol w}};\mathrm{{\boldsymbol w}}_0) \geq 0$, the lower bound is 
$$
f(\mathrm{{\boldsymbol w}}) \geq f(\mathrm{{\boldsymbol w}}_0) + (\nabla f)(\mathrm{{\boldsymbol w}}_0) (\mathrm{{\boldsymbol w}} - \mathrm{{\boldsymbol w}}_0)
$$
Thus, we have found a **tangent hyperplane** that is a lower bound for the loss function in more than two dimensions. And this tangent hyperplane exists for all choices of $w$.

## 3. Quadratic minimization

A nicer strategy than solving $\nabla f(w) = 0$: **Approximate $f$ locally with quadratic function $g_t$, then minimize $g_t$.**

#### Deriving GD: Quadratic Upper Bounds

* Goal: 

  * (1) $g_t(w) \geq f(w)~~\forall w$ 
  * (2) $g(w^t) = f(w^t)$

* Use **Taylor expansion** to get $g_t(w)$: in 1D
  $$
  \begin{aligned}
  f(u) & = f(w^t) + f'(w^t)(u-w^t) + 1/2 f''(w^t)(u-w^t)^2 + ...\\
  g_t(u) & = f(w^t) + f'(w^t)(u-w^t) + L/2(u-w^t)^2
  \end{aligned}
  $$

* Choose $L$ large enough to satisfy Taylor expansion equation

* Set $g_t'(u) = 0$ gives minimizer of $g$.
  $$
  u_t = w^t - \frac{1}{L}f'(w^t) = w^{t+1}\\
  $$
  or in higher dimension
  $$
  u_t = w^t - \frac{1}{L}\nabla f(w^t) = w^{t+1}
  $$

Note that when $f(u)$ is steep, set $L$ to be large, else when $f(u)$ is flat, set $L$ to be small. This is consistent with that when $f(u)$ is steep and $L$ is large, the update step is small, when $f(u)$ is flat and $L$ is small, the update step is large.

#### Example: Least Squares Regression

With update rule $w^{t+1} \leftarrow w^t - \alpha_t \nabla f(w^t)$, we now use the **gradient of squared loss**.
$$
\nabla_w (\sum^N_{i=1}(y_i - x_i \cdot w)^2) = \sum^N_{i=1} \nabla_w(y_i - x_i \cdot w)^2 = -2\sum^N_{i=1}(y_i - x_i \cdot w)x_i^T
$$
so $w^{t+1} = w^t + 2 \alpha_t\sum^N_{i=1}(y_i - x_i \cdot w)x_i^T$.

Note that 

* Part $(y_i - x_i \cdot w)$ is called "residual" where $x_i \cdot w$ is $\hat{y}$. The residual is positive if $y_i > \hat{y}$. 
* If the residual is positive, adding fraction of $x_i$ to $w^T$ increases dot product $x_i \cdot w$. 
* If the residual is negative, substract fraction of $x_i$ to $w^T$ decreases dot product $x_i \cdot w$. 
* The update rule could minimize residuals.

#### Newton's Method of Minimization

Recall that the lower bound of a convex loss function 
$$
f(w) \geq f(w_0) + f'(w_0) (w - w_0)
$$
is derived from the Taylor expansion
$$
f(w)= f(w_0) + f'(w_0) (w - w_0) + \frac{1}{2} f^{\prime \prime }(w_0) (w - w_0)^2 + \mathcal{O}(|w-w_0|^3)
$$
We approximate $f(w)$ by truncating this expansion at the second order
$$
g_0(w)= f(w_0) + f'(w_0) (w - w_0) + \frac{1}{2} f^{\prime \prime }(w_0) (w - w_0)^2
$$
As $f$ is convex, $f''(w_0) \geq 0$, $g_0(w)$ has a global minimum.

To find the minimum, we take the derivative with respect to $w$, set it to zero at $w = w_1$. We got
$$
\frac{\partial g_0}{\partial w}(w) = f'(w_0) + f^{\prime \prime }(w_0) (w - w_0) = 0\\
w_1 = w_0 - \frac{f'(w_0)}{f^{\prime \prime }(w_0)}
$$
Thus $w_t$ can be iteratively determined at step $t$
$$
w_{t+1} = w_ t - \frac{f'(w_ t)}{f^{\prime \prime }(w_ t)}
$$
In higher dimensions, 
$$
\mathrm{{\boldsymbol w}}_ {t+1}= \mathrm{{\boldsymbol w}}_ t - \left[ (\nabla \nabla f)(\mathrm{{\boldsymbol w}}_ t) \right]^{-1} (\nabla f)(\mathrm{{\boldsymbol w}}_ t)^{\intercal }
$$
Note that we should be careful about $f''(w_t) = 0$. 



> #### Exercise 13
>
> Will the multidimensional Newton's method work for any convex loss function?
>
> > **Answer**: No
>
> > **Solution**: 
> >
> > * Reason 1: a positive semi-definite matrix may not be invertible
> >
> >   The Hessian matrix for a multidimensional convex loss function is **positive semi-definite**. (Because a positive semi-definite matrix may have a zero determinant so it may not be invertible. )
> >
> >   If we can guarantee that the Hessian is positive definite, then we can always find the inverse Hessian, and we can always apply the multidimensional Newton's method. If the Hessian is positive definite, then the function is called **strongly** convex.
> >
> > * Reason 2: a multidimensional convex function might not have a minimum.
> >
> >   The Hessian may also have negative eigenvalues ( if it is negative semi-definite or indefinite ) in which case the quadratic approximation is no longer convex, and does not have a minimum. 

A common **stop criteria**: when the norm of the gradient is below some threshold $ϵ$:
$$
{| | (\nabla f)(\mathrm{{\boldsymbol w}}_ t) | |}^2 < \epsilon
$$
Another common **stop criteria**: when the difference of loss in the last two iteration is below some threshold $\epsilon$
$$
f(\mathrm{{\boldsymbol w}}_{t-1}) - f(\mathrm{{\boldsymbol w}}_{t}) < \epsilon
$$

#### Gradient Descent

The drawback of Newton's Method is to compute Hessian Matrix $\nabla \nabla f$, which is computationally demanding. To improve this, we can take a guess at the value of the Hessian as 
$$
(\nabla \nabla f) \approx \frac{1}{\alpha } {\boldsymbol I}
$$
where $\alpha$ is a positive real number, often called **step size**. Now the iterative update rule becomes 
$$
\mathrm{{\boldsymbol w}}_ {t+1}= \mathrm{{\boldsymbol w}}_ t - \alpha (\nabla f)(\mathrm{{\boldsymbol w}}_ t)^{\intercal }
$$
which requires knowledge of only the gradient. 

Note that there is no guarantee whether gradient descent would find a minimum if there is a minimum.

## 4. Step sizes and quadratic bounds