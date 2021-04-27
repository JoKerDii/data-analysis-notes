# Classification using Bayes Rule

There are 5 topics and 4 exercises.

## 1. Bayes Rule

Recall the given two events $A$ and $B$, Bayes' Rule is 
$$
\mathbf{P}(A|B)= \frac{\mathbf{P}(B|A)\mathbf{P}(A)}{\mathbf{P}(B)}
$$
where $P(A), P(B)$ are marginal probabilities, $P(A|B),P(B|A)$ are conditional probabilities.

In modeling data, for random variables $X, \Theta$, Bayes' rule states that 
$$
\mathbf{P}(\Theta =\theta |X=x)= \frac{\mathbf{P}(X=x|\Theta =\theta )\mathbf{P}(\Theta )}{\mathbf{P}(X)}
$$
Here we assume $X$ are **continuous variables** that take values in $\R^p$ and $\Theta$ are **finite discrete variables** that represent the classes.

In classification, let $C$ be a random variable representing class labels. We can estimate the class label of data point $x$ by finding the class that maximizes the posterior distribution $P(C|X=x)$, which is given by Bayes' Rule
$$
\mathbf{P}(C|X)\, =\,  \frac{\mathbf{P}(C) \mathbf{P}(X|C)}{\mathbf{P}(X)} \, \propto \,  \mathbf{P}(C) \mathbf{P}(X|C)
$$

* $P(C)$ is the **prior** or **prevalence** or fraction of samples in that class, which can be either based on prior studies and knowledge, or estimated from the prevalence in the training data.

* $P(X|C)$ can be assume to be any distribution. For example, it can follow Gaussian distribution 
  $$
  X|C \sim \mathcal{N}(\mu_c, \Sigma_c)
  $$
  we can use **MLE** to estimate $\mu_c$ and $\Sigma_c$ so as to maximize the probabilities of points belonging to each class.

## 2. Quadratic Discriminant Analysis (QDA)

Quadratic Discriminant Analysis (QDA):

* We model $X \in \R$ as Gaussian variables 
  $$
  X|C = c \sim \mathcal{N}(\mu_c, \Sigma_c)
  $$

* Estimate $P(C = c), \mu_c, $ and $\Sigma_c$ for each $c$ by choosing class $c$ to maximize 
  $$
  P(C = c|X=x) \propto P(C=c) P(X=x|C=c)
  $$

* Use the fact that maximizing $P(C=c|X=x)$ is equivalent to maximize $\log(P(C=c|X=x))$.
  $$
  \begin{aligned}
  \log(P(C=c|X=x)) &\propto \log(P(C=c) P(X=x|C=c))\\
  &\propto \log P(C=c) + \log P(X=x|C=c)\\
  &\propto \log P(C=c) + \log [\frac{1}{\det \Sigma_c^{1/2}} \exp(-\frac{1}{2}(x-\mu_c)^T \Sigma_c^{-1}(x-\mu_c))]\\
  & \propto \log P(C=c) - \frac{1}{2}\log \det\Sigma_c -\frac{1}{2}(x-\mu_c)^T \Sigma_c^{-1}(x-\mu_c)
  \end{aligned}
  $$

* The decision boundaries are **quadratic**.

> #### Exercise 20
>
> Suppose we have two classes in **1-D**, the $\mu$ and $\sigma$ are possibly different for each class
> $$
> X|C = 0 \sim \mathcal{N}(\mu_0, \sigma_0^2)\\
> X|C = 1 \sim \mathcal{N}(\mu_1, \sigma_1^2)
> $$
> the prior distribution of $C$ is
> $$
> \begin{aligned}
> \mathbf{P} (C =1) &= p\\
> \mathbf{P} (C =0) &= 1-p
> \end{aligned}
> $$
> Find the decision boundary
>
> > **Answer**: $x^2{{\left(-\frac{1}{2\sigma _0^2}+\frac{1}{2\sigma _1^2}\right)}} +2x\left(\frac{\mu _0}{2\sigma _0^2}-\frac{\mu _1}{2\sigma _1^2}\right)+\left(-\frac{\mu _0^2}{2\sigma _0^2}+\frac{\mu _1^2}{2\sigma _1^2}\right)+\frac12\ln \left(\frac{\sigma _1^2}{\sigma _0^2}\right)+\ln \left(\frac{p(0)}{p(1)}\right) = 0$.
>
> > **Solution**: 
> >
> > To find the decision boundary we solve 
> > $$
> > \mathbf{P}(C=0|X=x)=\mathbf{P}(C=1|X=x)\\
> > p(0)\frac{1}{\sigma _0}\exp \left(-\frac{(x-\mu _0)^2}{2\sigma _0^2}\right) = \displaystyle p(1)\frac{1}{\sigma _1}\exp \left(-\frac{(x-\mu _1)^2}{2\sigma _1^2}\right)
> > $$
> > equivalently we can solve it in an easier way
> > $$
> > \ln \left[\mathbf{P}(C=0)\mathbf{P}(X=x|C=0)\right]=\ln \left[\mathbf{P}(C=1)\mathbf{P}(X=x|C=1)\right] \\
> > \ln (p(0))-\frac{(x-\mu _0)^2}{2\sigma _0^2} = \ln (p(1))-\frac{(x-\mu _1)^2}{2\sigma _1^2}
> > $$
> > We get a quadratic equation in $x$, which is the decision boundary of this classifier.
> > $$
> > x^2{{\left(-\frac{1}{2\sigma _0^2}+\frac{1}{2\sigma _1^2}\right)}} +2x\left(\frac{\mu _0}{2\sigma _0^2}-\frac{\mu _1}{2\sigma _1^2}\right)+\left(-\frac{\mu _0^2}{2\sigma _0^2}+\frac{\mu _1^2}{2\sigma _1^2}\right)+\frac12\ln \left(\frac{\sigma _1^2}{\sigma _0^2}\right)+\ln \left(\frac{p(0)}{p(1)}\right) = 0
> > $$

## 3. Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA):

* Since the covariance matrices $\Sigma_c$ need a lot of sample, we can simply assume the same $\Sigma$ for all classes, i.e.
  $$
  X|C = c \sim \mathcal{N}(\mu_c, \Sigma)
  $$

* Estimate $P(C=c)$, $\mu_c$, and $\Sigma$ for each $c$ by choosing class $c$ to maximize $\log(P(C=c | X=x))$.
  $$
  \begin{aligned}
  \log(P(C=c|X=x)) &\propto \log(P(C=c) P(X=x|C=c))\\
  &\propto \log P(C=c) + \log P(X=x|C=c)\\
  &\propto \log P(C=c) + \log [\frac{1}{\det \Sigma^{1/2}} \exp(-\frac{1}{2}(x-\mu_c)^T \Sigma^{-1}(x-\mu_c))]\\
  & \propto \log P(C=c) - \frac{1}{2}\mu_c^T \Sigma^{-1}\mu_c +x^T \Sigma^{-1}\mu_c
  \end{aligned}
  $$
  Note that the term $x^T \Sigma x$ has been ignored because $\Sigma$ does not depend on the class any more and this term becomes a constant. Therefore, the decision boundary becomes **linear**.

> #### Exercise 21
>
> Continue from Exercise 20, now in special case we have same variance for two classes.
> $$
> \sigma _0^2\, =\, \sigma _1^2=\sigma ^2
> $$
> Find the decision boundary
>
> > **Answer**: $x = \left(\frac{-\mu _0^2+\mu _1^2}{2\sigma ^2}+\ln \left(\frac{1-p}{p}\right)\right)/2\left(\frac{\mu _1-\mu _0}{2\sigma ^2}\right)$.
>
> > **Solution**: 
> > $$
> > x^2\left(-\frac{1}{2\sigma _0^2}+\frac{1}{2\sigma _1^2}\right)+2x\left(\frac{\mu _0}{2\sigma _0^2}-\frac{\mu _1}{2\sigma _1^2}\right)+\left(-\frac{\mu _0^2}{2\sigma _0^2}+\frac{\mu _1^2}{2\sigma _1^2}\right)+\ln \left(\frac{p(0)}{p(1)}\right) = 0\\
> > 2x\left(\frac{\mu _0-\mu _1}{2\sigma ^2}\right)+\left(\frac{-\mu _0^2+\mu _1^2}{2\sigma ^2}+\ln \left(\frac{p(0)}{p(1)}\right)\right) = 0\\
> > x = \left(\frac{-\mu _0^2+\mu _1^2}{2\sigma ^2}+\ln \left(\frac{1-p}{p}\right)\right)/2\left(\frac{\mu _1-\mu _0}{2\sigma ^2}\right)\\
> > $$
> > (when $p = 0.4$, $x = \frac{\mu_0 + \mu_1}{2}$).

> #### Exercise 22
>
> Compute the decision boundary of QDA and LDA in high dimension for two classes, as well as the normal vector $\mathbf{n}$ of the decision boundary supposing $\Sigma$ has eigenvalues $\sigma^2$ and $1$.
>
> > **Answer**: See below
>
> > **Solution**: 
> >
> > Recall the PDF $f_{\mathbf{X}}(\mathbf{x})$ of a multivariate Gaussian is $\mathcal{N}(\mathbf{\mu},\Sigma)$,
> > $$
> > f_{\mathrm{{\boldsymbol \mu }},\Sigma }(\mathbf{x})=\frac{1}{\sqrt{(2\pi )^ p\mathrm{det}{\Sigma }}}\exp \left(-\frac12(\mathbf{x}-\mathrm{{\boldsymbol \mu }})^ T \Sigma ^{-1}(\mathbf{x}-\mathrm{{\boldsymbol \mu }}) \right).
> > $$
> > To solve for decision boundary we need to solve,
> > $$
> > \mathbf{P}(C=0|\mathbf{X} = \mathbf{x}) - \mathbf{P}(C=1|\mathbf{X} = \mathbf{x}) =0\\
> > \ln \left((1-p)\, f_{\mathrm{{\boldsymbol \mu }}_0,\mathrm{{\boldsymbol \Sigma }}_0}(\mathbf{x})\right)-\ln \left(p\, f_{\mathrm{{\boldsymbol \mu }}_1,\mathrm{{\boldsymbol \Sigma }}_1}(\mathbf{x})\right) = 0\\
> > \ln (1-p)+\  \ln \left(f_{\mathrm{{\boldsymbol \mu }}_0,\mathrm{{\boldsymbol \Sigma }}_0}(\mathbf{x})\right)-\ln (p)- \ln \left(f_{\mathrm{{\boldsymbol \mu }}_1,\mathrm{{\boldsymbol \Sigma }}_1}(\mathbf{x})\right) = 0\\
> >  \frac12\left((\mathbf{x}-\mathrm{{\boldsymbol \mu }}_0)^ T \mathrm{{\boldsymbol \Sigma }}_0^{-1}(\mathbf{x}-\mathrm{{\boldsymbol \mu }}_0) - (\mathbf{x}-\mathrm{{\boldsymbol \mu }}_1)^ T \mathrm{{\boldsymbol \Sigma }}_1^{-1}(\mathbf{x}-\mathrm{{\boldsymbol \mu }}_1)\right)+\frac12\ln \left(\frac{\mathrm{det}{\mathrm{{\boldsymbol \Sigma }}_0}}{\mathrm{det}{\mathrm{{\boldsymbol \Sigma }}_1}}\right)+\ln \left(\frac{p}{1-p}\right) = 0\\
> >  \frac12\left(\mathbf{x}^ T(\mathrm{{\boldsymbol \Sigma }}_0^{-1}-\mathrm{{\boldsymbol \Sigma }}_1^{-1})\mathbf{x}-2\left(\mathrm{{\boldsymbol \mu }}_0^ T \mathrm{{\boldsymbol \Sigma }}_0^{-1}- \mathrm{{\boldsymbol \mu }}_1^ T \mathrm{{\boldsymbol \Sigma }}_1^{-1}\right) \mathbf{x}+ (\mathrm{{\boldsymbol \mu }}_0^ T \Sigma _0^{-1}\mathrm{{\boldsymbol \mu }}_0-\mathrm{{\boldsymbol \mu }}_1^ T \Sigma _1^{-1}\mathrm{{\boldsymbol \mu }}_1 ) \right) +\frac12\ln \left(\frac{\mathrm{det}{\mathrm{{\boldsymbol \Sigma }}_0}}{\mathrm{det}{\mathrm{{\boldsymbol \Sigma }}_1}}\right)+\ln \left(\frac{p}{1-p}\right) = 0
> > $$
> > This is the quadratic equation in the vector $\mathbf{x}$, which is the decision boundary.
> >
> > For LDA, when we make the assumption that $ \mathrm{{\boldsymbol \Sigma }}_0=\mathrm{{\boldsymbol \Sigma }}_1=\mathrm{{\boldsymbol \Sigma }}$, then the quadratic term above vanishes and the equation becomes linear / a hyperplane in $\R^p$.
> > $$
> > \left(\mathrm{{\boldsymbol \mu }}_0 - \mathrm{{\boldsymbol \mu }}_1\right)^ T \mathrm{{\boldsymbol \Sigma }}^{-1} \mathbf{x} = \frac12\left(\mathrm{{\boldsymbol \mu }}_0^ T \Sigma ^{-1}\mathrm{{\boldsymbol \mu }}_0-\mathrm{{\boldsymbol \mu }}_1^ T \Sigma ^{-1}\mathrm{{\boldsymbol \mu }}_1 \right) +\ln \left(\frac{p}{1-p}\right)
> > $$
> > **Note that the change of prior distribution $p$ does not change the normal vector, which is the transpose of $\left(\mathrm{{\boldsymbol \mu }}_0 - \mathrm{{\boldsymbol \mu }}_1\right)^ T \mathrm{{\boldsymbol \Sigma }}^{-1}$. The normal vector is perpendicular to the hyperplane. From the equation above we know that changes in prior distribution shifts the hyperplane but does not change its direction.**
> >
> > We have eigenvalues and eigenvectors of the covariance matrix
> > $$
> >  \mathrm{{\boldsymbol \Sigma }}\, =\, \begin{pmatrix} \uparrow & \uparrow \\ \mathbf v_{\sigma ^2}& \mathbf v_{1}\\ \downarrow & \downarrow \end{pmatrix}\begin{pmatrix} \sigma ^2& 0\\ 0& 1\end{pmatrix}\begin{pmatrix} \leftarrow & \mathbf v_{\sigma ^2}^ T& \rightarrow \\ \leftarrow & \mathbf v_{1}& \rightarrow \end{pmatrix} \qquad \text {where } \\
> >  \mathbf v_{\sigma ^2} =  \frac{1}{\sqrt{2}}\begin{pmatrix} 1\\ 1\end{pmatrix}， \mathbf v_{1} =  \frac{1}{\sqrt{2}}\begin{pmatrix} 1\\ -1\end{pmatrix}
> > $$
> > We can visualize the multivariate Gaussian distribution for two classes below
> >
> > ![MultiGaussianTwoClasses](../assets/images/MultiGaussianTwoClasses.png)
> >
> > From the figure we know that when $\sigma^2 = 1$, the normal vector of the hyperplane is $[1,0]$.
> >
> > In general case, the normal vector is written as
> > $$
> > \begin{aligned}
> > \mathbf{n} &= \mathrm{{\boldsymbol \Sigma }}^{-1} \left(\mathrm{{\boldsymbol \mu }}_1 - \mathrm{{\boldsymbol \mu }}_0\right)\\
> > &=\begin{pmatrix} \uparrow & \uparrow \\ \mathbf v_{\sigma ^2}& \mathbf v_{1}\\ \downarrow & \downarrow \end{pmatrix}\begin{pmatrix} \frac{1}{\sigma ^2}& 0\\ 0& 1\end{pmatrix}\begin{pmatrix} \leftarrow & \mathbf v_{\sigma ^2}^ T& \rightarrow \\ \leftarrow & \mathbf v_{1}& \rightarrow \end{pmatrix} \left(\mathrm{{\boldsymbol \mu }}_1 - \mathrm{{\boldsymbol \mu }}_0\right)\\
> > &= \frac12 \begin{pmatrix} 1& 1\\ 1& -1\end{pmatrix}\left[ \begin{pmatrix} \frac{1}{\sigma ^2}& 0\\ 0& 1\end{pmatrix}\begin{pmatrix} 1& 1\\ 1& -1\end{pmatrix}\begin{pmatrix}  1\\ 0\end{pmatrix} \right]\\
> > &=\frac12 \begin{pmatrix} 1& 1\\ 1& -1\end{pmatrix}\begin{pmatrix} \frac{1}{\sigma ^2}\\ 1 \end{pmatrix}\,\\
> > &=\, \frac12\left(\frac{1}{\sigma ^2} \begin{pmatrix} 1\\ 1\end{pmatrix}+ 1 \begin{pmatrix} 1\\ -1\end{pmatrix} \right)
> > \end{aligned}
> > $$
> > Or $[\frac{1}{\sigma^2} + 1, \frac{1}{\sigma^2} - 1]$.



## 4. Transformation to Spherical Gaussians

To make the classification easier, we assume $\Sigma = \sigma^2I$ which means every class is a **circle**, and the classification procedure is simply assigning points to the class with the mean closest to the points.

* We transform the covariance matrix $\Sigma$ to be an **identity matrix** by using the trick of decomposition
  $$
  \mathbf{X}^T\mathbf{X} = \Sigma = \mathbf{V}\Lambda \mathbf{V}^T\\
  \Lambda^{-1/2}\mathbf{V}^T(\mathbf{X}^T\mathbf{X})\mathbf{V}\Lambda^{-1/2} = \Lambda^{-1/2}\mathbf{V}^T(\mathbf{V} \Lambda \mathbf{V}^T) \mathbf{V}\Lambda^{-1/2} = I_d
  $$

* Now we know what operation we have to do with matrix $\mathbf{X}$ to have $\Sigma = \sigma^2I$. It is just
  $$
  \mathbf{X}' = \mathbf{X}\mathbf{V}\Lambda^{-1/2}
  $$

## 5. Dimensional Reduction using LDA

#### Reduced-rank LDA (a.k.a. Fisher's LDA)

* Idea: $\mu_1, ..., \mu_K \in \R^p$ lie in a linear subspace of dim $K-1$ (usually $p>>k$)

* If $K=3$, then data can be projected into 2d

* If $K > 3$, combine LDA with PCA, i.e. perform PCA on class means 

  e.g. 1st LD is 1st PC of class means; 2nd LD is 2nd PC of class means

* The maximum number of LDs is $\min(p, k-1)$.

* Steps:

  1. Transform the data into a space in which the Gaussian distributions are spherical

     Previously, we found the $p \times p$ coordinate transformation matrix $\mathbf{P}$ such as
     $$
     \mathbf{P}^ T\mathrm{{\boldsymbol \Sigma }}\mathbf{P}=\mathbf{I}_ p
     $$
     which equals to the covariance matrix of the transformed data $\mathbf{P}^ T \mathbf{x}^{(i)}$, since
     $$
     \begin{aligned}
     \mathbf{P}^ T\mathrm{{\boldsymbol \Sigma }}\mathbf{P} &= \mathbf{P}^ T\left(\frac{1}{n-1} \mathbb {X}^ T \mathbb {X}\right) \mathbf{P}\\
      &= \frac{1}{n-1}\mathbf{P}^ T \begin{pmatrix}  \uparrow & \uparrow & & \uparrow \\ \mathbf{x}^{(1)}& \mathbf{x}^{(2)})& \cdots & \mathbf{x}^{(n)}\\ \downarrow & \downarrow & & \downarrow \\ \end{pmatrix}\begin{pmatrix}  \leftarrow & \left(\mathbf{x}^{(1)}\right)^ T& \rightarrow \\ \leftarrow & \left(\mathbf{x}^{(2)}\right)^ T& \rightarrow \\ & \vdots & \\ \leftarrow & \left(\mathbf{x}^{(n)}\right)^ T& \rightarrow \\ \end{pmatrix}\mathbf{P}\\
      &=\frac{1}{n-1}\begin{pmatrix}  \uparrow & \uparrow & & \uparrow \\ \mathbf{P}^ T\mathbf{x}^{(1)}& \mathbf{P}^ T\mathbf{x}^{(2)})& \cdots & \mathbf{P}^ T\mathbf{x}^{(n)}\\ \downarrow & \downarrow & & \downarrow \\ \end{pmatrix}\begin{pmatrix}  \leftarrow & \left(\mathbf{P}^ T\mathbf{x}^{(1)}\right)^ T& \rightarrow \\ \leftarrow & \left(\mathbf{P}^ T\mathbf{x}^{(2)}\right)^ T& \rightarrow \\ & \vdots & \\ \leftarrow & \left(\mathbf{P}^ T\mathbf{x}^{(n)}\right)^ T& \rightarrow \\ \end{pmatrix}
      \end{aligned}
     $$

  2. Perform PCA on the sample covariance matrix of the $k$ transformed class means.
     $$
     \begin{aligned}
     \tilde{\mathbf{S}} &= \mathbf{P}^ T \left[\frac{1}{k-1}\begin{pmatrix}  \uparrow & \uparrow & & \uparrow \\ \mathrm{{\boldsymbol \mu }}_1& \mathrm{{\boldsymbol \mu }}_2& \cdots & \mathrm{{\boldsymbol \mu }}_ k\\ \downarrow & \downarrow & & \downarrow \\ \end{pmatrix}\begin{pmatrix}  \leftarrow & (\mathrm{{\boldsymbol \mu }}_1)^ T& \rightarrow \\ \leftarrow & (\mathrm{{\boldsymbol \mu }}_2)^ T& \rightarrow \\ & \vdots & \\ \leftarrow & (\mathrm{{\boldsymbol \mu }}_ k)^ T& \rightarrow \\ \end{pmatrix}\right]\mathbf{P}\\
      &=  \frac{1}{k-1}\begin{pmatrix}  \uparrow & \uparrow & & \uparrow \\ \mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_1& \mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_2& \cdots & \mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_ k\\ \downarrow & \downarrow & & \downarrow \\ \end{pmatrix}\begin{pmatrix}  \leftarrow & (\mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_1)^ T& \rightarrow \\ \leftarrow & (\mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_2)^ T& \rightarrow \\ & \vdots & \\ \leftarrow & (\mathbf{P}^ T\mathrm{{\boldsymbol \mu }}_ k)^ T& \rightarrow \\ \end{pmatrix}
      \end{aligned}
     $$

  3. Project the transformed data into the subspace spanned by the desired number of PCs.

     e.g. If $\mathbf v_{PC1}$ is the first PC of the class means sample covariance matrix, the final projection of the data onto this component will be 
     $$
     \mathbf{x}^{(i)}\mapsto \mathbf v_{PC1}^ T (\mathbf{P}^ T \mathbf{x}^{(i)})
     $$

> #### Exercise 23
>
> What is an upper bound on the rank of the matrix $\tilde{\mathbf{S}}$ in step 2? In other word, what is an upper bound on the number of linearly independent eigenvectors of $\tilde{\mathbf{S}}$ corresponding to non-zero eigenvalue?
>
> > **Answer**: $p$ (dimension of the original space that the data is given in) and $k$ (the number of classes)
>
> > **Solution**: 
> > $$
> > \text {Rank}\begin{pmatrix}  \uparrow & \uparrow & & \uparrow \\ \mathrm{{\boldsymbol \mu }}_1& \mathrm{{\boldsymbol \mu }}_2& \cdots & \mathrm{{\boldsymbol \mu }}_ k\\ \downarrow & \downarrow & & \downarrow \\ \end{pmatrix}\leq \min (k,p)\qquad (\mathrm{{\boldsymbol \mu }}_ i\in \mathbb {R}^ p).
> > $$
> > Note that the rank of a matrix $\mathbf{A}$ is the same as the rank of the product $\mathbf{A}^T\mathbf{A}$.

#### Comparison of PCA and LDA in dimensionality reduction

PCA

* Linear projection of data without labels. It is an unsupervised learning technique.

* Find the component axes that maximize the variance of data

LDA

* Linear projection of data with class labels. It is a supervised learning technique.
* The goal of LDA is to project a feature space (a dataset n-dimensional samples) onto a smaller subspace $k$ ($k \leq n -1$) while maintaining the class-discriminatory information.
* In addition to finding the component axes that maximize the variance of data, LDA also maximize the separation between multiple classes. 



# Additional Readings

Comparison of PCA and LDA

https://sebastianraschka.com/Articles/2014_python_lda.html

https://towardsdatascience.com/dimensionality-reduction-for-data-visualization-pca-vs-tsne-vs-umap-be4aa7b1cb29

