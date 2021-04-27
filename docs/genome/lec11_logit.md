# Logistic Regression

* Assume two classes $C = \{0,1\}$.

* $Y \sim \text{Bernoulli}(p)$, and we would like to model $p$.

* Model: 
  $$
  \log(\frac{p}{1-p}) = \beta_0 + \beta^Tx
  $$

* We need to estimate $\beta_0$ and $\beta$, and then solve for $p$, namely
  $$
  p = \frac{\exp(\beta_0 + \beta^Tx)}{1 + \exp(\beta_0 + \beta^Tx)}
  $$

* Then we can make prediction by choosing $C=1$ if $p > 0.5$.

* LR makes less assumption than LDA since LR only makes assumption on the conditional probabilities while LDA also makes assumption on the joint probabilities (conditional and prior).

# SVM

* Given training data $(x_1, y_1), ..., (x_n, y_n)$, with $x_i \in \R^p, y_i \in \{-1,1\}$.

* If linear separable, determine hyperplane $(wx-b=0)$ that maximizes distance to the nearest point $x_i$ from each group.
  $$
  \text{minimize} \|w\|_2 \text{ such that }y_i(wx_i - b)\geq 1 \text{ for all } i
  $$

* If not linear separable, determine hyperplane $(wx - b=0)$ that maximizes distance to the nearest point $x_i$ from each group and minimizes sum of classification errors $\psi_i$:
  $$
  \text{minimize} \|w\|_2+ \lambda \sum^n_{i=1}\psi_i \text{ such that }y_i(wx_i - b)\geq 1-\psi_i \text{ for all } i
  $$
  