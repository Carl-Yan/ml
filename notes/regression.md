# Regression 回归问题

> 属于监督学习的一个范畴，用于训练一个预测器: "Predict real-valued output."
>
> 但是Logistic Regression却是用于处理分类问题……???

[TOC]

## Linear Regression

### Problem Formulation

For linear regression with one variable:

**Hypothesis**: 
$$
h_\theta(x)=\theta_0+\theta_1x
$$
**Parameters**: 
$$
\theta_0, \theta_1
$$
**Cost Function**: 
$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$
$m$为样本数，$x^{(i)}$表示第$i$个样本的值

**Goal**: 
$$
\min_{\theta_0,\theta_1} J(\theta_0,\theta_1)
$$
$J(\theta_0,\theta_1)$为凸函数

-----

For linear regression with multiple variables:

**Hypothesis**: 
$$
h_\theta(x)=\theta_0x_0+\theta_1x_1+\cdots+\theta_nx_n=\theta^Tx
$$
Here, $x_0=1$ (bias unit), $\theta^T=[\theta_0, \theta_1,\cdots, \theta_n]$ 

**Cost Function**: 
$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$
$m$ examples, $x^{(i)}$ is the $i_{\mathbb{th}}$ example

- Vectorization: `J=sum((X*theta-y).^2)/2/m;`

**Goal**: 
$$
\min_{\theta} J(\theta)
$$
$J(\theta)$不一定为凸函数

### Gradient Descent

$$
\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j}J(\theta)
$$

$$
\frac{\partial}{\partial\theta_0}J(\theta)=\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})
$$

($x_0=1$，乘了等于没乘)
$$
\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^m (h_{\theta}(x^{(i)})-y^{(i)})\cdot x_j^{(i)}
$$
, **for both linear regression and logistic regression**. Simultaneously update $\theta_j$ for all $j$.

- Vectorization: `partial_j=((X*theta-y)'*X)(:)/m;`

### Normal Equation 

The closed-form solution to linear regression is
$$
\theta=(X^TX)^{-1}X^Ty
$$

Using this formula does not require any feature scaling, and you will get an exact solution in one calculation: there is no “loop until convergence” like in gradient descent.

## Logistic Regression	

### Problem Formulation

**Hypothesis**: 
$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
常被写作如下形式：
$$
h_\theta(x)=g(\theta^Tx)
$$
, where
$$
g(z)=\frac{1}{1+e^{-z}}
$$
is the sigmoid function.

特性：

- $0\le h_\theta(x)\le 1$
- $g'(z)=g(z)(1-g(z))$
- predict 1 $\Leftrightarrow$ $h_\theta(x)\ge 0.5$ $\Leftrightarrow$ $g(\theta^Tx)\ge 0.5$ $\Leftrightarrow$ $\theta^T x\ge 0$

**Cost Function**: 
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m\Big[y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\Big]
$$
**Goal**: 
$$
\min_{\theta} J(\theta)
$$
$J(\theta)$不一定为凸函数?

**Output**: 
$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}= p(y=1|x;\theta)
$$

### Gradient Descent

$$
\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{m}\sum_{i=1}^m (h_{\theta}(x^{(i)})-y^{(i)})\cdot x_j^{(i)}
$$

推导过程见`cs229-notes1.pdf` Page 18

- Vectorization

  ```matlab
  % machine-learning-ex2/ex2/costFunction.m
  h=sigmoid(X*theta);
  J=-(y'*log(h)+(1-y')*log(1-h))./m;
  grad=(h-y)'*X./m;
  ```

## Advanced Optimization Algorithm

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

听说后面三个根本没有learning rate这玩意……而且更快

## Regularization

在原来的$J(\theta)$后面加一项$\lambda\sum_j\theta_j^2$，防止$\theta_j$过大的情况(overfitting)

- $\lambda$过大：$\theta$要保持尽量小才能最小化$J(\theta)$，会欠拟合(underfitting, high bias)
- $\lambda$过小：和没加一样，会过拟合(overfitting, high varience)

### Regularized Linear Regression

**Cost Function**: 
$$
J(\theta)=\frac{1}{2m}\Big[\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2+\lambda\sum_{j=1}^n\theta_j^2\Big]
$$
**Gradient Descent**: 
$$
\frac{\partial}{\partial\theta_j}J(\theta)=\Big(\frac{1}{m}\sum_{i=1}^m (h_{\theta}(x^{(i)})-y^{(i)})\cdot x_j^{(i)}\Big)+\frac{\lambda}{m}\theta_j ~~~~~~~j\in \{1,2,\cdots,n\}
$$

$$
\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

**Normal Equation**: 
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$

### Regularized Logistic Regression

**Cost Function**: 
$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)})) \Big]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$

- Vectorization

  ```matlab
  % machine-learning-ex2/ex2/costFunctionReg.m
  h=sigmoid(X*theta);
  J=-(y'*log(h)+(1-y')*log(1-h))./m+lambda/2./m*(theta'*theta-theta(1)^2);
  grad(1)=(h-y)'*X(:,1)./m;
  grad(2:end)=(h-y)'*X(:,2:end)./m+theta(2:end)'.*lambda/m;
  ```

