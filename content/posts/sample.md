+++
title = 'Sampling, sampling, and sampling!'
date = 2026-04-15T10:00:00+08:00
draft = false
math = true
tags = ['math', 'note', 'stats']
categories = ['note']
summary = "Sampling methods, from rejection sampling to Markov Chain Monte Carlo."

+++

It's an interesting topic to sample from a "weird" distribution,
_e.g._, high dimensional, with an unknown normalization term, or even lack of a well-formed math expression.
This is widely used in deep learning, such as the inference-time sampling of LLMs, or the denoising process of diffusion models.

To study this problem, let's first consider some easy cases and then move on to more difficult scenarios.

> Note: this article is a summary of Chapter 11 in the "PRML" book.

## Transform Sampling

Suppose we already know how to sample from the uniform distribution $z \sim U(0, 1)$.
Now consider sampling from the exponential distribution $y \sim \mathcal E(\lambda)$, *e.g.*, $f_y(y) = \lambda \exp(-\lambda y)$ where $y \geq 0$.
To sample from $y$, a strategy is to first sample from $z$ and then transform the result to follow the distribution of $y$.
That is, we want to find a function $g: [0, 1] \mapsto [0, +\infty)$ s.t. $g(z) \sim \mathcal E(\lambda)$.
Let's do some basic calculation:

$$
\begin{aligned}
& f_y(y) = f_z(z) \left|\frac{\mathrm dz}{\mathrm dy} \right| = \left|\frac{\mathrm dz}{\mathrm dy} \right|\\\
\Rightarrow{} & z = \int_{-\infty}^y f_y(\hat y)\mathrm d\hat y = 1 - \exp(-\lambda y) \\\
\Rightarrow{} & y = -\lambda^{-1} \ln (1-z) = g(z)
\end{aligned}
$$

Then $y$ will have the desired exponential distribution.

---

This strategy can be easily generalized to multivariate cases.
An example is the **Box-Muller method** to generate Gaussian distributions.
First, we sample a pair of two numbers $z_1, z_2 \sim U(-1, 1)$, and discard the pair if $z_1^2 + z_2^2 > 1$.
This leads to a uniform distribution of 2D points inside the unit circle with distribution $f_{z_1,z_2}(z_1, z_2) = 1/\pi$.
Then, we compute

$$
y_1 = z_1 \sqrt{\frac{-2 \ln(s)}{s}}, \quad y_2 = z_2 \sqrt{\frac{-2 \ln(s)}{s}}
$$

where $s = z_1^2 + z_2^2$. We have

$$
\begin{aligned}
&f_{y_1, y_2}(y_1, y_2) = f_{z_1, z_2}(z_1, z_2) \left|\frac{\partial(z_1, z_2)}{\partial(y_1, y_2)} \right| \\\
={}&\left[\frac{1}{\sqrt{2\pi}}\exp(-y_1^2/2) \right] \left[\frac{1}{\sqrt{2\pi}}\exp(-y_2^2/2) \right]
\end{aligned}
$$

This indicates that $y_1$ and $y_2$ both follow the Gaussian distribution $\mathcal N(0, 1)$ and are independent.

However, this strategy needs a carefully-designed transform function, which is not easy to construct for most distributions in practice (*e.g.*, if the integral cannot be expressed in a simple form).
We therefore need a more general and robust sampling strategy.

## Rejection Sampling

Suppose we want to sample from a target distribution $f(x)$.
For rejection sampling, we need a **proposal distribution** $g(x)$ and a constant $M \geq 1$, such that $M g(x) \geq f(x)$ for all $x$ (thus, $g(x)$ is also called the **envelope distribution**).
The algorithm proceeds as follows:

1. Sample $x \sim g(x)$
2. Sample $u \sim U(0, 1)$
3. If $u \leq \frac{f(x)}{Mg(x)}$, accept $x$ as a valid sample from $f(x)$
4. Otherwise, reject $x$ and go back to step 1

We can see that, the probability of accepting a sample is $\frac{1}{M}$.
This indicates that to optimize the sampling efficiency of this algorithm, the proposal distribution $g(x)$ must closely approximate the shape of the target distribution $f(x)$.

## Adaptive Rejection Sampling

In some cases, it's difficult to determine the proposal distribution $g(x)$.
To address this, Gilks and Wild (1992) proposed *adaptive rejection sampling*, a strategy that can dynamically constructs and iteratively refines the proposal distribution.
This strategy requires the target distribution $f(x)$ to be **log-concave** (*i.e.*, $\log f(x)$ is concave).
This requirement is satisfied for almost every commonly-seen distributions.

For log-concave distributions, the *piecewise tangent lines* naturally form an envelope distribution.
This property makes it easy to perform rejection sampling on the fly.

Suppose we have already sampled a set of values $\\{(x_1, f(x_1)), (x_2, f(x_2)), \cdots, (x_n, f(x_n)) \\}$.