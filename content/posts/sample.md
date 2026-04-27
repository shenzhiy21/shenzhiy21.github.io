+++
title = 'Sampling, sampling, and sampling!'
date = 2026-04-15T10:00:00+08:00
draft = false
math = true
tags = ['math', 'note', 'stats']
categories = ['note']
summary = "Sampling methods, from rejection sampling to Markov Chain Monte Carlo."

+++

> Note: This article is a summary of Chapter 11 in *Pattern Recognition and Machine Learning* (PRML).

Sampling from complex distributions, such as those that are high-dimensional, lack a known normalization term, or do not have a closed-form mathematical expressionm, is a fundamental problem in machine learning. 
These techniques are widely applied in deep learning contexts, including inference-time sampling for large language models and the denoising process in diffusion models. 
To study this problem, this article introduces foundational sampling strategies and progresses to more advanced scenarios.

## Transform Sampling

Assume we can draw samples from a uniform distribution, $z \sim U(0, 1)$. 
Consider the objective of sampling from an exponential distribution, $y \sim \mathcal{E}(\lambda)$, where the probability density function is $f_y(y) = \lambda \exp(-\lambda y)$ for $y \geq 0$. 
One approach is to sample $z$ and apply a transformation such that the result follows the distribution of $y$. 
Specifically, we seek a function $g: [0, 1] \mapsto [0, +\infty)$ such that $g(z) \sim \mathcal{E}(\lambda)$. 
The transformation is derived as follows:

$$
\begin{aligned}
& f_y(y) = f_z(z) \left|\frac{\mathrm dz}{\mathrm dy} \right| = \left|\frac{\mathrm dz}{\mathrm dy} \right|\\\
\Rightarrow{} & z = \int_{-\infty}^y f_y(\hat y)\mathrm d\hat y = 1 - \exp(-\lambda y) \\\
\Rightarrow{} & y = -\lambda^{-1} \ln (1-z) = g(z)
\end{aligned}
$$

Through this transformation, $y$ follows the target exponential distribution.

---

This strategy generalizes to multivariate distributions. 
A standard example is the **Box-Muller method** for generating Gaussian variables. 
First, we draw a pair of values $z_1, z_2 \sim U(-1, 1)$ and discard the pair if $z_1^2 + z_2^2 > 1$. 
This yields a uniform distribution of 2D points within the unit circle, characterized by the density $f_{z_1,z_2}(z_1, z_2) = 1/\pi$. We then compute:

$$
y_1 = z_1 \sqrt{\frac{-2 \ln(s)}{s}}, \quad y_2 = z_2 \sqrt{\frac{-2 \ln(s)}{s}}
$$

where $s = z_1^2 + z_2^2$. The joint distribution is given by:

$$
\begin{aligned}
&f_{y_1, y_2}(y_1, y_2) = f_{z_1, z_2}(z_1, z_2) \left|\frac{\partial(z_1, z_2)}{\partial(y_1, y_2)} \right| \\\
={}&\left[\frac{1}{\sqrt{2\pi}}\exp(-y_1^2/2) \right] \left[\frac{1}{\sqrt{2\pi}}\exp(-y_2^2/2) \right]
\end{aligned}
$$

This result demonstrates that $y_1$ and $y_2$ are independent and both follow the standard normal distribution $\mathcal{N}(0, 1)$.

However, transform sampling requires an analytically tractable cumulative distribution function, which is difficult to construct for most distributions in practice (e.g., if the integral lacks a closed form). 
Consequently, more general and robust sampling frameworks are required.

## Rejection Sampling

To sample from a target distribution $f(x)$ using rejection sampling, we introduce a **proposal distribution** $g(x)$ and a constant $M \geq 1$ such that $Mg(x) \geq f(x)$ for all $x$. 
In this context, $Mg(x)$ is called the **envelope distribution**. 
The algorithm operates as follows:

1. Sample $x \sim g(x)$.
2. Sample $u \sim U(0, 1)$.
3. If $u \leq \frac{f(x)}{Mg(x)}$, accept $x$ as a valid sample from $f(x)$.
4. Otherwise, reject $x$ and return to step 1.

The probability of accepting a sample is $1/M$. Therefore, to maximize sampling efficiency, the proposal distribution $g(x)$ must closely approximate the shape of the target distribution $f(x)$.

## Adaptive Rejection Sampling

Finding a suitable proposal distribution $g(x)$ a priori is often challenging. Gilks and Wild (1992) introduced **adaptive rejection sampling** (ARS) to dynamically construct and iteratively refine the envelope distribution. ARS requires the target distribution $f(x)$ to be **log-concave** (i.e., $\log f(x)$ is strictly concave), a property held by many standard distributions.

For log-concave distributions, piecewise linear tangents of the log-density naturally form an upper bound, facilitating efficient rejection sampling on the fly. Suppose we have evaluated the density at a set of points, yielding $S_n = \\{(x_1, f(x_1)), (x_2, f(x_2)), \dots, (x_n, f(x_n))\\}$, where $x_1 < x_2 < \dots < x_n$. We construct two piecewise linear functions to approximate $\log f(x)$:

* **Upper Envelope** $u_n(x)$: Formed by the intersection of the tangent lines to $\log f(x)$ at each $x_i$. Due to log-concavity, $u_n(x) \geq \log f(x)$ is mathematically guaranteed.
* **Lower Envelope** $l_n(x)$: Formed by the secant lines connecting adjacent points $(x_i, \log f(x_i))$ and $(x_{i+1}, \log f(x_{i+1}))$. It follows that $l_n(x) \leq \log f(x)$.

The algorithmic procedure is as follows:

1. Initialize the set $S_n$ with at least two distinct points.
2. Draw $x \sim \exp(u_n(x))$ and $u \sim U(0, 1)$.
3. **Squeeze test**: If $u \leq \exp(l_n(x) - u_n(x))$, accept $x$. This step computationally bypasses the direct evaluation of $f(x)$.
4. **Rejection test**: If the condition in step 3 is not met, evaluate $f(x)$. If $u \leq \exp(\log f(x) - u_n(x))$, accept $x$. Otherwise, reject $x$.
5. **Update**: If $f(x)$ was evaluated in step 4 (regardless of acceptance or rejection), append $(x, f(x))$ to $S_n$ and update the envelope functions to $u_{n+1}(x)$ and $l_{n+1}(x)$.

As $n$ increases, $u_n(x)$ and $l_n(x)$ converge to $\log f(x)$, systematically reducing the rejection rate.

---

Consider the application of rejection sampling in high-dimensional spaces. Suppose the target is an $n$-dimensional zero-mean Gaussian distribution with covariance $\sigma_p^2 I_n$, and the proposal is a zero-mean Gaussian with covariance $\sigma_q^2 I_n$. For rejection sampling to be valid, we must set $\sigma_q^2 \geq \sigma_p^2$ to ensure a constant $k$ exists such that $k f_q(x) \geq f_p(x)$. 

The optimal scaling constant is $k = (\sigma_q / \sigma_p)^n$, resulting in an acceptance rate of $1/k$. Consequently, **the acceptance rate decays exponentially as dimensionality increases**. While rejection sampling remains highly effective in low-dimensional settings, it becomes computationally intractable for high-dimensional distributions.