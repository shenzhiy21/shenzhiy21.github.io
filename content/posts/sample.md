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

## Simple Distributions

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
f_{y_1, y_2}(y_1, y_2) = f_{z_1, z_2}(z_1, z_2) \left|\frac{\partial(z_1, z_2)}{\partial(y_1, y_2)} \right|
=\left[\frac{1}{\sqrt{2\pi}}\exp(-y_1^2/2) \right] \left[\frac{1}{\sqrt{2\pi}}\exp(-y_2^2/2) \right]
$$

This means that $y_1$ and $y_2$ follow the Gaussian distribution $\mathcal N(0, 1)$ and are independent.