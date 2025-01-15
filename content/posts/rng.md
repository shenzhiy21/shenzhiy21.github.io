+++
title = 'Random Number Generator'
date = 2024-04-20T10:46:52+08:00
draft = false
math = true
tags = ['code', 'stats', 'algorithm']
categories = ['code']
summary = "Lecture notes on RNG"
+++

## Abstract

RNG (Random Number Generator? Royal Never Give-up!) 或者说（伪）随机数生成器，在统计计算中很常见。我们经常在 $\mathcal U(0,1)$ 均匀分布或者 $\mathcal N(0,1)$ 正态分布中进行采样。然而，我们不满足于当一个调包侠，所以自然会好奇：这些分布是怎么生成的呢？（怎么确保生成的随机数服从我们想要的分布？）所以，本文将从零开始（不使用 Python 的任何包含随机数的库，例如 `random`, `numpy`）写一个随机数生成器，实现常见的概率分布。

下面的内容 follow 周在莹老师的 lecture slides. 下文的随机数指的都是伪 (pseudo) 随机数。

对一个随机数生成器，我们要求它具有 `__init__()` 和 `next()` 方法，后者需要返回下一个服从分布的随机数。先定义一个基类：

```python
class RandomNumberGenerator:
    """
    Basic class for random number generator.
    """
    def __init__(self):
        raise NotImplementedError("__init__() function is not implemented yet")
    
    def next(self):
        """
        Get the next random number from the generator
        """
        raise NotImplementedError("next() function is not implemented yet")
```

## Uniform Distribution

最常见的 RNG 是线性同余生成器 (Linear Congruential): 为了生成 $0,1,\cdots,m-1$ 这些整数，可以考虑
$$
X_{i+1}=aX_i+c \pmod m
$$
这样生成的 $\{X_i\}$ 序列在 $a,c,m,X_0$ 处于特定条件下时，可以取遍 $\{0,1,\cdots,m-1\}$. 详见 Hull and Dobell, 1962. 一个常见的取法是：$a=16807,c=0,m=2^{31}-1$. (Lewis, Goodman and Miller, 1969)

于是，当 $m$ 足够大时，$\{X_i/m\}$ 可以视为 $[0,1)$ 上的均匀分布。

```python
class LinearCongruentialGenerator(RandomNumberGenerator):
    """
    Generate a random integer in [0..m-1] using the linear congruential generator.
    """
    def __init__(self, seed, a, c, m):
        self.seed = seed
        self.a = a
        self.c = c
        self.m = m
    
    def next(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed


class UniformGenerator(RandomNumberGenerator):
    """
    Uniform [0, 1) random number generator using linear congruential generator.
    """
    def __init__(self, seed=1, a=16807, c=0, mod=2147483647):
        self.mod = mod
        self.rng = LinearCongruentialGenerator(seed, a, c, self.mod)
    
    def next(self):
        return self.rng.next() / self.mod
```

当然，生成均匀分布的方法有很多，这不是本文的重点。

重点是，假设我们已经有了一个 $\mathcal U(0,1)$ 的随机数生成器，如何生成其它分布呢？

## Discrete Distributions

对于离散分布，有通用的生成方法。下面给出两个引理：

**Lemma 1**: Let $U\sim \mathcal U[0,1]$ and $n\in\mathbb N$. Define a random variable $X=\lfloor nU\rfloor$, then $X\sim U\{0,1,\cdots,n-1\}$.

*Proof*. $\mathbb P(X=k)=\mathbb P(\lfloor nU\rfloor=k)=\mathbb P(nU\in [k,k+1))=\mathbb P(U\in [k/n,(k+1)/n))=1/n$.

**Lemma 2**: Assume $A=\{a_i|i\in I\}$ where either $I=\{1,2,\cdots,n\}$ for some $n\in\mathbb N$ or $I=\mathbb N$, and where $a_i\neq a_j$ for $i\neq j$. Let $p_i\geq 0$ with $\sum_I p_i=1$. Finally let $U\sim\mathcal U[0,1]$ and define $K=\min\{k\in I|\sum_{i=1}^k p_i\geq U \}$. Then $X=a_K\in A$ satisfies $\mathbb P(X=a_K)=p_k$ for all $k\in I$.

*Proof*. $\mathbb P(X=a_K)=\mathbb P(\sum_{i=1}^{k-1}p_i<U,\sum_{i=1}^k p_i\geq U)=p_k$.

根据以上引理，假设我们已经有 $U\sim \mathcal U[0,1]$ 了，那么只需要构造特定的 $\{a_i\}$ 和 $\{p_i\}$ 即可得到想要的离散分布。

例如对于几何分布而言，目标是
$$
\mathbb P(X=i)=(1-p)^{i-1}p=p_i,i\in\mathbb N^+
$$
对照 **Lemma 2** 可知，应当令 $a_i=i$, $p_i=(1-p)^{i-1}p$. 则
$$
\sum_{i=1}^k p_i=\sum_{i=1}^k (1-p)^{i-1}p=1-(1-p)^k
$$

$$
\sum_{i=1}^k p_i\geq U\Longleftrightarrow 1-(1-p)^k\geq U\Longleftrightarrow k\geq\frac{\log(1-U)}{\log (1-p)}
$$

所以，
$$
\left\lceil\frac{\log(1-U)}{\log(1-p)}\right\rceil\sim G(p)
$$

```python
class GeometricGenerator(RandomNumberGenerator):
    """ 
    Generate a random number from a geometric distribution with parameter p.
    """
    def __init__(self, uniform_generator: RandomNumberGenerator, p):
        self.rng = uniform_generator
        self.p = p
        if not isinstance(p, float):
            raise ValueError("p must be a float")
        if p <= 0 or p >= 1:
            raise ValueError("p must be in range (0, 1)")
    
    def next(self):
        U = self.rng.next()
        return math.ceil(math.log(U) / math.log(1 - self.p))
```

## Inverse Transform Method

对于连续分布，有类似的处理方法。假设 $X$ 是连续型随机变量，它的 cumulative distribution 是 $F_X(x)$. 那么 **Probability Integral Transformation** 定理告诉我们，$Y=F_X(X)\sim \mathcal U(0,1)$. 

*Proof*.
$$
F_Y(y)=\mathbb P(Y\leq y)=\mathbb P(F_X(X)\leq y)=\mathbb P(X\leq F_X^{-1}(y))=F_X(F_X^{-1}(y))=y
$$
例如，对于 Rayleigh distribution:
$$
f(x;\sigma^2)=\frac{x}{\sigma^2}\exp(-x^2/2\sigma^2),x\geq 0
$$
我们试图通过均匀分布得到该分布：
$$
F(x)=\int_0^x f(t)\mathrm dt=1-\exp(-x^2/2\sigma^2)=y
$$
则
$$
x=\sqrt{-2\sigma^2\log(1-y)}
$$
于是，
$$
\sqrt{-2\sigma^2\log (1-U)}\sim f
$$
当然，这个方法仅适用于 cumulative distribution 容易计算的分布。对于 normal distribution 这样没有累积分布的解析表达式的分布，需要考虑其它方法。

## Acceptance-Rejection Method

