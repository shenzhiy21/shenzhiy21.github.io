+++
title = 'Alias Method for Sampling from Discrete Distribution'
date = 2024-04-13T13:00:00+08:00
draft = false
math = true
tags = ['code', 'algorithm', 'stats', 'math']
categories = ['code']
summary = "An Introduction to Alias Method"

+++

## Introduction

Alias method 是一个统计学的抽样方法，下文所写的主要都是统计学上的原理和实现上的算法和证明。但是 alias method 的思想也可以用在其它领域，例如用于压缩算法 `rANS` 在 decode 阶段的高效实现。我们会在文章的最后简单介绍一下 `rANS`.

## Problem Setting & Naive Solution

考虑这样一个离散分布 $P(X=i)=p_i$ where $0,1,\cdots,n-1$ and $\sum_i p_i=1$. 现在考虑如何高效地从中抽样。这显然是个相当 fundamental 的问题，所以读者可以自行查阅统计学相关文献。这里为了引出 alias method, 我们讲一种很 simple 但是很巧妙的处理方法。

假设我们的概率分布不是 $p_i$, 而是**频率**分布 $f_i$. 其中每个 $f_i$ 都是整数, $\sum_i f_i=M\in\N^+$. 设 $\max f_i=k$. 想象这样一个 histogram: 横坐标为 $0,1,\cdots,n-1$, 纵坐标为 $f_i$, 且纵坐标的范围是 $0,1,\cdots,k-1$. 我们把每个横纵坐标皆为整数的点看成一个 grid, 于是有 $nk$ 个 grid. 现在在 $I=\{0,1,\cdots,nk-1 \}$ 这个离散均匀分布上抽样。

> 注意：可以假设我们已经实现了一个离散均匀分布上的 (pseudo) random number generator, 例如使用线性同余生成器 (Linear Congruential Generator).

假设抽到的数是 $a$. 它的横纵坐标分别是 $x,y$. 也就是 $a=kx+y$. 神奇的事情发生了：我们可以轻而易举地由这个均匀分布过渡到我们关心的分布！做法如下：考虑 $y$ 和 $f_x$ 的关系。若 $y\leq f_x$, 就 accept 此次抽样，并返回 $x$. 若 $y>f_x$, 就 reject 此次抽样。这也算是 accept & reject (舍选法) 的典型应用了。

这个算法的正确性非常 straightforward: $a$ 的横坐标 $x$ 取值为 $0,1,\cdots,n-1$ 是等概率事件，而对每个 $i$, 我们有
$$
\mathbb P(\text{accept }i)=\mathbb P(x=i\wedge y<f_i)=\frac{f_i}{nk}\propto p_i
$$
所以 accept 的概率是
$$
\mathbb P(\text{accept})=\sum_i \mathbb P(\text{accept }i)=\frac{\sum_i f_i}{nk}=\frac{M}{nk}
$$
所以有 $1-\dfrac{M}{nk}$ 的概率 reject. 换句话说，必须采样 $(1-\dfrac{M}{nk})^{-1}$ 次才能 accept 一次。

在某些情况下，这个结果是不能令人满意的。例如当 $f_0=k,f_1=f_2=\cdots=f_{n-1}=1$ 时，
$$
\mathbb P(\text{reject})=1-\frac{k+n-1}{nk}=\frac{(n-1)(k-1)}{nk}\approx 1
$$
也就是几乎不能命中。

从几何直观上看，直方图中的“空白”越多，拒绝的概率越高。所以一个自然的想法是：是否可以把直方图中的某一些柱子“拼起来”，尽量减少空白？

## Alias Method

正在施工……