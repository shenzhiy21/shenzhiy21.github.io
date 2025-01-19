+++
title = 'Notes on Linear Attention'
date = 2025-01-18T09:00:00+08:00
draft = false
math = true
tags = ['note', 'machine learning', 'system']
categories = ['note']
summary = "Notes on topics related to linear attention."

+++

## 写在前面

写这篇 blog 的动机是，最近 [MiniMax-O1](https://arxiv.org/abs/2501.08313) 把 linear attention model scale up 起来了，虽然他们是第一个吃螃蟹的人、这个模型对 long-text 的表现据说一般，但我感觉 efficient attention 大概是 LLM 未来发展的一个方向。此外还有 Google 的 [Titans](https://arxiv.org/abs/2501.00663) 也是在做线性 RNN. 包括之前的 [Mamba](https://arxiv.org/abs/2312.00752) 等等。另外想借着学习 linear attention, 再补一下 MLsys 层面的一些[知识](https://github.com/fla-org/flash-linear-attention)，紧跟 [Sonta](https://sustcsonglin.github.io/) 姐姐的步伐！

## Linear Attention Blog @ 科学空间

这一节作为 intro, 来自 Jianlin Su 老师的一篇 [blog](https://spaces.ac.cn/archives/7546).

我们回顾一下 scaled-dot attention 的形式：

$$
\text{Attn}(Q, K, V) = \text{softmax}(QK^T)V
$$

这里忽略了缩放因子，且假设 $Q, K, V\in \R^{n\times d}$, 其中对于 long-context 有 $n\gg d$.

由于这个 `softmax` 的存在，矩阵乘法不能换序，必须先算出 $QK^T$, 这里的复杂度是 $O(n^2d)\approx O(n^2)$. 但是如果没有 `softmax`, 就可以先算 $K^TV$, 复杂度是 $O(nd^2)\approx O(n)$.

实际上，Attention 可以写成如下的形式：

$$
\text{Attn}(Q, K, V)_i = \frac{\sum _{j=1}^{n} e^{q_i^T k_j} v_j}{\sum _{j=1}^{n} e^{q_i^T k_j}}
$$

可以据此提出一个对 Attention 的一般定义：

$$
\text{Attn}(Q, K, V)_i = \frac{\sum _{j=1}^n \text{sim}(q_i, k_j) v_j}{\sum _{j=1}^n \text{sim}(q_i, k_j)}
$$

其中需要保证 $\text{sim}(q_i, k_j)\geq 0$. 一个想法是

$$
\text{sim}(q_i, k_j) = \phi(q_i)^T \varphi(k_j)
$$

其中取 $\phi, \varphi$ 是非负的函数。[例如](https://arxiv.org/abs/2006.16236) $\phi(x) = \varphi(x) = \text{elu}(x) + 1$.

接下来我们考虑自回归生成任务。此时每个位置的 embedding 无法和未来的 embedding 产生 attention, 相当于加上一个 mask. 对应到公式中，只需要改变 $\sum _{j=1}^n$ 为 $\sum _{j=1}^i$.

$$
\text{Attn}(Q, k, V)_i = \frac{\sum _{j=1}^i (\phi(q_i)^T \varphi(k_j))v_j}{\sum _{j=1}^i \phi(q_i)^T \varphi(k_j)} = \frac{\phi(q_i)^T \sum _{j=1}^i \varphi(k_j)v_j^T}{\phi(q_i)^T \sum _{j=1}^i \varphi(k_j)}
$$

在推理时，令 $S_i = \sum _{j=1}^i \varphi(k_j)v_j^T$, $z_i=\sum _{j=1}^i \varphi(k_j)$, 则有

$$
\text{Attn}(Q, K, V)_i = \frac{\phi(q_i)^T S_i}{\phi(q_i)^T z_i}
$$

以及递推关系

$$
S_i = S_{i-1} + \varphi(k_i)v_i^T, z_i = z_{i-1} + \varphi(k_i)
$$

说明这种 Attention 可以用 RNN 来实现。空间复杂度很低。