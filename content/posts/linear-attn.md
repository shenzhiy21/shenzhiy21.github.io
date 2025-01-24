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

> 这一节作为 intro, 来自 Jianlin Su 老师的一篇 [blog](https://spaces.ac.cn/archives/7546).

我们回顾一下 scaled-dot attention 在**训练**阶段的形式：

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

接下来我们考虑自回归生成任务，也就是推理阶段。此时每个位置的 embedding 无法和未来的 embedding 产生 attention, 相当于加上一个 mask. 对应到公式中，只需要改变 $\sum _{j=1}^n$ 为 $\sum _{j=1}^i$.

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

说明这种 Attention 可以用 RNN 来实现。好处是，生成每个 token 的时间复杂度为 $O(1)$, 空间复杂度也为 $O(1)$.

需要注意的是，这里的 $O(1)$ 是对于 $n$ 而言的。对于 $d$ (也就是 `head_size`) 的复杂度其实是 $O(d^2)$. 由于 $d$ 通常取值为 $32, 64, 128$ 等，所以此时还能接受。但是如果对于 $d$ 的复杂度达到 $\Theta(d^3)$, 就不太能接受了：例如 $64^3=262,144$, 已经超过了 `128k` 的 sequence length, 没法忽略。

## Modern RNN

> 这一节参考了 Sonta 姐姐的 [talk](https://www.bilibili.com/video/BV1MDwAeWEoM/). 我从 Sonta 身上学到的最重要的一点就是，设计这些新的模型架构，一方面需要算法方面的 insight, 另一方面也需要深刻体会 system 层面如何高效实现你提出的算法（甚至算法本身也可以去 align 系统的计算优劣势）。

### Chunkwise Linear Attention

其实，苏老师说的这个 non-negative similarity function $\text{sim}(\cdot, \cdot)$ 不要也行。或者说，为什么不允许 attention score 有负值呢？所以，接下来我们考虑 linear attention 的一种最简单的情形：$\text{sim}(q_i, k_j) = q_i^Tk_j$.

在 training 阶段，有两种方式：parallel form 和 recurrent form. 形式分别为（注意 parallel form 加上了 mask）：

$$
O=(QK^T\odot M)V\in\R^{n\times d}
$$

以及

$$
\begin{aligned}
S_t &= S_{t-1} + v_tk_t^T \in \R^{d\times d}\\\
o_t &= S_t q_t \in \R^d
\end{aligned}
$$

他们各自的 pros and cons 在于：
- Parallel form
  - pro: 矩阵乘法的形式，GPU 友好，方便用 tensorcore 加速
  - con: 时间复杂度为 $O(n^2)$, 没有解决长文本的高复杂度问题
- Recurrent form
  - pro: 时间复杂度 $O(n)$, 空间复杂度 $O(1)$
  - con: 难以并行计算，此外没有 pure matmul 运算，GPU 利用率差

所以一个自然的想法是能不能把它们的优势结合起来，也就是 "chunkwise".

> 我记得当时在做 ANS 的加速的时候也遇到了这个问题。甚至可能所有 sequential form 的计算逻辑都会在试图并行加速时遇到这个问题。但是 ANS 的一个不同之处是，只有 recurrent form, 至少我没推出 parallel form. 但当时我们的并行方案包括：
> 1. 对整个 sequence 分块 (block) 并行；
> 2. 每个 block 内部，以 32bit (而非原本针对 ASCII 的 8bit) 为读写单元，也就是把 4 个 ASCII 字符看成一个 chunk.
> 
> 其实是和 linear attention 有异曲同工之处的。

在 notation 上为了和原文保持一致，我们下面把 sequence length 记为 $L$ 而非 $n$.

把 $L$ 分成 $L/C$ 个 chunk, 在每个 chunk 内部采用 parallel form, 而对于历史信息采用 recurrent form.

Notation:

$$
\begin{aligned}
S_{[i]} &:= S_{iC}\in\R^{d\times d}\\\
\square_{[i]} &:= \square_{iC+1:(i+1)C}\in \R^{C\times d},\text{for } \square\in{Q,K,V,O}
\end{aligned}
$$

状态矩阵 $S$ 的递推关系为：

$$
S_{[t+1]} = S_{[t]} + V_{[t]}^TK_{[t]}
$$

Attention 计算公式为：

$$
O_{[t]} = Q_{[t]}S_{[t]}^T + (Q_{[t]}K_{[t]}^T\odot M)V_{[t]}
$$

这就是最基本的 chunkwise linear attention 形式。

### Decay Term

$S_t$ 可以视为当前状态的一种表示，然而我们发现，对 $S_t$ 的更新方式 $S_t = S_{t-1} + v_t k_t^T$ 很容易跑飞。一种修正方式是加上 decay term $\gamma$:

$$
S_t = \gamma S_{t-1} + v_t k_t^T
$$

在实际应用 (RetNet, Lightning Attention) 中，加上 decay term 之后效果就会提升不少。

更一般地，$\gamma$ 未必需要是定值。我们可以为它加上 "selectivity" (data-dependent decay, *e.g.* Mamba2, mLSTM, Gated Retention):

$$
S_t = \gamma_t S_{t-1} + v_t k_t^T
$$

不同的 $\gamma_t$ 可以某种程度上“控制”模型的记忆和遗忘。

更一般地，$\gamma$ 为什么一定得是一个 scalar 呢？为什么不能是一个 tensor 呢？例如，$G_t\in \R^{d\times d}$:

$$
S_t = G_t \odot S_{t-1} + v_t k_t^T
$$

### Delta Rule

另一方面，我们可以重新审视一下 linear attention 对 $S_t$ 的更新：

$$
S_t = S_{t-1} + v_t k_t^T
$$

这可以看成对如下的优化目标的单步优化：

$$
\mathcal L_t(S) = -\lang S k_t, v_t\rang
$$

使用 gradient descent:

$$
\begin{aligned}
S_t &= S_{t-1} - \beta_t \nabla \mathcal L_t(S_{t-1})\\\
&= S_{t-1} + \beta_t v_t k_t^T
\end{aligned}
$$

如果不用 dot product, 而是以 regression loss 作为优化目标：

$$
\mathcal L_t(S) = \frac{1}{2} \|\| S k_t - v_t \|\|^2
$$

同样使用 gradient descent 可以得到新的递推公式 (DeltaNet):

$$
\begin{aligned}
S_t &= S_{t-1} - \beta_t (S_{t-1}k_t - v_t) k_t^T\\\
&= S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T
\end{aligned}
$$

### Gated DeltaNet

如果我们把 Mamba2 (gated update rule) 和 DeltaNet (delta update rule) 结合起来，就可以得到：

$$
S_t = S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T
$$

其中 $\alpha_t$ 的计算方式参考 Mamba2.

Sonta 他们做了 single needle in a haystack (海底捞针) 这个任务上面的实验。简单介绍一下这个 task. 它分为三个 level:

- Lv1: 一段合成的长文本中，插入一个 magic number. 模型需要记忆并输出 magic number.
- Lv2: 不再用合成文本，而是真实文本。
- Lv3: magic number 的 pattern 不再是简单的 0-9 数字，而是更复杂的 uuid.

大概的结论是：

- Decay term in Mamba2 hurts memory retention; Delta rule memorizes better.
- Data-dependent decay helps filter out irrelevant information; Delta rule fail to filter.
- Gated DeltaNet performs best.

## Accelerating DeltaNet

这一节参考了 Sonta 的 blog [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-2/).

我们考虑如下的事情：对于 DeltaNet 的状态更新

$$
\begin{aligned}
S_t &= S_{t-1}\left(I - \beta_t k_t k_t^T\right) + \beta_t v_t k_t^T\\\
&= \sum_{i=1}^t \left(\beta_i v_i k_i^T \prod_{j=i+1}^t \left(I - \beta_j k_j k_j^T \right) \right)
\end{aligned}
$$

如何找到一个关于 $d$ 是平方复杂度的算法？

