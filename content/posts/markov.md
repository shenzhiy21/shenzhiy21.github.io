+++
title = 'Perron-Frobenius Theorem and the Stationary Distribution for Markov Chains'
date = 2026-01-24T14:00:00+08:00
draft = false
math = true
tags = ['diffusion', 'math', 'note', 'stats']
categories = ['note']
summary = "Proof of the uniqueness of stationary distribution for finite irreducible Markov chains, using Perron-Frobenius Theorem"

+++

在随机过程课程中，我们学习了这样的一个定理：

Let $\\{X_n\\}_{n\geq 0}$ be a discrete-time Markov chain defined on a finite state space $S = \\{1,2,\cdots,N\\}$. Let $P$ be the transition probability matrix. Assume the Markov chain is irreducible. Then, there exists a **unique** stationary distribution $\pi$ which is **strictly positive** ($\pi_i >0$ for all $i$).

这个定理的证明可以使用线性代数课上学过的 Perron-Frobenius 定理。下面记录证明思路。

首先证明存在性。定义 $\Delta$ 为该空间上所有可能的概率分布的集合：

$$
\Delta = \\{ x\in \R^N\mid x_i \geq 0\text{ for all }i,\text{and }\sum_i x_i=1 \\}
$$

定义映射 $T$ 为 $T(x) = xP$. 下面证明 $T$ 是闭映射。令 $y = T(x)$. 则显然 $y_{i}\geq 0$. 并且

$$
\sum_{j=1}^N y_j = \sum_{j=1}^N\sum_{i=1}^N x_i P_{ij} = \sum_{i=1}^N x_i\left(\sum_{j=1}^N P_{ij} \right) = \sum_{i=1}^N x_i = 1
$$

因此 $y\in\Delta$. 对于闭映射，由 Brouwer's Fixed Point Theorem 可知，存在至少一个 $\pi\in\Delta$ 使 $\pi P = \pi$.

> 事实上，注意到 $P$ 显然有一个右特征向量为 $(1/N,\cdots,1/N)$, 对应特征值为 $1$. 由左特征子空间和右特征子空间维度相同可知，必然存在特征值为 $1$ 的左特征向量 $\pi$.

接下来证明 $\pi$ 的唯一性。

由于 $\pi$ 对应于 $P$ 的特征值为 $1$ 的特征向量，所以只需证明该特征子空间维度是 1.

我们知道，左特征子空间和右特征子空间的维度是一样的，也就是：

$$
\text{dim}(\text{Null}(P^T - I)) = \text{dim}(\text{Null}(P - I))
$$

因此，任取 $v$ 满足 $P v = v$, 只需证明 $v$ 的唯一性。

我们取 $i = \arg\max_k v_k$. 记 $M = v_i$. 观察 $Pv=v$ 的第 $i$ 行：

$$
M = v_i = \sum_{j=1}^N P_{ij} v_j
$$

我们已知 $\sum_{j=1}^N P_{ij} = 1$, 并且 $v_j\leq M,\forall j$. 因此，为了使上述等式成立，必然有：

$$
\text{If }P_{ij}>0,\text{ then }v_j =M
$$

这说明：对于 $i$ 可达的状态 $j$, 必然有 $v_i = v_j = M$. 由于该 Markov chain 是不可约的 (也就是说，各个状态之间是彼此可达的)，所以可以归纳出：所有 $v_j$ 都等于 $M$.

至此，我们证明了右特征向量 $v$ 的唯一性，也推出了左特征向量 $\pi$ 的唯一性。

最后，我们证明 $\pi$ 是 strictly positive 的。假设存在 $k$ 使 $\pi_k = 0$. 代入 $\pi P = \pi$ 可知：

$$
\pi_k = \sum_{j=1}^N \pi_j P_{jk} = 0
$$

因此，对任意的 $j$, 如果 $P_{jk} >0$, 就必然有 $\pi_j = 0$. 这意味着，任何单步可达 $k$ 的状态的平稳概率都是 $0$. 根据不可约性可知，所有状态平稳概率都是 $0$, 不满足概率分布的定义 ($\sum \pi_i = 1$). 从而导出矛盾！

> 这里证明 unique 和 strictly positive 的过程是非常相似的，类似于前向和后向的 Kolmogorov 方程。我们也从中看出了线性代数的 eigenspace 理论和 Markov chain 理论的联系。虽然整个证明过程没有用到什么高深的线性代数/概率论技巧，但是证明得十分精致优美。是以记之。
