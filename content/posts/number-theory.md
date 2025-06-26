+++
title = 'What if Each Number is a Polynomial?'
date = 2025-06-26T13:00:00+08:00
draft = false
math = true
tags = ['math']
categories = ['math']
summary = "关于数论的一些很久以前的小想法 / Some naive ideas about number theory years ago ..."

+++

总觉得应该把这个想法记录下来，否则老是忘记。高中学竞赛的时候就开始想这回事了，大一入学的时候曾经幻想本科所学的数学知识可以或多或少帮助我看清这个问题的本质，但是没想到一下子过了这么多年，也并没有什么新的进展或是发现。或许有研究数论方向的读者可以告诉我，我想表达的这个概念是否已经被前人研究过了呢？如果是这样的话就太好啦！

大致的 motivation 如下：数论的研究对象是数（特别地，最常见的研究对象是正整数），然而一般来说数只是数，不是一个表达能力更广泛的函数 or 映射 or 集合 or whatever，因此不太容易自然地使用我们更习惯的分析工具 (*e.g.* Calculus) 对其进行研究。所以，我希望构造一个函数式的工具来辅助研究数论问题。

这个小想法是这样的：任取一个正整数 $A$，它可以表示为

$$
A = p_1^{\alpha_1} p_2^{\alpha_2}\cdots p_n^{\alpha_n}
$$

其中 $2\leq p_1 < \cdots < p_n$ 是 $A$ 的质因子，$\alpha_1,\cdots,\alpha_n$ 是对应的幂次。

对每个 $A\in \N^+$，都可以导出一个唯一的多项式 $F_A$：

$$
F_A(x) = (x - p_1)^{\alpha_1} \cdots (x - p_n)^{\alpha_n}
$$

它具有一些显而易见的良好的性质：

1. $\left |F_A(0)\right |=A$
2. $A_1 | A_2 \iff F_{A_1} | F_{A_2}$
3. 对于质数 $p$, 我们有 $p | A \iff F_A(p) = 0$

其中，性质 1 告诉我们 $A$ 和 $F_A$ 之间的联系；性质 2 说明 $F_A$ 在整除关系上和 $A$ 具有相同的性质和结论。其实【多项式】本身也是数论领域的一个常见的研究对象，因此这个构造方式也容易带来一些优美的性质。

此外，可以对 $F_A$ 求导：

$$
F_A'(x) = \sum_{i=1}^n \frac{\alpha_i}{x - p_i} F_A(x)
$$

对某个 $i$, 若 $\alpha_i > 1$，那么上述和式的每一项都具有一次多项式因子 $x - p_i$. 因此有

$$
F_A'(p_i) = \begin{cases}
0, &\alpha_i > 1\\\
F_{\frac{A}{p_i^{\alpha_i}}}(p_i), &\alpha_i = 1
\end{cases}
$$

当然，也可以继续对 $F_A$ 求高次导数，或许还有关于幂次的一些更广泛的性质。

接下来，我们尝试用 $F_A$ 的视角来重新解决初等数论中的一些小问题。

> To Be Continued ...