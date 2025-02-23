+++
title = 'Random Thoughts & Ideas'
date = 2025-02-06T13:00:00+08:00
draft = false
math = true
tags = ['idea', 'note']
categories = ['note']
summary = "碎碎念"

+++

这里将随机留下一些想法和碎碎念。写给自己看的。
- 为什么要写下来呢？为了方便之后回看，以免忘记。
- 为什么要公开呢？因为可以督促我写下来。如果可以给其他好事者一些启发就更好了。

## Abstraction: Tree or Graph?

最近因为做 composite visualitazion, 所以经常会思考 tree 和 graph 之间的区别。例如，
- 如果要刻画一个场景 (scene) 的结构, 到底是 scene tree 还是 scene graph?
- 刻画 composite visualization 的结构，是 tree 还是 graph?
- 对 network 的层层抽象 (app / transport / network / data link / physical layer), 看似是 tree structure, 但是是否实际上也是 graph?

今天在听 Tianqi Chen 的 [ML Compilation course](https://mlc.ai/zh/chapter_introduction/index.html), 他特别强调 abstraction 和 implementation 这两个概念，大概意思是说，特别在系统领域，往往是层层抽象 (abstraction) 的，而更底层的是对更高层的需求的实现 (implementation).

由于底层的实现可能是多样的，例如 PyTorch, TensorFlow, JAX 就是三种实现形式，所以这种 abstraction-implementation 似乎可以用一个 tree structure 来刻画：相同深度的节点代表相同层级的抽象，每个节点的孩子们代表对这层抽象的各种更底层实现。

然而，真的是 tree structure 吗？

这里有一个 observation: 对 network 而言，上述的抽象方式未必是唯一正确的，也未必是真的合理的。它可能只是便于我们加深理解，但实际的实现上可能并非真的如此。例如 ARP 协议到底归于哪个 layer? 这里其实是模糊的。

换句话说，在这个抽象关系的依赖图中，**一个节点未必只能有一个父亲**。这就自然地推导出 graph structure 了。更准确地，是 DAG structure (因为显然这种 "abstraction-implementation" 的关系是属于包含关系，必然无环).

我们也可以为 DAG 检查上述所说的 tree 具有的性质：
- 根据拓扑排序，可以为每个节点定义“深度”，相同深度的节点代表相同层级的抽象；
- 每个节点的孩子们代表对这层抽象的各种更底层实现。

不管对于 scene 还是 composite visualization 的建模，其实都有在 tree or graph 的选择中犹豫过。在 composite visualization 中，我们目前的想法是，可以从一个 DAG 中抽出若干个树，它们每个都是这个 DAG structure 的一个“实例”。

这样的想法似乎也可以用到其它领域。

## "Predicting" and "Understanding"

Source: Machine Learning: A Probabilistic Perspective

> Over the passed years, deep learning focused more too much on the “prediction of observable patterns”, but not enough on “understanding” the underlying latent structure behind these patterns.

## Knowledge Transfer

最近在读 [PagedAttention](https://arxiv.org/pdf/2309.06180), 这篇论文的 idea 其实是比较简单的，但是它促成了 [vLLM](https://github.com/vllm-project/vllm) 的发展，足见一个学术上的想法是如何慢慢转变为工业界的产品。其实 PagedAttention 就是一种 memory management 方案，并且完全借鉴了操作系统课上学过的页式内存管理。只不过，OS 管理的是各个**进程**占用的内存，而 LLM 管理的是各个**序列**的 KV cache.

PagedAttention 如此成功的原因是什么呢？我觉得不在于算法层面的创新——毕竟页式内存管理的算法谁都会写——而是最初的那个 observation: LLM 的推理就和 OS 一样，在显存管理上会出现大量的 fragments.

一旦注意到了这一点，其实也就知道如何用 OS 那边的老方法来解决 LLM 这边的新问题了。

我觉得这样的“知识迁移”是挺值得我们学习的。同时，写这一段话也是希望自己可以多回顾一下以前学过的一些经典的算法，不要觉得和自己的领域没什么关系，说不定哪天就能用上了呢。你看 vLLM, 对于 beam search 场景下的 KV cache 管理，就还是参考了 OS 内存管理中的 COW (copy-on-write), 建议把 vLLM 纳入 OS 课程的 course project (bushi).

最后推荐一下关于 Paged Attention 的一个 [知乎专栏](https://zhuanlan.zhihu.com/p/720157057) 以及一个 [GitHub 项目](https://github.com/tspeterkim/paged-attention-minimal)。