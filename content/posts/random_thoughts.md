+++
title = 'Random Thoughts & Ideas'
date = 2025-01-15T13:00:00+08:00
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