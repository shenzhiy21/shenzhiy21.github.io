+++
title = 'Notes on Linear Attention'
date = 2025-01-18T13:00:00+08:00
draft = false
math = true
tags = ['note', 'machine learning', 'system']
categories = ['note']
summary = "Notes on topics related to linear attention."

+++

## 写在前面

写这篇 blog 的动机是，最近 [MiniMax-O1](https://arxiv.org/abs/2501.08313) 把 linear attention model scale up 起来了，虽然他们是第一个吃螃蟹的人、这个模型对 long-text 的表现据说一般，但我感觉 efficient attention 大概是 LLM 未来发展的一个方向。此外还有 Google 的 [Titans](https://arxiv.org/abs/2501.00663) 也是在做线性 RNN. 包括之前的 [Mamba](https://arxiv.org/abs/2312.00752) 等等。另外想借着学习 linear attention, 再补一下 MLsys 层面的一些[知识](https://github.com/fla-org/flash-linear-attention)，紧跟 [Sonta](https://sustcsonglin.github.io/) 姐姐的步伐！

