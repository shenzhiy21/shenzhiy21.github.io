+++
title = 'Machine Learning Compilation'
date = 2025-01-15T13:00:00+08:00
draft = false
math = true
tags = ['note', 'machine learning', 'system']
categories = ['note']
summary = "Lecture Notes on ML Compilation by Tianqi Chen"

+++

## Course information
- [Course website](https://mlc.ai/zh/)
- Instructor: Tianqi Chen
- [Official notes](https://mlc.ai/zh/chapter_introduction/index.html)

---

## 概述

机器学习编译 (machine learning compilation, MLC) 是指，将机器学习算法从开发阶段，通过变换和优化算法，使其变成部署状态。

MLC 的关键元素: tensor 和 tensor functions.

Takeaway msg: **值得思考 abstraction 和 implementation.** 例如 MLC 做的大部分事情就是 tensor function 在不同层面的 abstraction 之间的变换。

## 张量程序抽象

- 元张量函数表示机器学习模型计算中的单个单元计算。
  - 一个机器学习编译过程可以有选择地转换元张量函数的实现。

- 张量程序是一个表示元张量函数的有效抽象。
  - 关键成分包括: 多维数组，循环嵌套，计算语句。
  - 程序变换可以被用于加速张量程序的执行。
  - 张量程序中额外的结构能够为程序变换提供更多的信息。

引入 `TensorIR` 的一个重要原因是, ML 的开发流程和传统的程序开发有一些不同, ML (system) 的开发流程往往是先搞一个初步的实现，再通过**变换 (transformation)** 生成不同的 `TensorIR` 变体，进行 tuning.