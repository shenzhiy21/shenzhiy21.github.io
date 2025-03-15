+++
title = 'Understanding Diffusion Models'
date = 2025-03-15T12:00:00+08:00
draft = false
math = true
tags = ['diffusion', 'math', 'note', 'stats']
categories = ['note']
summary = "From differential equation to diffusion models"

+++

## Introduction

最近在学习 diffusion 的数学原理和代码实现，目标是能自己手搓一个 image generation model. 之前对 diffusion 当然也有一些了解，主要是实习的时候做了一些系统层面的尝试，例如剪枝、量化、FFT. 然而对模型原理的认识并不是很充分（虽然也看了 ddpm, latent diffusion 这些文章）。当时主要是 follow 了这篇 tutorial: [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) 来学习数学原理的，当然作者写得非常好，但是我并没有太 get 到从 ELBO 和 VAE 开始讲起的心路历程（虽然但是也可以料想到十年左右前的 researcher 就是很曲折的吧）、以及用 Markov chain 建模的动机。最近被安利了 MIT 的这门课程 [Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/), 它是从 ODE/SDE 的角度入手和推导的，我觉得这样的心路历程更自然一些，学习之后大有醍醐灌顶之感。所以下面记录一下我学习这门课的一些 notes 和心得体会。

我追求的是所谓“美”，也就是希望这套算法背后的数学推导是足够自然的。所以，接下来的内容将花大量篇幅“考察”每一步推导的“动机”是否足够自然。如果你不喜欢这种风格，那很抱歉，这里没有只给结论的 "TL;DR".

## How to Generate Images

我们还是先想想如何 formalize 最原始的 image generation 任务（“原始”的意思是，不考虑输入额外的 prompt/image 作为 condition）。按照统计学习的基本认知，这个任务的定义是：假设有一批训练数据 $z\in \R^d$ 服从概率分布 $z\sim p_{\text{data}}$, 我们的目标是生成更多满足相同分布的数据。最直接的想法就是用 neural nets 建模并学习 $p_{\text{data}}$, 然后从中采样。然而如何用 NN 来描述一个 prob distribution 是一件很棘手的事情。为了 walk around 这个难题，diffusion & flow matching 采用的对策是：从一个简单的概率分布 $p_{\text{init}}$（例如 Gaussian）中采样 $x\sim p_{\text{init}}$, 然后考虑如何把 $x$ “变换”到服从 $p_{\text{data}}$.

> Note: Gaussian distribution $p_{\text{init}}$ 可以视作噪声，所以 diffusion 也被称为去噪 (denoise) 的过程。

这里其实描述的是这样一个问题：**给定两个概率分布 $p_{\text{init}}$ 和 $p_{\text{data}}$, 我该如何建模它们之间的“变换”关系？**（“变换”，也就是说从 $p_{\text{init}}$ 中采集的样本经过该变换可以对应到 $p_{\text{data}}$ 中的一个样本）

## Probability Path

针对上述问题，probability path 是一个还算自然的想法：我们引入时间变量 $t\in [0, 1]$, 并定义一个随时间变化的概率分布 $p_t$, 只要让它满足 $p_0 = p_{\text{init}}$, $p_1 = p_{\text{data}}$ 即可描述此变换。由于 $t$ 是连续的，所以可以想象成 $p_0$ 随着时间不断演化至 $p_1$. 这就是 probability path 名字的由来。这样的建模在数学里应该是挺常见的，例如微分几何中描述一段闭合曲线就常常用 $s_t (t\in [0, 1])$ 这样的形式。

然而，如何直接地形式化地“写出” $p_t$ 仍然是一件很麻烦的事情。什么是我们会做的呢？很自然地联想到：虽然我们不会描述概率分布之间的变换 (cuz it's stochastic), 但是我们会描述两个样本之间的变换呀 (cuz it's deterministic) ！给定两个点 $x_0, x_1\in \R^d$, 随便画一条曲线连接它们，并用 $t\in [0, 1]$ 来刻画曲线上的每一个点。太简单了！

与此同时，我们还有一整套成熟的工具来研究这个变换——常微分方程！

## ODE Background

这里复习一下 ODE 的知识，顺便统一一下后文的 notations.

ODE 可以由 vector field $u$ 来定义。

$$
u:\R^d\times [0, 1]\rightarrow \R^d,\ \  (x, t)\rightarrow u_t(x)
$$

使得

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}X_t &= u_t(X_t)\\\
X_0 &= x_0
\end{aligned}
$$

这个 ODE 的解叫做 "flow": $\psi:\R^d\times [0,1]\rightarrow \R^d$, $(x_0, t)\rightarrow \psi_t(x_0)$ 使得

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}\psi_t(x_0) &= u_t(\psi_t(x_0))\\\
\psi_0(x_0) &= x_0
\end{aligned}
$$

给定一个初始值 $X_0=x_0$, 可以根据 $\psi$ 来定义轨迹 (trajectory) $X_t = \psi_t(X_0)$, 使得它满足 vector field 的约束。

再复习一下数值分析课上学过的知识：给定 $u$ 和 $X_0=x_0$, 如何数值求解 $X_1$?

最简单的是 Euler method: 给定 $n\in \N$ 和 $h=n^{-1}$, 采取如下的 update rule:

$$
X_{t+h} = X_t + hu_t(X_t), \ \ (t = 0,h,2h,\cdots, 1-h)
$$

我们也可以用更精细的方法，例如 Heun's method:

$$
\begin{aligned}
X_{t+h}' &= X_t + h u_t(X_t)\\\
X_{t+h} &= X_t + \frac{h}{2} \left(u_t(X_t) + u_{t+h}(X_{t+h}') \right)
\end{aligned}
$$

大概的解释是：先用 Euler method 给出每个时刻 $X_{t+h}$ 的一个 initial guess $X_{t+h}'$, 再对它进行修正。

然而，在 diffusion model 中，使用简单的 Euler method 基本就够了。

## Conditional and Marginal Probability Path

那么，既然有了 ODE 的知识储备，如何解决我们最开始提出的问题呢？即：如何建模两个概率分布之间的变换？

我们还是慢慢来。现在会建模点到点的变换 (point2point) 了，那就接着试试建模从概率分布到点的变换 (dist2point), 最后推广到两个概率分布之间的变换 (dist2dist).

给定概率分布 $p_{\text{data}} = p_1$ 中的一个样本 $z\in\R^d$, 如何建模 $p_{\text{init}}$ 到 $z$ 的变换呢？也就是说，我们要做的事情是：设计一个变换过程，使得任意采样一个点 $x\sim p_{\text{init}}$, 总可以变换到给定的 $z$.

简单起见，还是假设 $p_{\text{init}} = \mathcal N(0, I_d)$. 我们可以定义如下的随时间变化的概率分布 $p_t(\cdot|z)$:

$$
p_t(\cdot|z) = \mathcal(\alpha_t z,\beta_t^2 I_d)
$$

其中 $\alpha_t,\beta_t$ 被称为 sheduler, 并且满足 $\alpha_0=\beta_1=0, \alpha_1 = \beta_0 = 1$. 我们自然有 $p_0(\cdot|z)=p_{\text{init}}$ 和 $p_1(\cdot|z) = \delta_z$. 其中 $\delta_z$ 是 Dirac delta distribution, 也就是我们变换的终点。

> 注意：这里其实是用一个特殊的概率分布 $\delta_z$ 来替换了 $z$ 这个点。所以我们已经学会了一种特殊的 dist2dist, 即 Gaussian to Dirac. 而我们定义的 $p_t$ 就是一条 probability path.

我们可以一般化地定义：A **conditional probability path** is a set of distribution $p_t(x|z)$ such that

$$
p_0(\cdot|z) = p_{\text{init}}, \ \ p_1(\cdot| z) = \delta_z, \text{  for all }z\in\R^d
$$

这里的 "conditional" 的意思是，这一条 path 必须先给定 condition $z$.

那么，一个自然的想法是：遍历 $p_{\text{data}}$ 中的“所有” $z$, 不就是我们需要的、一般化的 dist2dist 了吗！为此，我们定义 **marginal probability path** $p_t(x)$: a distribution that we obtain by first sampling a data point $z\sim p_{\text{data}}$ and then sampling from $p_t(\cdot|z)$. 形式化地：

$$
z\sim p_{\text{data}}, x\sim p_t(\cdot|z)\Rightarrow x\sim p_t\\\
p_t(x) = \int p_t(x|z)p_{\text{data}}(z)\mathrm dz
$$

这里的 "marginal" 意思就是遍历所有可能的 condition $z$, 并根据 $z$ 的分布做一次加权平均，得到的新分布。

我们可以验证：$p_0 = p_{\text{init}}$, $p_1 = p_{\text{data}}$:

$$
\begin{aligned}
p_0(x) &= \int p_0(x|z)p_{\text{data}}(z)\mathrm dz = \int p_{\text{init}}(x)p_{\text{data}}(z)\mathrm dz = p_{\text{init}}(x)\\\
p_1(x) &= \int p_1(x|z)p_{\text{data}}(z)\mathrm dz = \int \delta_z(x)p_{\text{data}}(z)\mathrm dz = p_{\text{data}}(x)
\end{aligned}
$$

所以，marginal probability path 就可以看成任意两个分布 $p_{\text{init}}$ 和 $p_{\text{data}}$ 之间的一种 interpolation.

So we are done!

## Conditional and Marginal Vector Fields

回顾一下我们现在想明白了什么东西：

- What has been done:
  - 如何形式化建模 image generation task：先从已知分布 $p_{\text{init}}$ 中采样，再变换到目标分布 $p_{\text{data}}$
  - 如何刻画两个分布之间的变换：用 marginal probability path
- What remains unknown:
  - 作为一个 neural net, 怎么建模？
  - 如何训练？即：如何定义优化目标 (loss function)？

我们还是慢慢来。假设现在的任务仍然是 point2point. 给定初始点 $x_0$ 和变换规则 ODE, 如何训练一个 NN 来预测 $x_1$? 这其实就是一个经典的 regression task. 根据刚才的 ODE 小节，有两个显而易见的思路：

1. 把 NN 建模为 vector field $u$, 然后使用 Euler method 对时间步 $t$ 反复迭代求出 $X_1$
2. 把 NN 建模为 flow function $\psi$, 然后直接代入 $X_0 = x_0$ 和 $t=1$ 即可求出 $X_1$

这里我们尝试思路 1.

> TODO: 其实我也没想清楚为什么不采用思路 2. 一个可能的解释是，虽然 flow matching model 可以用 ODE 建模、可以定义 flow function, 但是 diffusion model 是用 SDE 来建模的、无法定义 flow function, 也就只能采用思路 1 了。这种解释有些牵强，毕竟现在 SOTA 的模型大多开始用 flow matching 来替换 diffusion 了，明明可以用思路 2. 此处有待进一步思考。

对于思路 1, 意思就是我们训练的神经网络 $u^\theta$ 要尽可能逼近真实的 vector field $u^{\text{target}}$. 这谁都会。采取和刚才类似的心路历程，接下来还有两步：

1. 假设 $X_0$ 不再是一个 deterministic value, 而是概率分布 $p_{\text{init}}$ 中的一个样本，那该如何定义这种带有随机性的 "stochastic conditional vector field"?
2. 假设终点 $X_1$ 也不再是一个 deterministic value, 而是 $p_{\text{data}}$ 中的一个样本，又如何定义这种 "marginal vector field"?

第一步的定义是很自然的：

For every data point $z\in\R^d$, let $u_t^{\text{target}}(\cdot|z)$ denote a **conditional vector field**, defined so that the corresponding ODE yields the conditional probability path $p_t(\cdot|z)$. Formally speaking,

$$
X_0\sim p_{\text{init}}, \ \ \frac{\mathrm d}{\mathrm dt}X_t = u_t^{\text{target}}(X_t|z)\Rightarrow X_t\sim p_t(\cdot|z)
$$

这里同样有 "conditional" 字眼，因为 $z$ 是给定的点。

同样以 Gaussian distribution 为例。我们首先构造一个 conditional flow function $\psi_t^{\text{target}}(x|z)$ 如下：

$$
\psi_t^{\text{target}}(x|z) = \alpha_t z + \beta_t x
$$

根据定义，如果 $X_0\sim p_{\text{init}}=\mathcal N(0, I_d)$, 那么

$$
X_t = \psi_t^{\text{target}}(X_0|z) = \alpha_t z + \beta_t X_0 \sim \mathcal N(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot|z)
$$

所以，$\psi$ 是满足 conditional probability path 的 flow function. 接下来，为了得到 conditional vector field, 只需 $\psi$ 对时间求导：

$$
\begin{aligned}
&\frac{\mathrm d}{\mathrm dt}\psi_t^{\text{target}}(x|z) = u^{\text{target}}(\psi_t^{\text{target}}(x|z)|z), \forall x, z\in \R^d\\\
\Leftrightarrow{}& \dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\text{target}}(\alpha_t z + \beta_t x | z), \forall x, z\in \R^d\\\
\Leftrightarrow{}& \dot{\alpha}_t z + \dot{\beta}_t \left(\frac{x - \alpha_t z}{\beta_t} \right) = u_t^{\text{target}}(x|z), \forall x,z\in\R^d\\\
\Leftrightarrow{}& \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x = u_t^{\text{target}}(x|z),\forall x,z\in\R^d
\end{aligned}
$$

因此，对应的 conditional vector field 为

$$
u_t^{\text{target}}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

类比上述 probability path 的做法，下一步就是把 $u_t^{\text{target}}(\cdot|z)$ 中的 condition $z$ 去除掉 (by weighted average)，得到 marginal vector field.

于是，我们定义 **marginal vector field** $u_t^{\text{target}}(x)$ 如下：

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\mathrm dz
$$

只不过这里不再是以 $p_{\text{data}}(z)$ 作为 weights, 而是采取了 Bayesian 后验 $\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}$.

为什么这么定义？因为我们希望 marginal vector field $u_t^{\text{target}}(x)$ 能对应到 marginal probability path, 正如 conditional vector field 能对应到 conditional probability path. 只有对应上了，我们才能以 $u_t^{\text{target}}$ 作为 neural net 的建模对象。于是，接下来的定理就揭示了这一点：

*Theorem*: the marginal vector field follows the marginal probability path, i.e.

$$
X_0\sim p_{\text{init}}, \ \ \frac{\mathrm d}{\mathrm dt}X_t = u_t^{\text{target}}(X_t)\Rightarrow X_t\sim p_t,\forall t\in [0, 1]
$$

如果该定理成立，那么特别地取 $t=1$ 就有 $X_1\sim p_1=p_{\text{data}}$ 了。该定理的证明如下。

*Proof*.

首先给一个引理 [continuity equation](https://en.wikipedia.org/wiki/Continuity_equation) or [divergence theorem](https://en.wikipedia.org/wiki/Divergence_theorem)（微分形式）：

Given a vector field $u_t^{\text{target}}$ with $X_0\sim p_{\text{init}}$. Then $X_t\sim p_t$ for all $t\in [0, 1]$ if and only if

$$
\frac{\mathrm d}{\mathrm dt}p_t(x) = -\text{div}(p_t u_t^{\text{target}})(x) \ \ \text{for all }x\in \R^d, t\in [0, 1]
$$

where $\text{div}$ is the divergence operation defined as: $\text{div}(v)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v(x)$

根据该引理，只需证明我们定义的 marginal vector field $u_t^{\text{target}}$ 是满足 continuity equation 的：

$$
\begin{aligned}
\frac{\mathrm d}{\mathrm dt}p_t(x) &= \frac{\mathrm d}{\mathrm dt}\int p_t(x|z)p_{\text{data}}(z)\mathrm dz\\\
&= \int \frac{\mathrm d}{\mathrm dt}p_t(x|z)p_{\text{data}}(z)\mathrm dz\\\
&= \int -\text{div}\left(p_t(\cdot|z)u_t^{\text{target}}(\cdot|z) \right)(x)p_{\text{data}}(z)\mathrm dz\\\
&= -\text{div}\left(\int p_t(\cdot|z)u_t^{\text{target}}(\cdot|z)p_{\text{data}}(z)\mathrm dz \right)(x)\\\
&= -\text{div}\left(p_t(\cdot)\int u_t^{\text{target}}(\cdot|z) \frac{p_t(\cdot|z)p_{\text{data}}(z)}{p_t(\cdot)}\mathrm dz \right)(x)\\\
&= -\text{div}(p_tu_t^{\text{target}})(x)
\end{aligned}
$$

So we are done!

这里主要的 trick 是使用两次 continuity equation, 再结合 divergence operator 和 integral 可以交换次序。如果要追求更自然一些，其实应该是先有了这个证明，再反向推导出了 marginal vector field 的构造公式（定义）。

## Loss Function

回顾一下刚才的 "what remains unknown". 我们已经知道了如何建模，剩下唯一的问题就是如何定义 loss function.

假设我们的 NN 参数化为 $\theta$, 输入为 $t$ 和 $x=X_t$, 一个自然的想法就是使用 L2 loss:

$$
\mathbb E\left[ || u_t^\theta(x) - u_t^{\text{target}}(x) ||^2\right]
$$

其中 $t$ 和 $x$ 都需要 take expectation. 不妨考虑 $t\sim U(0, 1)$, $x\sim p_t$. 注意由于 $p_t$ 是 marginal probability path, 并不好直接计算，所以对 $x$ 的采样也必须折中地通过 conditional probability path 来 walk around, 也就是先采样 $z\sim p_{\text{data}}$, 再采样 $x\sim p_t(\cdot|z)=\mathcal(\alpha_t z,\beta_t^2 I_d)$. 所以，我们定义 **flow matching loss**

$$
\mathcal L_{FM}(\theta) = \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[|| u_t^\theta(x) - u_t^{\text{target}}(x)||^2 \right]
$$

然而，由于计算 marginal vector field 需要对 $z$ 积分：

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\mathrm dz
$$

所以上述 loss function 并不容易计算。但是，conditional vector field 是容易计算的：

$$
u_t^{\text{target}}(x|z) = \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

所以，我们“退而求其次”修改 loss function. 定义 **conditional flow matching loss** 为

$$
\mathcal L_{CFM}(\theta) = \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[|| u_t^\theta(x) - u_t^{\text{target}}(x|z)||^2 \right]
$$

不过，我们要思考的是：这样暴力地把 marginal 换为 conditional 是否合理呢？接下来的定理就要告诉我们：这样替换对训练而言是完全等价的。

**Theorem**:

$$
\mathcal L_{FM}(\theta) = \mathcal L_{CFM}(\theta) + C
$$

where $C$ is independent of $\theta$.

*Proof*.

$$
\begin{aligned}
\mathcal L_{FM}(\theta) &= \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[|| u_t^\theta(x) - u_t^{\text{target}}(x) ||^2 \right] \\\
&= \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[||u_t^\theta(x)||^2\right]\\\ &\ \ \ \ - 2 \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[u_t^\theta(x)^Tu_t^{\text{target}}(x) \right]\\\ &\ \ \ \ + \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[||u_t^{\text{target}}(x)||^2 \right]\\\
&= \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[||u_t^\theta(x)||^2\right]\\\ &\ \ \ \ - 2 \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[u_t^\theta(x)^Tu_t^{\text{target}}(x) \right]\\\ &\ \ \ \ + C_1\\\
\end{aligned}
$$

其中第二项为

$$
\begin{aligned}
&\mathbb E_{t\sim U(0, 1), x\sim p_t}\left[u_t^\theta(x)^Tu_t^{\text{target}}(x) \right]\\\
={}&\int_0^1 \int p_t(x)u_t^\theta(x)^Tu_t^{\text{target}}(x)\mathrm dx\mathrm dt\\\
={}&\int_0^1 \int p_t(x)u_t^\theta(x)^T\left[\int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\mathrm dz  \right]\mathrm dx\mathrm dt\\\
={}&\int_0^1 \int \int u_t^\theta(x)^Tu_t^{\text{target}}(x|z) p_t(x|z) p_{\text{data}}(z)\mathrm dz\mathrm dx\mathrm dt\\\
={}&\mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[u_t^\theta(x)^Tu_t^{\text{target}}(x|z) \right]
\end{aligned}
$$

因此

$$
\begin{aligned}
\mathcal L_{FM}(\theta) &= \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[||u_t^\theta(x)||^2\right]\\\ &\ \ \ \ - 2 \mathbb E_{t\sim U(0, 1), x\sim p_t}\left[u_t^\theta(x)^Tu_t^{\text{target}}(x) \right]\\\ &\ \ \ \ + C_1\\\
&= \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[||u_t^\theta(x) - u_t^{\text{target}}(x|z)||^2 \right]\\\ &\ \ \ \ - \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim p_t(\cdot|z)}\left[||u_t^{\text{target}}(x|z)||^2 \right]\\\ &\ \ \ \ + C_1\\\
&= \mathcal L_{CFM}(\theta) + C_2 + C_1
\end{aligned}
$$

So we are done. 这里主要的 trick 是用了两次 $||a-b||^2 = ||a||^2 - 2a^Tb + ||b||^2$.

如果以 Gaussian distribution 作为 $p_{\text{init}}$, 那么 loss function 即为：

$$
\mathcal L_{CFM}(\theta) = \mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, x\sim \mathcal N(\alpha_t z,\beta_t^2 I_d)}\left[ ||u_t^\theta(x) - \left(\dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t}\alpha_t \right) z - \frac{\dot{\beta}_t}{\beta_t}x ||^2 \right]\\\
$$

$$
=\mathbb E_{t\sim U(0, 1), z\sim p_{\text{data}}, \epsilon \sim \mathcal N(0, I_d)}\left[|| u_t^\theta(\alpha_t z +\beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t\epsilon)||^2 \right]
$$

对 sheduler 的一种特殊取法为 $\alpha_t = t$, $\beta_t = 1-t$. 此时的 probability path 被称为 **CondOT probability path**. 代入得到如下的 training algorithm:

1. Sample $z$ from training dataset
2. Sample a random time $t\sim U(0, 1)$
3. Sample noise $\epsilon\sim \mathcal N(0, I_d)$
4. Set $x = tz + (1-t)\epsilon$
5. Compute loss $\mathcal L(\theta) = ||u_t^\theta(x) - (z -\epsilon)||^2$
6. Update $\theta$ via gradient descent on $\mathcal L(\theta)$

这就是 Stable Diffusion 3 和 Movie Gen Video 的训练算法。

## SDE and Diffusion Models

