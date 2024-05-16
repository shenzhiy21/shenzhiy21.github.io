+++
title = 'FFT On the Road'
date = 2024-04-20T10:46:52+08:00
draft = false
math = true
tags = ['code', 'fft', 'parallel', 'system']
categories = ['code']
summary = "Guide to implement FFT in C++ with parallelism"
+++

## Introduction

Discrete Fourier Transform (DFT) 是一种重要的信号处理方法。目前流行的许多 ML 框架（例如语音、图像模型）都会用到卷积 (convolution) 的计算，而通过 Fourier Transform 实现卷积 (FFT-conv) 的理论复杂度较低，特别是当卷积核 (kernel) 较大时，相比其它方法（例如 Winograd convolution）具有明显的速度优势。因此，如何高效地实现 DFT, 即 Fast Fourier Transform (FFT), 就显得尤为重要。`PyTorch`, `numpy` 等工具中都具有相当全面且高效的 FFT 实现，也包括其它变种（例如 FFT 的逆变换 `ifft`, 高维版本 `fftn`, 作用于实数（而非一般复数）的 `rfft`, 作用于实数的高维版本 `rfftn`, 作用于实数的高维版本的逆变换 `irfftn` ... ...）。当然，我们不能只满足于当一个“调包侠”，所以本文旨在详细剖析 FFT 的数学原理，并探讨如何使用 C++ 高效实现 FFT, 例如并行化处理。

## Background

### Linear Algebra

首先，到底什么是 FFT? 参考 Wikipedia 上的定义：

> Let $x_0,\cdots,x_{n-1}$ be complex numbers. The DFT is defined by the formula:
> $$
> X_k=\sum_{m=0}^{n-1}x_m e^{-\mathrm i2\pi km/n},k=0,\cdots,n-1
> $$

这里 $e^{\mathrm i2\pi/n}$ 是 $n$ 次单位根 (primitive $n$-th root of unity).

我们可以给出一个 trivial 的 C++ 实现，时间复杂度为 $O(n^2)$.

```cpp
std::vector<std::complex<float>> fft_naive(const std::vector<std::complex<float>>& x) {
    size_t len = x.size();
    std::vector<std::complex<float>> y(len);
    for (size_t i = 0; i < len; ++i) {
        std::complex<float> sum(0.0, 0.0);
        for (size_t j = 0; j < len; ++j) {
            float angle = 2 * M_PI * i * j / len;
            std::complex<float> w(cos(angle), sin(-angle));
            sum += x[j] * w;
        }
        y[i] = sum;
    }
    return y;
}
```

从线性代数的角度来看，$\\{X_k\\}$ 是 Fourier matrix 作用于 $\\{x_k\\}$ 得到的。Fourier matrix 形式很优美，举个 4 阶的例子：
$$
F_4=\left(\begin{array}{ccc}
1 & 1 & 1 & 1 \\\
1 & \omega^{1} & \omega^{2} & \omega^3 \\\
1 & \omega^2 & \omega^4 & \omega^6 \\\
1 & \omega^3 & \omega^6 & \omega^9
\end{array}\right)
$$
这里的 $\omega=e^{-\mathrm i2\pi/n}$. 显然，$F_n$ 是一个对称矩阵 $F_n^T=F_n$. 更神奇的是，$F_n$ 是某个酉矩阵 (unitary matrix) 的倍数：
$$
\frac{1}{n}\overline{F_n}=F_n^{-1}
$$
这里扯到线性代数其实在逻辑上没那么 straightforward, 但是它能让我们从另一个角度看到 FFT 的原理。有如下的定理（参考了杨一龙老师的线性代数教材）：

> We have the following decomposition, where $D_n=\text{diag}(1,\omega,\cdots,\omega^{n-1})$, and $P$ is a matrix permuting all odd columns to the left and all even columns to the right.
> $$
> F_{2n}=\begin{pmatrix}I_n&D_n\\\I_n&-D_n\end{pmatrix}\begin{pmatrix}F_n&0\\\0&F_n\end{pmatrix}P
> $$

也就是说，假设求出了 $F_n$, 我们就能快速地计算出 $F_{2n}$. 类似地，也可以推广到 $F_{3n}$.

如果考虑求 $F_n$ 的时间复杂度，那么上述线性变换的递推式为
$$
T_{2n}=2T_n+\Theta(n)
$$
根据主定理 (Master Theorem) 有
$$
T_n=\Theta(n\log n)
$$
这就是 FFT in pure mathematics.

### Polygon Interpolation

线性代数的解释难免让人感到难以接近，所以自然想问：有没有更直观的算法原理？实际上，考虑一个多项式 $f(x)=\displaystyle\sum_{k=0}^n a_k x^k$, Fourier 变换做的事情就是求 $f$ 作用在 $n$ 次单位根上的值。举个例子，假设
$$
\begin{aligned}
f(x)&=a_0+a_1x+\cdots+a_6x^6+a_7x^7\\\
&=(a_0+a_2x^2+a_4x^4+a_6x^6)+x(a_1+a_3x^2+a_5x^4+a_7x^6)\\\
&=G(x^2)+x\cdot H(x^2)
\end{aligned}
$$
其中 $G$ 和 $H$ 是分别取奇数次幂和偶数次幂的系数得到的多项式。因此有
$$
\begin{aligned}
f(\omega_n^k)&=G((\omega_n^k)^2)+\omega_n^k\cdot H((\omega_n^k)^2)\\\
&=G(\omega_n^{2k})+\omega_n^k\cdot H(\omega_n^{2k})\\\
&=G(\omega_{n/2}^k)+\omega_n^k\cdot H(\omega_{n/2}^k)
\end{aligned}
$$
类似地有
$$
f(\omega_{n}^{k+n/2})=G(\omega_{n/2}^k)-\omega_n^k\cdot H(\omega_{n/2}^k)
$$
因此，求出 $G(\omega_{n/2}^k)$ 和 $H(\omega_{n/2}^k)$ 后就能同时求出 $f(\omega_{n}^k)$ 和 $f(\omega_n^{k+n/2})$. 可以递归求解。复杂度是 $O(n\log n)$.

## Optimization

我们不喜欢递归的函数，所以希望转换成非递归的实现。此外，递归实现需要占用大量的额外内存，更好的解决方案是**就地**进行运算。

关于就地的问题，我们模拟上述数组的拆分过程：
$$
\begin{aligned}
&\\{a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7 \\}\\\
\Rightarrow &\\{a_0,a_2,a_4,a_6 \\},\\{a_1,a_3,a_5,a_7 \\}\\\
\Rightarrow &\\{a_0,a_4 \\},\\{a_2,a_6\\},\\{a_1,a_5\\},\\{a_3,a_7\\}\\\
\Rightarrow &\\{a_0\\},\\{a_4\\},\\{a_2\\},\\{a_6\\},\\{a_1\\},\\{a_5\\},\\{a_3\\},\\{a_7\\}
\end{aligned}
$$
不难发现这个变换的规律：把原先 index 的二进制表示进行翻转，即可得到最终的 index. 例如：$a_3$ 的二进制是 $011$, 翻转后为 $110$, 所以 index 为 $6$. 这个过程可以在 $O(n)$ 的时间内递归实现：

```cpp
void change(std::vector<std::complex<float>>& x) {
	size_t len = x.size();
    std::vector<int> rev(len);
    for (size_t i = 0; i < len; ++i) {
        rev[i] = rev[i >> 1] >> 1;
        if (i & 1) {
            rev[i] |= len >> 1;
        }
    }
    for (size_t i = 0; i < len; ++i) {
        if (i < rev[i]) {
            std::swap(x[i], x[rev[i]]);
        }
    }
}
```

在这一操作之后，按照公式
$$
\begin{aligned}
f(\omega_n^k)&=G(\omega_{n/2}^k)+\omega_n^k\cdot H(\omega_{n/2}^k)\\\
f(\omega_{n}^{k+n/2})&=G(\omega_{n/2}^k)-\omega_n^k\cdot H(\omega_{n/2}^k)
\end{aligned}
$$
假设 $G(\omega_{n/2}^k)$ 和 $H(\omega_{n/2}^k)$ 分别在数组下标为 $k$ 和 $k+n/2$ 的位置，则可以把 $f(\omega_n^k)$ 和 $f(\omega_n^{k+n/2})$ 覆写到这两个位置，从而实现 in-place 的目标。这一方法被称为 Radix-2 蝶形运算。下面是非递归的实现：

```cpp
void fft(std::vector<std::complex<float>>& x) {
    change(x);
    int n = x.size();
    for (int h = 2; h <= n; h <<= 1) {
        std::complex<float> wn = std::complex<float>(cos(2*M_PI/h), sin(-2*M_PI/h));
        for (int j = 0; j < n; j += h) {
            std::complex<float> w = std::complex<float>(1.0, 0.0);
            for (int k = j; k < j + h / 2; k++) {
                std::complex<float> u = x[k];
                std::complex<float> t = w * x[k + h / 2];
                x[k] = u + t;
                x[k + h / 2] = u - t;
                w = w * wn;
            }
        }
    }
}
```

事实上，如果这个方法用线性代数的方式表出，就是上面的矩阵表达式
$$
F_{2n}=\begin{pmatrix}I_n&D_n\\\I_n&-D_n\end{pmatrix}\begin{pmatrix}F_n&0\\\0&F_n\end{pmatrix}P
$$

## Parallelism & Optimization

以上是一个足够好的串行 FFT 了。然而，在实际场景中，并行化是不可避免的。这里涉及的并行化有：

1. 对一个高维的 tensor, 我们只需要在某个维度上做 FFT, 所以其它维度自然可以高效并行；
2. 上面的串行 FFT 实现当然可以并行地实现，但并不能**充分**并行。

我们重点关注第二个问题。剖析一下上面的代码（及其原理），它涉及了三重 `for` 循环。外层（自变量是 `h`）循环显然没法并行（Divide-and-Conquer 的计算方式是有先后依赖关系的）。中层（自变量是 `j`）划分了若干个组，而内层循环（自变量是 `k`）代表每个组内部的就地更新，所以各个组之间是可以并行的。

但是，随着 `h` 的变化，组的数量也会变化（数量为 `n // h`）。由于 `h` 以指数级增大，所以并行度会显著降低，当 $h=n/2$ 时只有两个组了，不能充分利用多个线程。

美中不足的是，各个组的内部是不能并行计算的，因为代码中涉及到对 `w`（也就是 $\omega_n^k$）的更新。这不难解决：只要事先为所有的 $\omega_n^k$ 打表，即可通过查表的方式得到 `w`, 从而内层循环也可以并行了。

不过，这还不够。随着 `h` 增大，中层循环的并行度越来越小，内层循环的并行度越来越大，但它们的乘积始终是 `n`. 所以，或许可以把它们合并成一个长度为 `n` 的单层循环？这并不难实现。

此外，我们需要对每个 $h$ 和每个 $k\in\\{0,\cdots,h/2-1\\}$ 打表，假设 `n` 的最大值是 $2048$, 那么需要一个
$$
1+2+4+8+\cdots+1024=2047
$$
长度的表。然而，注意到 $\omega_{2h}^{2k}=\omega_h^k$, 所以这个表可以压缩至长度为 $1024$, 换句话说，只要对 $h=1024$ 的情形打表即可。下面是完整代码：

```cpp
#define MAX_LEN 2048

std::vector<std::complex<float>> roots(MAX_LEN / 2);

for (int i = 0; i < MAX_LEN / 2; ++i) {
    roots[i] = std::complex<float>(
        cos(2 * M_PI * i / MAX_LEN),
        sin(-2 * M_PI * i / MAX_LEN)
    );
}

void fft_precompute(std::vector<std::complex<float>>& x) {
    // roots have already been precomputed.
    change(x);
    int n = x.size();
    for (int h = 2; h <= n; h <<= 1) {
        // parallelism here
        for (int i = 0; i < n / 2; i++) {
            int j = i / (h / 2) * h;
            int k = i % (h / 2) + j;
            int idx = (k - j) * (MAX_LEN / h);
            std::complex<float> u = x[k];
            std::complex<float> t = roots[idx] * x[k + h / 2];
            x[k] = u + t;
            x[k + h / 2] = u - t;
        }
    }
}
```

最后，在 `change` 函数中计算出的 `rev[]` 数组其实是 `[0..n]` 上的一个双射，这一步同样可以通过打表的方式预处理，以避免后续每一个调用 FFT 时都重新计算（因为我们可能会遍历其它维度，或遍历多个 tensor）。使用 cuda 重写上述代码，加入 64 个线程的并行化之后，效率已经基本和 `torch.fft.fft` 相差不多了。

至于 FFT 的逆变换 `ifft`, 只需稍稍改变 `root` 数组，并在最后归一化处理一下即可，本质上是一样的。

## Discussion

不过，我们也不知道 PyTorch 对 FFT 做了哪些其它的优化，例如针对输入为实数的 `torch.fft.rfft`, PyTorch 的实现比 general 的 `torch.fft.fft` 快很多（简单的 intuition: 当输入为实数时，FFT 作用后的数组的前一半和后一半其实是共轭的关系，所以可以只维护前一半）。此外，PyTorch 也能高效处理数组长度不是 2 的幂的 FFT, 但我们的 radix-2 算法是做不到这一点的，需要实现其它算法（可参考 [stack exchange: non-power-of-2-ffts](https://math.stackexchange.com/questions/77118/non-power-of-2-ffts)）。不过，如果我们实现 FFT 的目的是将其接入 conv, 那么长度不是 2 的幂次也没问题：只需 padding 到 2 的幂次，计算完之后切片 (slice) 即可。

然而这样的 padding 方法很可能在内存上带来一定的开销：考虑 2D conv 的情形，假设 image size $256\times 256$, kernel size $3\times 3$, Stable Diffusion 等架构中常见的一个操作是，将 image padding 到 $258\times 258$, 这样经过卷积的输出图像大小仍然是 $256\times 256$. 然而，由于我们的方法只支持 2 的幂次，所以需要把 $258\times 258$ 进一步 padding 到 $512\times 512$, 同时将 $3\times 3$ 的 kernel 同样 padding 到 $512\times 512$, 几乎占用了 $8$ 倍的内存。

目前的解决方案是：把 image 切割成若干个大小为 $2$ 的幂次的 chunk, 对每个 chunk 分别做 conv. 这样就可以规避内存的问题了。但是各个 chunk 之间不是独立的，所以在计算时需要处理 overlap, 这会增大计算时间（不过理论时间复杂度不变）。

值得注意的是，PyTorch 中对卷积的实现（例如 `torch.nn.functional.conv2d`）使用的不是 FFT-conv, 在 kernel size 较小时比我们的方法快，但 kernel size 较大（例如和 image size 规模相当）时就不如我们的方法了。然而，FFT-based method 的误差会大于直接计算卷积。

## Conclusion

本文讨论了 FFT 的原理、并行化实现，以及如何在工程上优化内存开销、接入 conv 算子。回过头来看，最终版本的 FFT 代码已经和 naive implementation 面貌全非了，这启示我们，从基本的数学原理、到优美的代码实现、再到完整的工程应用，有很长的一段距离啊 (a long way to go!).

## Reference

* [`fft-conv-pytorch`](https://github.com/fkodom/fft-conv-pytorch)
* [`torch.fft`](https://pytorch.org/docs/stable/fft.html)
* [OI wiki for FFT](https://oi-wiki.org/math/poly/fft/)
* [Wikipedia for FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
* Linear Algebra lecture notes, by Prof. Yilong Yang