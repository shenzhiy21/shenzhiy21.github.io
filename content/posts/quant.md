+++
title = 'LLM Quantization'
date = 2025-02-23T13:00:00+08:00
draft = false
math = true
tags = ['llm', 'code', 'system', 'quant']
categories = ['code']
summary = "8bit-Quantization Implementation for LLama-2-7b Model"

+++

之前实习的时候做了一些 LLM quantization 的尝试，然而当时连最简单的 8bit quantization 都没做到，深感挫折。最后虽然用了 [GPTQ](https://arxiv.org/abs/2210.17323) 在微软的 [AutoRT](https://github.com/microsoft/antares/) 框架下实现了 4bit quantization, 但是还是直接用的别人做好的 quant 后的权重。最近忽然意识到当时的代码有一个严重的 bug, 导致怎么调都调不出来。所以最近重新尝试，在这里记录一下一些实验现象以及小思考。

## Quantization Overview

首先，为什么要研究大模型的量化？



目前对于 LLM 的量化方案基本都是在 `matmul` 这一步进行的。在 Llama-2 系列模型中，一共有 6 处 `matmul` 运算，分别是（命名规范请参考 `transformers` library）：

- `self_attn.vqk_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.down_proj.weight`
- `mlp.up_proj.weight`
- `lm_head.weight`

其中前 5 处在每个 Attention block 中都会出现，最后一处则是在最后计算 logits 时才会出现。

考虑到系统硬件的设计，计算过程中的 weights & activations 大多数采用 `fp32` (single precision) 或 `fp16` (half precision) 来存储。对于以 `fp32` 格式存储 weights 的模型，推理阶段使用 `fp16` 就是最 naive 的一种 quantization, 并且效果基本不会差。原因是 `fp32` 转换到 `fp16` 的过程中，浮点数舍入误差并不大，而最后 decode 时只对 logits 取 top-1 (or top-k), 所以这个误差基本是可以忽略不计的。

然而，当每个参数的比特数变为 8bit 甚至更低 (4bit, 2bit, [1.58bit](https://arxiv.org/abs/2402.17764)) 时，误差就会大很多了。如何做到更低比特的 quantization 呢？

> 有人会问，既然有 8bit 和 16bit, 为什么不考虑它们中间的 10bit, 12bit, ... 呢？
>
> 大概因为目前的这些 GPU 架构，最小寻址单元就是 8bit 吧，例如 10bit 其实也会被 cast 成更高的 16bit 才能参与后续计算，这里的计算开销并不小。所以，工程上没有太大的意义。

LLM 社区对 quantize 的研究和探索非常广泛，这里简单列举一些工作：

- [GPTQ](https://arxiv.org/abs/2210.17323): 我很喜欢它的 Math formulation.
- [AWQ](https://arxiv.org/abs/2306.00978): Song Han 的工作，也是 MLSys 2024 Best Paper Award, 值得一读。这篇文章的两个 key point 是：(1) 对量化误差而言，参数并不是同等重要的 (2) 应当根据 activation value 而非 weight value 来找到那些重要的参数。
- [llama.cpp](https://github.com/ggml-org/llama.cpp): 它实现了非常多的 quantize 方法，并且是无外部依赖、pure C++ 的实现，对于学习 LLM 和本地部署 LLM 都很有用。

此外，评估量化结果的好坏，最先需要考虑的是如下两个 metric:

- MSE loss: 衡量 quantize 前后 weight matrix 的差异
- PPL: 评估 LLM 性能的很重要的 metric, 不多说了

当然，只测这俩肯定是不够的。还是得跑一些大的 benchmark 来看看和原始模型的差异有多大。

接下来，我们配合代码讲一讲我做的一些尝试。Config:

- Model: llama-2-7b-chat-hf
- Pure PyTorch + AutoRT implementation, without `transformers` or other high level libraries
- RTX 3090
- Max sequence length: 512

## Naive Quantization

我们试图做的是 8bit quantization. 由于 CUDA backend 本身不支持 `fp8` (其实现在也有啦), 所以折中的策略是：用 `fp16` 计算 activation value, 用 `int8` 存储 quantized weights, 并在 dequantize 阶段通过一些操作将其转换为 `fp16` 或 `fp32`, 再与 activation value 进行 `matmul` 计算。

我们首先对 weights 进行观察。有两个有趣的现象：

- 它们的绝对值基本都很小
- 它们近似呈现正态分布（此外，不同层、不同作用的 weights 往往具有不同的 sigma）

针对现象 1, 一个自然的想法是：在 [-1, 1] 区间内指定 256 个浮点数。Quant 阶段，把每个 weight 映射到离它最近的那个浮点数，并记录下 int8 格式的 index. Dequant 阶段，根据 index 在 lookup table 中查到对应的浮点数，并复原回 `fp32/fp16` 即可。

写成数学公式。我们假设 weights 是一维 tensor, 长度是 N. (实际上应该是 2d 矩阵，但是可以 flatten 到 1d)

`Quant`:

* Input: weights `w[N] (fp32)`, lookup table `a[0..255] (fp32)`
* for each `w[i]`:
  * if `w[i] >= 1`: let `id = 255`;
  * else if `w[i] < -1`: let `id = 0`;
  * else: find `id in [0, 255]` s.t. `a[id] <= w[i] < a[id+1]`.
* Output: `id[N] (int8)`

`Dequant`:

* Input: quantized weights `id[N] (int8)`, lookup table `a[0..255] (fp32)`
* Output: `_w[i] = a[id[i]] (fp32)`

其中由于 `id` 作为 int8 的 range 是 [-128, 127], 负数并不应该作为 tensor index. 所以实际上需要先转换成 uint8 (unsigned char) 来移动到正确的 [0, 255] 区间。算法过程中为了简洁没有写这步符号转换。下同。

对于 `a[]` 的选取，最简单的策略是均匀划分 [-1, 1] 区间。另一种策略是，由于 weights 呈现正态分布，所以按照正态分布的分位数点进行不均匀的划分 (可以参考 llama.cpp 中的 `iq` 量化策略)。我当时测了一下后者，发现虽然做不到 8bit quant, 但是 12bit (也就是 `a.len() == 4096`) 还是绰绰有余的。

## Blockwise Quantization

反思一下上述策略做不到 8bit 的原因。我们对所有参数都采用了同样的 cast range, 然而参数间的差异仍然是非常大的。例如可能某一段参数都集中在 `[-1e-5, 1e-5]` 区间内，这时采用 256 个预先定好的数对其进行舍入，它们可能都会被 quantize 到同一个值上——因为 $\frac{1}{256}$ 大约是 `3e-3`. 比这些参数大了两个数量级。

所以，我们可以对参数进行分块，每块采取不同的 cast range. 例如，每 32 个数作为一个 block. 在每个 block 内部计算 `weights.abs().max()`, 然后把所有 weights normalize 到 [-1, 1] 区间，再进行 quantize.

`Quant`:

* Input: weights `w[N] (fp32)`, lookup table `a[0..255] (fp32)`
* slice `w[N]` into `K` blocks `wk[]`, each of size `32`
* for each `wk[]`:
  * let `scale = wk.abs().max()`
  * `wk = wk / scale`
  * for each `wk[i]`:
    * cast `wk[i]` into `id` s.t. `a[id] <= wk[i] < a[id+1]`
* Output: `id[N] (int8)` and `scales[N/32] (fp32)`

`Dequant`:

* Input: quantized weights `id[N] (int8)`, lookup table `a[0..255] (fp32)` and `scales[N/32] (fp32)`
* Output: `_w[i] = a[id[i]] * scales[i/32]`

这种方法，其实对应了 llama.cpp 中的 `q8_0`. 此时压缩率不再是正正好好的 2.0 (假设原本是 `fp16`, 现在是 `int8`). 这是因为我们需要额外存储辅助数组 `scale`. 然而，每 32 个 weights 才对应一个 scale, 其实也是比较小的规模。

但是，如果直接这么做，模型的输出仍然不太好。什么原因呢？

回顾当初，我们使用 [-1, 1] 作为 cast range 是因为 weights 满足近似正态分布的假设，也就是正值和负值出现的概率相似，或者说均值接近 0. 然而，当我们分成大小为 32 的块之后，这个假设就显得不太恰当了。我们应当使用不同的方法进行 normalization.

不过话又说回来了，32 一组的采样是比较小的，如果样本数达到 1024, 是不是又可以保证近似正态假设了呢？

## Biased Blockwise Quantization

作为上述方法的修正，我们不但记录 `scale`, 还记录区间最小值 `min_val`. 也就是说：

- `min_val = wk.min()`
- `scale = wk.max() - min_val`
- `wk <- (wk - min_val) / scale`

这样使得变换后的 `wk[]` 位于 [0, 1] 之间，再采用上述量化策略即可。这对应 llama.cpp 中的 `q8_1`.

## Another Hardware Efficient Approach

上述方法其实是按照待量化矩阵 `w` 的内存连续性进行了分块。一般也不会把 block_size 取成 32 这么小，1024 甚至 4096 基本也是能 work 的。如果为了写 kernel 时比较方便，也可以取成 `block_size = w.size(1)`, 也就是一行分为一组，这样或许更直观。

当然，其实我们还有另一种分块策略。由于 `w` 是 2d tensor, 所以不妨划分成若干个 `KxK` 的子矩阵块。这样其实和 `matmul` 的分块加速是比较兼容的，或许更适合硬件执行（我乱说的）。

## Experiments

我们把 `K` 改记为步长 stride, 利用 AutoRT 简单实现一下 dequantize kernel:

```python
device = autort.device()

value_map = torch.tensor([3.90625e-3 * x for x in range(256)], dtype=torch.float32)
value_map_gpu = value_map.to(autort.device())

sm_table = {}
stride = 32

table_lookup = {
  "o_proj": "self_attn.o_proj.weight",
  "gate_proj": "mlp.gate_proj.weight",
  "down_proj": "mlp.down_proj.weight",
  "up_proj": "mlp.up_proj.weight",
  "vqk_proj": "self_attn.vqk_proj.weight",
  "weight_classify": "lm_head.weight"
}

def weight_preprocess(w: torch.Tensor, name: str):
  original_shape = w.shape
  w = w.unfold(0, stride, stride).unfold(1, stride, stride)
  n0, n1 = w.size(0), w.size(1)
  w = w.reshape(n0, n1, -1) # [n0, n1, stride * stride]
  w_min = w.min(dim=-1, keepdim=True)[0] # [n0, n1, 1]
  w_max = w.max(dim=-1, keepdim=True)[0] # [n0, n1, 1]
  scale = w_max - w_min
  w = (w - w_min) / scale # [n0, n1, stride * stride]
  w_low = torch.bucketize(w, value_map[:-1]).to(torch.int8).view(n0, n1, stride, stride).permute(0, 2, 1, 3).reshape(*original_shape)
  sm_table[f"{name}.scale"] = scale.view(n0, n1).to(torch.float32)
  sm_table[f"{name}.min"] = w_min.view(n0, n1).to(torch.float32)
  
  return w_low

my_custom_fn = autort.export(ir=f"""
  w[M, K] = value_map_gpu[input1[M, K].unsigned_cast()]
  my_result[M] +=! input0[K] * (w[M, K] * scale[M/{stride}, K/{stride}] + min_val[M/{stride}, K/{stride}])
""", inputs=["input0=float32[K]", "input1=int8[M, K]", "min_val=float32[A, B]", "scale=float32[A, B]", "value_map_gpu=float32[L]"])

def matmul_dequat(x, w, name, layer, memory_out=None):
  # parse name
  name = table_lookup[name]
  name = f"model.layers.{layer}.{name}" if layer >= 0 else name
  scale = sm_table[f"{name}.scale"].to(device)
  min_val = sm_table[f"{name}.min"].to(device)
  
  x = x.view(-1)
  memory_out = memory_out if memory_out is not None else torch.empty([w.size(0)], dtype=x.dtype, device=x.device)
  
  return my_custom_fn(x, w, min_val, scale, value_map_gpu, memory_out.view(-1), out=5)

def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, dim)

  for l in range(n_layers):
    xb = rmsnorm(x, rms_att_w[l])
    local_cache = val_cache.select(0, l).narrow(0, pos, 3)
    matmul_dequat(xb, weight_vqk[l], "vqk_proj", l, memory_out=local_cache.view(-1, 3 * xb.size(-1)))
    sq, sk = local_cache[1], local_cache[2]

    sq_out = torch.empty_like(sq).view(n_heads, head_size)
    sk_out = key_cache.select(0, l).narrow(0, pos, 1).view(n_heads, head_size)
    autort.ops.rotary_f32(sq.view(n_heads, 2, -1), inv_freq, sq_out, extra=[pos,])
    autort.ops.rotary_f32(sk.view(n_heads, 2, -1), inv_freq, sk_out, extra=[pos,])
    sq, sk = sq_out, sk_out

    b_sq = sq.view(n_heads, head_size)
    b_sk = key_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)
    b_sv = val_cache.select(0, l).view(seq_len, n_heads, head_size).narrow(0, 0, pos + 1)

    xb = autort.ops.attention_f32(b_sq, b_sk, b_sv, att_f)

    xb = matmul_dequat(xb, weight_o[l], "o_proj", l)
    x = x + xb
    xb = rmsnorm(x, rms_ffn_w[l])

    xb = torch.nn.functional.silu(matmul_dequat(xb, weight_f1[l], "gate_proj", l)) * matmul_dequat(xb, weight_f3[l], "up_proj", l)
    xb = matmul_dequat(xb, weight_f2[l], "down_proj", l)
    x = x + xb

  x = rmsnorm(x, rms_end_w)
  logits = matmul_dequat(x, weight_classify, "weight_classify", -1)
  return logits.half()
```

类似也可以实现一个不带 `min_val` 的版本。

接下来，我们试着取不同的 stride (16, 32, 64), 看看模型的表现。

Prompt: `"The Atlantic Ocean is"`

下面 `q8_1` 指的是含有 `scale` 和 `min_val` 的有偏方法，`q8_0` 指的是只含有 `scale` 的无偏方法。

---

**stride = 16, `q8_1`**: 7.69 tokens/s

The Atlantic Ocean is the second largest ocean in the world, covering approximately 20% of the Earth's surface. It is located between the Americas and Europe and Africa, and it separates the continents of North and South America from Europe and Africa. The Atlantic Ocean is a major component of the global ocean system and plays a crucial role in the Earth's climate and weather patterns.
The Atlantic Ocean is bounded by several major ocean currents, including the Gulf Stream, which brings warm water from the equator to the northern hemisphere, and the North Atlantic Current, which helps to moderate the climate of Western Europe. The ocean is also home to several major ocean basins, including the Caribbean Sea, the Gulf of Mexico, and the Mediterranean Sea.
The Atlantic Ocean is a vital source of food, with many countries relying on its fisheries for their economic well-being. It is also an important transportation route, with many ports and harbors providing access to the ocean for trade and commerce.
The Atlantic Ocean has a diverse range of marine life, including whales, dolphins, turtles, and many species of fish. It is also home to several important marine ecosystems, including coral reefs, estuaries, and mangrove forests.
However, the Atlantic Ocean is facing several environmental challenges, including pollution, overfishing, and the impacts of climate change. Rising ocean temperatures and acidification are affecting marine ecosystems and the animals that depend on them, and plastic pollution is accumulating in the ocean and harming marine life.
Overall, the Atlantic Ocean is a vital and complex system that plays a crucial role in the Earth's climate and weather patterns, as well as in the economic and cultural well-being of many countries around the world.

---

**stride = 32, `q8_1`**: 10.63 tokens/s

The Atlantic Ocean is the second largest ocean in the world, covering approximately 20% of the Earth's surface. It is located between the Americas and Europe and Africa, and it separates the continents of North and South America from Europe and Africa. The Atlantic Ocean is a major component of the Earth's climate system and plays a crucial role in regulating global weather patterns, ocean currents, and marine ecosystems.
The Atlantic Ocean is bounded by several major ocean currents, including the Gulf Stream, which originates in the Gulf of Mexico and flows northward along the eastern coast of the United States and Canada, and the North Atlantic Current, which flows southward along the western coast of Europe. These currents play a significant role in shaping the climate of the regions they affect, and they are also important for the transport of heat and nutrients across the ocean.
The Atlantic Ocean is home to a diverse range of marine life, including many species of fish, whales, dolphins, and other marine mammals. The ocean's waters are also home to a variety of coral reefs, kelp forests, and other marine ecosystems that provide important habitats for many species of fish and other marine organisms.
The Atlantic Ocean has a long history of human exploration and settlement, with many ancient civilizations establishing trade routes and colonies along its shores. Today, the Atlantic Ocean is an important source of food, transportation, and recreation for millions of people around the world.
Some of the key features of the Atlantic Ocean include:
* The Gulf Stream, a warm ocean current that flows northward along the eastern coast of the United States and Canada
* The North Atlantic Current, a cold ocean current that flows southward along the western coast of Europe
* The Mid-Atlantic Ridge, a mountain range that runs along the center of the Atlantic Ocean, where new ocean crust is being created as the tectonic plates move apart
* The Canary Current, a warm ocean current that flows eastward across the Atlantic Ocean from the Gulf of Mexico
* The Brazil Current, a warm ocean current that flows southward along the eastern coast of South America
* The South Atlantic Gyre, a large-scale circulation of ocean water that flows clockwise in the southern hemisphere
* The North Atlantic Gyre, a large

---

**stride = 64, `q8_1`**: 11.93 tokens/s

The Atlantic Ocean is the second largest ocean in the world, covering approximately 20% of the Earth's surface. The Atlantic Ocean is located between the Americas and Europe and Africa, and it connects with the Indian and Arctic Oceans to the east and the Pacific Ocean to the west. The Atlantic Ocean is a major component of the Earth's climate system and plays a significant role in the global ocean circulation.
The Atlantic Ocean is divided into several sections, including the:

* North Atlantic Ocean: This section extends from the Arctic Ocean to the equator and includes the waters off the coasts of North America, Europe, and Africa.
* South Atlantic Ocean: This section extends from the equator to the Antarctic Ocean and includes the waters off the coasts of South America, Africa, and Australia.
* Caribbean Sea: This is a smaller section of the Atlantic Ocean located between the Gulf of Mexico and the Virgin Islands.
* Gulf of Mexico: This is a smaller section of the Atlantic Ocean located between the Yucatan Peninsula and the Florida Panhandle.

The Atlantic Ocean is home to a diverse range of marine life, including:

* Whales: Several species of whales, including humpback, blue, and fin whales, can be found in the Atlantic Ocean.
* Dolphins: Bottlenose dolphins, orcas, and other species can be found in the Atlantic Ocean.
* Fish: The Atlantic Ocean is home to a wide variety of fish, including tuna, mackerel, and herring.
* Sharks: Several species of sharks, including great whites, tiger sharks, and hammerheads, can be found in the Atlantic Ocean.

The Atlantic Ocean has a significant impact on the climate and weather patterns of the surrounding land masses. The Gulf Stream, a warm ocean current, plays a major role in the climate of Western Europe, while the North Atlantic Ocean is known for its harsh winters and storms.

（接下来不停地输出换行符）

---

**stride = 16, `q8_0`**: 11.75 tokens/s

The Atlantic Ocean is the second largest ocean in the world, covering approximately 20% of the Earth's surface. The Atlantic Ocean is located between the Americas and Europe and Africa, and it connects with the Indian and Arctic Oceans to the east and the Pacific Ocean to the west. The Atlantic Ocean is a vital component of the Earth's climate system, and it plays a significant role in regulating the planet's weather patterns, ocean currents, and marine ecosystems.
The Atlantic Ocean is divided into several sections, including the:
1. North Atlantic Ocean: This section extends from the Arctic Ocean to the equator and includes the waters off the coasts of Europe, Africa, and North America.
2. South Atlantic Ocean: This section extends from the equator to the Antarctic Ocean and includes the waters off the coasts of South America, Africa, and Australia.
3. Caribbean Sea: This is a smaller section of the Atlantic Ocean located between the Gulf of Mexico and the coast of South America.
4. Gulf of Mexico: This is a smaller section of the Atlantic Ocean located off the coast of North America.
The Atlantic Ocean is home to a diverse range of marine life, including whales, dolphins, sharks, and many species of fish. The ocean's currents and tides play a crucial role in shaping the coastlines of the surrounding landmasses, and they also help to distribute heat and nutrients throughout the ocean.
The Atlantic Ocean has a long history of human exploration and settlement, with many ancient civilizations establishing trade routes and colonies along its shores. Today, the Atlantic Ocean is an important source of food, transportation, and recreation for millions of people around the world.
The Atlantic Ocean is also a vital component of the Earth's climate system, and it plays a significant role in regulating the planet's weather patterns and climate. The ocean's currents help to distribute heat and nutrients throughout the ocean, and they also play a role in the formation of hurricanes and other severe weather events.
Overall, the Atlantic Ocean is a vast and complex body of water that plays a vital role in the Earth's climate system and ecosystem. Its diverse marine life, rich history, and ongoing importance to human society make it a fascinating and important feature

---

**stride = 32, `q8_0`**: 13.40 tokens/s

The Atlantic Ocean is the second-largest ocean in the world, covering an area of approximately 106,400,000 square kilometers (41,000,000 square miles). It is located between the Americas and Europe and Africa, and it connects with the Indian and Arctic Oceans to the east.
The Atlantic Ocean is bound by several major ocean currents, including the Gulf Stream, which originates in the Gulf of Mexico and flows northward along the eastern coast of the United States and Canada, and the North Atlantic Current, which flows southward along the western coast of Europe.
The Atlantic Ocean is home to a diverse range of marine life, including whales, dolphins, sharks, and many species of fish. The ocean's waters are also rich in nutrients, including nitrogen and phosphorus, which support the growth of phytoplankton and other marine plants.
The Atlantic Ocean has played a significant role in shaping the Earth's climate and weather patterns. The ocean's warm waters help to regulate the Earth's temperature, and the Gulf Stream's influence on the climate of Western Europe has been particularly significant.
The Atlantic Ocean has also been the site of many significant historical events, including the discovery of the New World, the colonization of the Americas, and the transatlantic slave trade. Today, the ocean continues to be an important route for international trade and transportation, and it is a popular destination for tourists and recreational boaters.
Overall, the Atlantic Ocean is a vast and complex body of water that plays a critical role in the Earth's climate and weather patterns, as well as in human history and culture.

---

**stride = 64, `q8_0`**: 生成乱码，速度无意义

The Atlantic Ocean is the second-larg Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py Py

----

**Baseline: Quantize-Free Inference**: 47.60 tokens/s

The Atlantic Ocean is the second largest ocean in the world, covering approximately 20% of the Earth's surface. It is located between the Americas and Europe and Africa, and it separates the continents of North and South America from Africa and Europe. The Atlantic Ocean is a vital component of the Earth's climate system and plays a crucial role in regulating global weather patterns, ocean currents, and marine ecosystems.
The Atlantic Ocean is home to a diverse range of marine life, including whales, dolphins, turtles, and a vast array of fish species. The ocean's waters are also rich in nutrients, including nitrogen, phosphorus, and iron, which support the growth of phytoplankton and other microorganisms. These microorganisms, in turn, support the entire marine food chain, from tiny zooplankton to large predators like sharks and tuna.
The Atlantic Ocean has a significant impact on the Earth's climate, particularly in regulating global temperatures and weather patterns. The ocean acts as a heat sink, absorbing and storing heat from the sun, which helps to moderate temperatures around the world. The ocean also plays a key role in the Earth's water cycle, evaporating water vapor into the atmosphere, which eventually falls as rain or snow.
In addition to its ecological importance, the Atlantic Ocean has significant economic and cultural value. The ocean provides a source of food, including fish and other seafood, which are an important part of the diets of millions of people around the world. The ocean also supports a variety of industries, including shipping, tourism, and offshore energy production.
However, the Atlantic Ocean is facing a range of threats, including pollution, overfishing, and the impacts of climate change. Plastic pollution, in particular, is a major problem in the Atlantic, with millions of tons of plastic waste entering the ocean each year, harming marine life and contaminating the food chain. Overfishing is also a significant issue, with many fish populations being depleted due to unsustainable fishing practices.
Climate change is also having a significant impact on the Atlantic Ocean, with rising temperatures and acidification of the water affecting marine ecosystems and the organisms that live in them. Rising sea

---

给生成速度打个表（单位 token/s）

| | stride=16 | stride=32 | stride=64 |
| ---- | ---- | ---- | ---- |
| `q8_0` | 11.75 | 13.40 | - |
| `q8_1` | 7.69 | 10.63 | 11.93 |

明显可见 `q8_0` 比 `q8_1` 推理速度更快，但是 `q8_0` 在 `stride=64` 时就寄了，反观 `q8_1` 还能打。

## Conclusion

这篇文章主要在靠最后的实验输出凑字数（笑）。实验结论也并不太出乎意料，有趣的地方大概在于自己实现这些个算法的成就感吧！GPTQ int4 量化的版本在这里 [link](https://github.com/microsoft/antares/blob/latest/samples/05_llama2_7b_int4.py). 最后强推一下 [AutoRT](https://github.com/microsoft/antares) ！如果有手搓 customized kernel 的需求，又觉得 Triton/CUDA 太麻烦，不如试试它 :)