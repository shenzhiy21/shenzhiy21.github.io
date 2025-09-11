+++
title = 'Lecture Notes (Pieces)'
date = 2025-09-11T13:00:00+08:00
draft = false
math = true
tags = ['note']
categories = ['note']
summary = "Lecture notes (pieces) collected from everywhere."

+++

This post is a collection of random notes & ideas from lectures, seminars, talks, and articles.
Some of them are just a few lines, some are more detailed. Hopefully they can be useful to someone (Yeah, including me).

## "The Bitter Lesson"

By Rich Sutton. [Link here](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

- **The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin.**
- The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.
- One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.
- The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries.

## Intuitions on Language Models

By Jason Wei @ OpenAI, at Stanford CS25: V4. [Link here](https://www.youtube.com/watch?v=3gb-ZkVRemQ&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM&index=30).

- **next word prediction == massive multi-task learning**, such as:
  - grammar, world knowledge, semantics, math, code
- **While overall loss improves smoothly, individual loss can improve suddenly ("emergent").**
  - GPT-3.5 & GPT-4 are similar in grammer-understanding, but GPT-4 is much better in math

BTW, in this lecture, Hyung Won Chung @ OpenAI also shared some really interesting opinions about transformers.
He "looked back" and compared the decoder-only structure nowaday with the encoder-decoder structure in the past.

The key point is like, we should carefully always "look back" and "rethink" about the assumptions we made along with the consequent design choices.
Some assumptions might be outdated because of "scaling law" or a more general task we're now facing. In this case, it's better to give up the assumptions and redesign the model.

## "Success in a Competitive Training Task"

By [Keller Jodan](https://kellerjordan.github.io/), inventor of the optimizer [Muon](https://github.com/KellerJordan/Muon), which is later used by [MoonShotAI](https://github.com/MoonshotAI/Moonlight).

I would like to quote the entire section from the author's [post](https://kellerjordan.github.io/posts/muon/).

---

The neural network optimization research literature is by now mostly filled with a graveyard of dead optimizers that claimed to beat AdamW, often by huge margins, but then were never adopted by the community. Hot take, I know.

With billions of dollars being spent on neural network training by an industry hungry for ways to reduce that cost, we can infer that the fault lies with the research community rather than the potential adopters. That is, something is going wrong with the research. Upon close inspection of individual papers, one finds that the most common culprit is bad baselines: Papers often don’t sufficiently tune their AdamW baseline before comparing it to a newly proposed optimizer.

I would like to note that the publication of new methods which claim huge improvements but fail to replicate / live up to the hype is not a victimless crime, because it wastes the time, money, and morale of a large number of individual researchers and small labs who run and are disappointed by failed attempts to replicate and build on such methods every day.

To remedy this situation, I propose that the following evidential standard be adopted: The research community should demand that, whenever possible, new methods for neural network training should demonstrate success in a competitive training task.

Competitive tasks solve the undertuned baseline problem in two ways. First, the baseline in a competitive task is the prior record, which, if the task is popular, is likely to already be well-tuned. Second, even in the unlikely event that the prior record was not well-tuned, self-correction can occur via a new record that reverts the training to standard methods. The reason this should be possible is because standard methods usually have fast hardware-optimized implementations available, whereas new methods typically introduce some extra wallclock overhead; hence simply dropping the newly proposed method will suffice to set a new record. As a result, the chance of a large but spurious improvement to a standard method being persistently represented in the record history for a popular competitive task is small.

To give an example, I will describe the current evidence for Muon. The main evidence for it being better than AdamW comes from its success in the competitive task “NanoGPT speedrunning.” In particular, switching from AdamW to Muon set a new NanoGPT training speed record on 10/15/24, where Muon improved the training speed by 35%. Muon has persisted as the optimizer of choice through all 12 of the new NanoGPT speedrunning records since then, which have been set by 7 different researchers.

Muon has a slower per-step wallclock time than AdamW, so if there existed hyperparameters that could make AdamW as sample-efficient as Muon, then it would be possible to set a new record by simply chucking Muon out of the window and putting good old AdamW back in. Therefore, to trust that Muon is better than AdamW, at least for training small language models, you actually don’t need to trust me (Keller Jordan) at all. Instead, you only need to trust that there exist researchers in the community who know how to tune AdamW and are interested in setting a new NanoGPT speedrunning record. Isn’t that beautiful?

## Towards High-Quality Writing for Research Publications

By Steve Lin @ Microsoft.

### Clear Writing Comes From Clear Thinking

- "Fortunately the act of composition, or creation, disciplines the mind; writing is one way to go about thinking ..." quote from book *The Elements of Style*.
- "Don't worry if you don't understand this book completely on the first reading. We didn't understand it all on the first writing!" quote from book *Design Patterns*.
- "If you can't explain it simply, you don't understand it well enough." by Albert Einstein.

A good research process: writing during the whole stage of developing (coding).

Writing helps to organize the ideas clearly.

### How to Write a **Bad** Research Paper

- Write the paper for only yourself (asssume the audience knows everything)
- Dismiss previous work (say nothing positive about previous methods)
- Just describe what you did (not explain why; not mention alternatives)
- Focus mainly on details (This is really important for me! You spent 3 months of the 6-month project on optimizing a certain part of the code, so you should spend half the paper talking about it)

### Tips on Writing Clearly

- Keep it simple !!!
  - Use plain (not complex) sentences to express
  - Be precise
  - Use examples
- Clear structure
  - One main idea per paragraph
  - Begin each section/subsection with a brief note on what will be presented and how it fits into the paper structure
- Good use of language
  - Stress position
  - Topic position. An example:
    - Collectin samples is the first step in water quality studies. (Next to talk about collecting samples)
    - The first step in water quality studies is collecting samples. (Next to talk about the second step)
  - Avoid grammar errors and style errors
    - American English or British English
    - [doesn't] $\rightarrow$ does not. Because [doesn't] is informal.
    - 4 $\rightarrow$ four. (in some cases, more formal)

### Guideliens for Different Paper Sections

- Abstract: cover the following parts
  - The problem your work addresses
  - Why the problem is interesting
  - What is new about your solution
  - Results and impact of solution
- Introduction
  - Introduce problem and explain why it is interesting and relevant
  - Briefly review common approaches to problem
  - Describe your novel solution at a high level
  - Highlight contributions, what is different
  - What results you get
- Discussion (perhaps the most interesting part for experts)
  - Limitations, how they might be addressed
  - Conditions when assumptions can be relaxed
  - Insight on contributions, how they can be useful in other problems

Other tips: get others to read; read after taking a long break.