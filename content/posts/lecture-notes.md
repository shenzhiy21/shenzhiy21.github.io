+++
title = 'Lecture Notes'
date = 2025-01-14T13:00:00+08:00
draft = false
math = true
tags = ['note']
categories = ['note']
summary = "Lecture Notes"

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