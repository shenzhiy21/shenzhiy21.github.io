+++
title = 'Research Idea'
date = 2025-09-16T18:21:52+08:00
draft = false
math = true
tags = ['note']
categories = ['note', 'visualization', 'math']
+++

This post records research ideas of my interest.

> 靡不有初，鲜克有终。

## composite.js

The rendering tool used in our composite visualization project
can be converted into a useful javascript library (although we have
to refactor the code).
The purpose of this tool is to make it easier for users to design
and create composite visualizations.

Reference:
- PiCCL: Data-Driven Composition of Bespoke Pictorial Charts
- Manipulable Semantic Components: a Computational Representation of Data
  Visualization Scenes

A basic example:

```javascript
import * from composite;

const chart1 = createChart('bar', data1, config1);
const chart2 = createChart('line', data2, config2);
const composite = compose(chart1, chart2, 'stack', 'vertical');
composite.toSVG('result.svg');
```

Users can either use the built-in chart rendering function in our tool,
or customize their own rendering functions for new chart types or variants.
This makes our method extensible.

## Diffusing Game of Life

Inspired by [Physics of Language Models](https://physics.allen-zhu.com/home),
I want to investigate the physics of diffusion models.
The general method of PhysicLM is to construct controllable environments 
to train language models and make observations, instead of directing playing
with pretrained large language models.

For example, to examine the ability of language models to learn grammar,
Allen Zhu designed a group of CFGs, and trained a language model from scratch.
After that, by observing the hidden layers and activations of the model,
he concluded with some interesting takeaways.

A similar approach is to transfer this research style into 
the understanding of diffusion models. Then, what environments can I design
to construct controlled experiments? The only thing we need to do is
to generate enough training data ("enough" is not enough, 
we actually want "infinite"!) to train a diffusion model to learn the rules
(*i.e.*, the physic of this designed world).

Therefore, the first thing that comes to my mind is Conway's Game of Life.
It has specific rules for state transfer, and has infinite states. Besides,
a key advantage of this environment is that,
a minor change in the input state may lead to a 
significant change in output states.
We can test if a diffusion model can actually learn the rules
of this game. My belief is that, a diffusion (or flow matching) model
actually learns the "transfer" from one distribution to another distribution.
This makes it possible for these models to learn the specific rules.

## Infographic Generation

## Understanding Flow Matching Models

## Thinking Compression