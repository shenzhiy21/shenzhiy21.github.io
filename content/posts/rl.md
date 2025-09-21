+++
title = 'Reinforcement Learning Notes'
date = 2025-09-20T18:21:52+08:00
draft = false
math = true
tags = ['note']
categories = ['note']
summary = "Notes on everything related with RL."
+++

## Sutton Book Notes

### Chapter 3: Finite Markov Decision Processes

Problem Formulation: "an *agent* learns from interaction (with *environment*) to achieve a goal".

![](/images/rl/pipeline.png)

At each time step $t$:
- Agent gets a state $S_t$ from environment.
- Agent selects an action $A_t$ based on $S_t$.
- Agent receives a reward $R_{t+1}$ from environment, and finds itself in a new state $S_{t+1}$.

At each time step $t$, the agent implements a mapping from states to probabilities of selecting each possible action, called *policy*, denoted $\pi_t$. $\pi_t(a|s)$ is the probability that $A_t = a$ if $S_t = s$.

The agent's goal is to maximize the *cumulative reward* it receives over the long run, often called *return*:
$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
where $0\leq \gamma\leq 1$ is called the *discount rate*.

