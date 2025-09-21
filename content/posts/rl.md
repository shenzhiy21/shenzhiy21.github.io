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

Markov property: if for all $s'$, $r$, and histories $S_0, A_0, R_1, \cdots, S_{t-1}, A_{t-1}, R_t, S_t, A_t$, we have
$$
\begin{aligned}
&\text{Pr}\\{R_{t+1}=r,S_{t+1}=s' |S_t,A_t \\}\\\
={}&\text{Pr}\\{R_{t+1}=r,S_{t+1}=s'|S_0,A_0,R_1,\cdots,S_{t-1},A_{t-1},R_t,S_t,A_t \\}
\end{aligned}
$$

In this book, assume every environment to have Markov property. Many real-world scenarios can be viewed as an approximate Markov.
Also, this requires the "state" to be *informative* enough to encode all the information needed for state transition.

A reinforcement learning task that satisfies the Markov property is called a *Markov decision process*, or *MDP*. If the state and action spaces are finite, it's called a *finite MDP*.

A finite MDP is defined by the one-step dynamic of the environment:
$$
p(s',r|s,a)=\text{Pr}\\{S_{t+1}=s',R_{t+1}=r|S_t=s,A_t=a \\}
$$

Then we can compute the expected rewards for state-action pairs:
$$
r(s,a) =\textbf{E}[R_{t+1}|S_t=s,A_t=a]=\sum_{r\in\mathcal R}r\sum_{s'\in\mathcal S}p(s',r|s,a)
$$

The state-transition probabilities:
$$
p(s'|s,a)=\text{Pr}\\{S_{t+1}=s'|S_t=s,A_t=a \\}=\sum_{r\in\mathcal R}p(s',r|s,a)
$$

The expected rewards for state-action-next_state triples:
$$
r(s,a,s')=\mathbb E[R_{t+1}|S_t=s,A_t=a,S_{t+1}=s']=\frac{\sum_{r\in\mathcal R}rp(s',r|s,a)}{p(s'|s,a)}
$$