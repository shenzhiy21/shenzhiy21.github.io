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
r(s,a) =\mathbb{E}[R_{t+1}|S_t=s,A_t=a]=\sum_{r\in\mathcal R}r\sum_{s'\in\mathcal S}p(s',r|s,a)
$$

The state-transition probabilities:
$$
p(s'|s,a)=\text{Pr}\\{S_{t+1}=s'|S_t=s,A_t=a \\}=\sum_{r\in\mathcal R}p(s',r|s,a)
$$

The expected rewards for state-action-next_state triples:
$$
r(s,a,s')=\mathbb E\left[R_{t+1}\middle|S_t=s,A_t=a,S_{t+1}=s'\right]=\frac{\sum_{r\in\mathcal R}rp(s',r|s,a)}{p(s'|s,a)}
$$

*Value function*: functions of *states* or *state-action pairs* that estimates how good it is for that state (or state-action pair).
Formally, the *state-value function* of a state $s$ under a policy $\pi$, denoted $v_{\pi}(s)$, is the expected return when starting in $s$ and following $\pi$ thereafter.
$$
v_{\pi}(s) = \mathbb E_{\pi}\left[G_t\middle|S_t=s \right]=\mathbb E_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\middle|S_t=s \right]
$$

Similarly, the *action-value function* of taking action $a$ in state $s$ under policy $\pi$ is defined as:
$$
q_{\pi}(s, a) = \mathbb E_{\pi}\left[G_t\middle|S_t=s,A_t=a \right]=\mathbb E_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} \middle| S_t=s, A_t=a \right]
$$

These functions can be estimated by many methods, *e.g.*:
- Monte Carlo methods
- parameterized function approximators

*Bellman equation*:

$$
\begin{aligned}
v_{\pi}(s) &= \mathbb E_{\pi}\left[G_t \middle| S_t=s\right]\\\
&= \mathbb E_{\pi}\left[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}\middle|S_t=s \right]\\\
&= \mathbb E_{\pi}\left[R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^k R_{t+k+2}\middle|S_t=s \right]\\\
&= \sum_{a} \pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)\left[r+\gamma\mathbb E_{\pi}\left[\sum_{k=0}^\infty \gamma^k R_{t+k+2}\middle|S_{t+1}=s' \right] \right]\\\
&= \sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a)\left[r+\gamma v_{\pi}(s') \right]
\end{aligned}
$$

*Optimal policy*: if $v_{\pi_*}(s)\geq v_{\pi}(s)$ for all policy $\pi$.