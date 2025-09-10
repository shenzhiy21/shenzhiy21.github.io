+++
title = 'CUDA Notes'
date = 2025-09-11T00:41:54+08:00
draft = false
math = true
tags = ['note']
categories = ['note', 'system', 'parallel', 'memory']
summary = "Just tracing the process of learning CUDA programming"
+++

## Resources

- [GPU-Mode Lectures](https://github.com/gpu-mode/lectures)
- Programming Massively Parallel Processors

## Programming Massively Parallel Processors (PMPP) TakeAways

### Chapter 1. Introduction

Design philosophies for CPU/GPU design:

- CPU: latency oriented, optimized for sequential code performance
- GPU: throughput oriented (floating-point calculations and memory access throughput, "parallel")

Two types of bounds

- Memory bound: applications where execution speed is limited by memory access latency and/or throughput
- Compute bound: ... limited by the number of instructions performed per byte of data

### Chapter 2. Heterogeneous data parallel computing