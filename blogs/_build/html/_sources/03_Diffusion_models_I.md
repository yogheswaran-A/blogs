# Blog 02: Flow Matching: The Theory Behind Stable Diffusion - 3.

> **We can know only that we know nothing. And that is the highest degree of human wisdom.**  
> — *Leo Tolstoy, War and Peace*.

- general case, po = some static distribution, pf = p_data, langevian dynamics.
- conditional gaussion trnsitions, affince drift co-efficiet
- OU process for final VP


## What is the post about?

Diffusion models and flow matching have improved image generation(they both can be wriiten under the same formulation). In this blog post I will write my learnings about diffusion models from the ground up. The topis covered are:

1. Diffusion theory with self contained proofs.
2. Guidance.
2. Training a model on MINST dataset.

I think the MIT's course [Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/) (thanks a lot!!) is much better than what I have written, I learned from it, please take a look at it, they have notes, codes and video lectures. This post is inspired from it.

## Pre-requiste

Knowledge of stachastic Process and calculas are needed. My previous blog post on [SDE](https://yogheswaran-a.github.io/blogs/01_weiner_process.html) covers the neccesary topics, and I believe it is self contained.

## Introduction

Given a collection of images, we want to generate a new image that looks similar to those in the dataset. How might we solve this problem using our knowledge of Stochastic Differential Equations (SDEs)?

The collection of images represents a data distribution, denoted by $P_{\text{data}}$. Our goal is to generate samples from this same distribution.

Since neural networks (NNs) are excellent function approximators, we can train a NN to generate images from the distribution $P_{\text{data}}$.

The input to the network should be simple to sample from. A standard approach is to use a vector $z$ sampled from a standard Gaussian distribution, $\mathcal{N}(0,I)$.

We now need a principled way to transform this Gaussian noise into a sample from $P_{\text{data}}$. This sounds like the reverse of a process that corrupts data into noise. This is precisely the core idea behind score-based generative models.

We can construct a forward process that gradually adds noise to an image from $P_{\text{data}}$ over a time interval $[0,T]$, until it becomes indistinguishable from pure Gaussian noise. This is our forward diffusion process. If we can learn to reverse this process, we can start with noise at time $T$ and generate a clean image at time $0$.

## Constructing the Forward Process

Mathematically, we can model this forward diffusion using an SDE. A common and tractable choice is a process with linear coefficients:

$$dx_t = f(x_t,t)dt + g(t)dw_t$$

where:

* $x_t$ is the noisy image at time $t$.
* $f(x_t,t)$ is the drift function, which governs the deterministic evolution of the process. For simplicity and analytical tractability, we choose a linear form: $f(x_t,t) = a(t)x_t + b(t)$.
* $g(t)$ is the diffusion coefficient, controlling the magnitude of the random noise added at time $t$.
* $w_t$ is a standard Wiener process (Brownian motion).

This gives us the following linear SDE:
$$dx_t = (a(t)x_t + b(t))dt + g(t)dw_t$$

### Solution of the SDE

This linear SDE has an analytical solution. Given an initial data point $x_0 \sim P_{\text{data}}$, the state $x_t$ at any time $t$ can be expressed as a conditional Gaussian distribution.
The solution is found using an integrating factor, similar to solving linear ordinary differential equations. The resulting distribution of $x_t$ conditioned on $x_0$ is:

$$p(x_t|x_0) = \mathcal{N}(x_t; \mu_t, \sigma_t^2 I)$$

## The Goal: A Simple Known Prior

The key is to design the functions $a(t)$, $b(t)$, and $g(t)$ such that, at the end of the process at time $T$, the distribution of $x_T$ is independent of the starting point $x_0$ and converges to our desired simple prior, $\mathcal{N}(0,I)$.

A widely used example is the Variance Preserving (VP) SDE, which is closely related to the Ornstein-Uhlenbeck process. In this case, we can set $b(t)=0$ and choose $a(t)$ and $g(t)$ carefully. For the VP SDE, the conditional distribution simplifies to:

$$p(x_t|x_0) = \mathcal{N}(x_t; \alpha(t)x_0, [1-\alpha(t)^2]I)$$

Here, $\alpha(t)$ is a schedule function that decreases from $\alpha(0)=1$ to $\alpha(T) \approx 0$. As $t \to T$:

* The mean, $\alpha(T)x_0$, approaches $0$, effectively erasing the initial information from $x_0$.
* The variance, $1-\alpha(T)^2$, approaches $1$.

Therefore, the distribution $p(x_T) = \int p(x_T|x_0)p_{\text{data}}(x_0)dx_0$ will approach $\mathcal{N}(0,I)$. We have successfully transformed our complex data distribution into simple Gaussian noise. The next step, which involves training a neural network, is to learn the reverse SDE that takes us from $x_T$ back to $x_0$.


### Reverse Process

The [reverse-time SDE](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#reverse-time-equation) is given by:

$$
\boxed{
dX(t) = \left[\mu(X,t) - \sigma(X,t)\sigma(X,t)^\top \nabla_X \log p(X,t)\right]\,dt + \sigma(X,t)\,d\hat{W}_t
}
$$
