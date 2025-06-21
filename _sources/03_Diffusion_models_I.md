# Viewing Diffusion, Score, Rectified flow, Heirrachical VAEs From The Same Lens

This blog contains the basics, derivations, intuition and idea behind diffusion models, score based models, Heirrachical VAEs and rectified flow. I have also explained how these formulations relate to one another and comes under the same umberala. I will start with each formulation and as we progress we will see how all of these fall under the same umberela. 

## Contents
- Diffuions Models

## Pre requistes   
For understanding diffusion: SDEs, Ito's lemma, OU process, Forward And Backward. One can read more about this in my previous [blog post](https://yogheswaran-a.github.io/blogs/01_weiner_process.html).       

## Diffusion Models

Here is the problem statement:  
Given a collection of images, we want to generate a new image that looks similar to those in the dataset. How might we solve this problem using our knowledge of **Stochastic Differential Equations (SDEs)**? Some thoughts:

1. The collection of images represents a distribution of images, denoted by $P_I$. Our goal is to generate a sample from this same distribution.

2. Since neural networks NNs are good at function approximation, we can maybe train a NN to generate an image from the distribution $P_I$?

3. What should the input to the network be? Ideally, it should be something simple and low-dimensional to make inference efficient. A common approach is to sample a vector $x$ from a standard Gaussian distribution $\mathcal{N}(0, I)$, so we'll use this as our input.

Now, we want this Gaussian noise to eventually produce samples from $P_I$. Wait...isn’t this the reverse of the [Ornstein–Uhlenbeck (OU) process ](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#ornstein-uhlenbeck-process)? Yes it is. We start from a stationary distribution $P_I$ and should end in a stationary distribution $\mathcal{N}(0, I)$.
So what we need is to construct a stochastic process such that, as time progresses, $P_I$ evolves into $\mathcal{N}(0, I)$, that is, a **forward diffusion**. Then, by learning its **reverse process**, we can go from Gaussian noise back to realistic images.

Okay, now how do I construct the forward process? Mathematically, the OU process is defined by the stochastic differential equation (SDE):

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

Here:

* $\mu$ is the long-term mean toward which the process is pulled,
* $\theta > 0$ is the rate of mean reversion,
* $\sigma$ controls the intensity of the randomness, and
* $W_t$ is standard Brownian motion (Wiener process).

The Ornstein–Uhlenbeck process solution is [given by](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#ornstein-uhlenbeck-process):

$$
\boxed{
X_t = e^{-\theta t} X_0 + \mu(1 - e^{-\theta t}) + \sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s
}
$$

As $t \to \infty$, we get:
* $\mathbb{E}[X_t] \to \mu$ (since $e^{-\theta t} \to 0$ as $\theta > 0$)
* $\operatorname{Var}(X_t) \to \frac{\sigma^2}{2\theta}$ (since $e^{-2\theta t} \to 0$ as $\theta > 0$)

Therefore,

$$
X_t \overset{d}{\to} \mathcal{N}\left(\mu, \frac{\sigma^2}{2\theta}\right)
$$

The [reverse time equation](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#reverse-time-equation) is given by:   

$$
\boxed{
dX(t) = \left[\mu(X,t) - \sigma(X,t)\sigma(X,t)^\top \nabla_X \log p(X,t)\right]dt + \sigma(X,t)\,d\hat{W}_t
}
$$

