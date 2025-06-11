# Viewing Diffusion, Score, Rectified flow, Heirrachical VAEs From The Same Lens

This blog contains the basics, derivations, intuition and idea behind diffusion models, score based models, Heirrachical VAEs and rectified flow. I have also explained how these formulations relate to one another and comes under the same umberala. I will start with each formulation independently and at the end we will see how all of these fall under the same umberela. 

## Contents
- Diffuions Models

## Pre requistes   
For understanding diffusion: Stochastic differential equations, mainly [SDE](https://ludwigwinkler.github.io/blog/SDE/), [Ito's lemma](https://ludwigwinkler.github.io/blog/ItosLemma/), [OU process](https://ludwigwinkler.github.io/blog/SolvingSDEs/), [Forward And Backward](https://ludwigwinkler.github.io/blog/Kramers/).    
Basic Mathematics like differention, integration.   

## Diffusion Models

To understand this section, knowldge of Stochastic differential equation is required. I have provided only a brief intro and some famous results of SDE which we will use. You can read more about SDE in these blog posts: [SDE](https://ludwigwinkler.github.io/blog/SDE/), [Ito's lemma](https://ludwigwinkler.github.io/blog/ItosLemma/), [OU process](https://ludwigwinkler.github.io/blog/SolvingSDEs/), [Forward And Backward](https://ludwigwinkler.github.io/blog/Kramers/). The author also has other related SDE posts if you want to know more.

Here is the problem statement:  
Given a collection of images, we want to generate a new image that looks similar to those in the dataset. How might we solve this problem using our knowledge of **Stochastic Differential Equations (SDEs)**? Some thoughts:

1. The collection of images represents a distribution of images, denoted by $P_I$. Our goal is to generate a sample from this same distribution.

2. Since neural networks NNs are good at function approximation, we can train a NN to generate an image from the distribution $P_I$.

3. What should the input to the network be? Ideally, it should be something simple and low-dimensional to make inference efficient. A common approach is to sample a vector $x$ from a standard Gaussian distribution $\mathcal{N}(0, I)$, so we'll use this as our input.

Now, we want this Gaussian noise to eventually produce samples from $P_I$. Wait...isn’t this the reverse of the Ornstein–Uhlenbeck (OU) process? Yes it is. We start from a stationary distribution $P_I$ and should end in a stationary distribution $\mathcal{N}(0, I)$.
So what we need is to construct a stochastic process such that, as time progresses, $P_I$ evolves into $\mathcal{N}(0, I)$, that is, a **forward diffusion**. Then, by learning its **reverse process**, we can go from Gaussian noise back to realistic images.

Okay, now how do I construct the forward process? OU process generally have the following SDE:   
$dX_t = \theta (\mu - X_t)\,dt + \sigma\,dW_t$   
where $\theta$ is the momentum parameter that makes the OU process undulate around the mean. The mean parameter $\sigma$ sets the value around which the OU process moves in somewhat smooth arcs.