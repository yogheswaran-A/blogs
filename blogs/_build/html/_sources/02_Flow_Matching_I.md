# Blog 02: Flow Matching: The Theory Behind Stable Diffusion - 3.

Most of the posts start with a pic or a qoute, I'm going to start with a lame joke.

A gaussian noise and MINST dataset fell in love with each other. How did they meet?   
They used flow matching app.

## What is the post about?

Diffusion models and flow matching have improved image generation(they both can be wriiten under the same formulation). In this blog post I will write my learnings about flow matching from the ground up, which was used to develop SD3, open ai SORA, Meta's movie gen video, etc. The topis covered are:

1. Flow matching theory with self contained proofs.
2. Guidance.
2. Training a model on MINST dataset.

I think the MIT's course [Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/) (thanks a lot!!) is much better than what I have written, I learned from it, please take a look at it, they have notes, codes and video lectures. This post is inspired from it.

## Introduction

We want to generate images belonging to $P_{data}$. How can we go about this? Suppose we have some input $X_0$ belonging to some initial distribution $P_0$. At time $t = T$, we want $X_T \sim P_{data}$. So at each time step we follow a path which connects initial distribution $P_0$ and final distribution $P_{data}$. We can write the change of $X_t$ as follows, 

$$
dX_t = U_t^{target}(X_t)\ dt
$$

where 

$$
X_0 \sim P_0 \quad , X_t \sim P_t \ , \text{and} \quad X_T \sim P_{data}
$$

We will often restrict the time to lie between 0 and 1, i.e, $0\le t \le 1$.    
Now the problem simplifies (or becomes complex, we don't know yet) to finding good $U_t^{target}(X_t)$. If we know $U_t^{target}(X_t)$, we can iteratively do the below,

$$
X_{t + \Delta t} = X_t + U_t^{target}(X_t) \Delta t \ \text{; for t starting from 0 and ending at 1.}
$$

This is what flow matching is all about, finding a good $U_t^{target}(X_t)$. Thats it the post is over.   

I always try to ask questions like how does one come up with such any idea or formulation? what could have motivated them to? Those who know Stochastic differential equations might have an idea about this. The langevin dynamics,

$$ 
dX(t) = \mu(X,t)dt + \sigma(X,t)dW_t 
$$

converges to a static distribution (more curious readers can read my previous blog about [stochastic-process-with-affine-drift-co-efficients](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#stochastic-process-with-affine-drift-co-efficients)). We can use this, We can constuct a process such that the initial distribution can be the data distribution, the final can be a normal distribution. We reverse this using reverse time equation (read more on [revrese time equation](https://yogheswaran-a.github.io/blogs/01_weiner_process.html#reverse-time-equation)) 


$$
\boxed{
dX(t) = \left[\mu(X,t) - \sigma(X,t)\sigma(X,t)^\top \nabla_X \log p(X,t)\right]dt + \sigma(X,t)\,d\hat{W}_t
}
$$

to go from known normal distribution to the data distribution. But this is an SDE process, flow matching which I described above is an ODE, so where is the motivation? In the lagevian dynamics if we put $\sigma(X,t) = 0$ we get 

$$ 
dX(t) = \mu(X,t)dt + 0 * dW_t 
$$

the corresponding ODE, which follows the same probability distribution as $P_t$ for $0\le t \le 1$ as the SDE. This could be one of the motivations behind flow matching.   

## Marginal Target

We need to learn $U_t^{target}(X_t)$, but we don't have the ground truth value. So how do we approach this? We will make use of $U_t^{target}(X_t/Z)$, where $Z$ is a data point from the dataset.  $U_t^{target}(X_t/Z)$ can be made to have a simple analytical form, as we will see we can derive  $U_t^{target}(X_t)$ from  $U_t^{target}(X_t/Z)$.

Let, 

$z \sim p_{data}$, $x_t \sim p_t(x/z)$. 

$p_0(x/z)$ the initial distribution is noise, a normal distribution.    
$p_1(x/z)$ the final distribution is a diarc delta centered around $z$, $\delta_z = \delta(x-z)$. 

Then it follows that $p_1(x) \sim p_{data}$

$$
p_1(x \mid z) = \delta(x-z)
$$

and the marginal at time $t = 1$ is:

$$
p_1(x) = \int p_1(x \mid z) \, p_{\text{data}}(z) \, dz
$$

$$
=> p_1(x) = \int \delta(x - z) \, p_{\text{data}}(z) \, dz
$$

by property of diarc delta function, the above integral is zero everywhere except $x = z$, where it's value is $p_{data}(z = x)$

$$
p_1(x) = p_{\text{data}}(x)
$$

Now, we can recover $U_t^{target}(X_t)$ from  $U_t^{target}(X_t/Z)$ using the below.

> ### Theorem 1: Marginal Vector Field Property
> For every data point $z \in \mathbb{R}^d$, let $u_t^{\text{target}}(z)$ denote a conditional vector field, defined so that the corresponding ODE yields the conditional probability path $p_t(\cdot|z)$, viz.,
>
> $ X_0 \sim p_{\text{init}}, \quad \frac{d}{dt}X_t = u_t^{\text{target}}(X_t|z) \implies X_t \sim p_t(\cdot|z) \quad (0 \le t \le 1).$   
>
> Then the **marginal vector field** $u_t^{\text{target}}(x)$, defined by
>
> $u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}dz,$
>
> follows the marginal probability path, i.e.
>
> $$X_0 \sim p_{\text{init}}, \quad \frac{d}{dt}X_t = u_t^{\text{target}}(X_t) \implies X_t \sim p_t \quad (0 \le t \le 1).$$
>
> In particular, $X_1 \sim p_{\text{data}}$ for this ODE, so that we might say "$u_t^{\text{target}}$ converts noise $p_{\text{init}}$ into data $p_{\text{data}}$".

Proof:

$p_t(x)$ can be written as:

$$
p_t(x) = \int p_t(x, z)\, dz = \int p_t(x | z) \, p_{\text{data}}(z) \, dz.
$$

For the conditional distribution, the continuity equation (proof in appendix) holds:

$$
\frac{\partial}{\partial t} p_t(x | z) = -\nabla \cdot (p_t(x | z)\, u_t^{\text{target}}(x | z)),
$$

Multiply both sides by $p_{\text{data}}(z)$, and integrate over $z$:

$$
\int \frac{\partial}{\partial t} p_t(x | z) \, p_{\text{data}}(z) dz = \int -\nabla \cdot \left(p_t(x | z)\, u_t^{\text{target}}(x | z)\right) p_{\text{data}}(z) dz.
$$

Since $p_{\text{data}}(z)$ is independent of $t$ and $x$, we can move the partial derivative with respect to $t$ outside the integral on the LHS, and the divergence operator outside the integral on the RHS:

$$
\frac{\partial}{\partial t} \int p_t(x | z) \, p_{\text{data}}(z) dz = -\nabla \cdot \int p_t(x | z)\, u_t^{\text{target}}(x | z)\, p_{\text{data}}(z) dz.
$$

Using $\int p_t(x | z) \, p_{\text{data}}(z) dz = p_t(x)$. The LHS can be written  as:

$$
\frac{\partial}{\partial t} p_t(x) = -\nabla \cdot \int p_t(x | z)\, u_t^{\text{target}}(x | z)\, p_{\text{data}}(z) dz.
$$

For the marginal distribution $p_t(x)$ to evolve according to $u_t^{\text{target}}(x)$ via the continuity equation, it must satisfy:

$$
\frac{\partial}{\partial t} p_t(x) = -\nabla \cdot (p_t(x) u_t^{\text{target}}(x)).
$$

Comparing the two expressions for $\frac{\partial}{\partial t} p_t(x)$, we must have:

$$
p_t(x) u_t^{\text{target}}(x) = \int p_t(x | z)\, u_t^{\text{target}}(x | z)\, p_{\text{data}}(z) dz.
$$

Now, define the marginal vector field $u_t^{\text{target}}(x)$:

$$
u_t^{\text{target}}(x) := \frac{1}{p_t(x)} \int p_t(x | z)\, u_t^{\text{target}}(x | z)\, p_{\text{data}}(z) dz,
$$

which simplifies to:

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z)\, \frac{p_t(x | z)\, p_{\text{data}}(z)}{p_t(x)} dz.
$$

Therefore, If $X_t \sim p_t(\cdot | z)$ evolves under the conditional vector field $u_t^{\text{target}}(x | z)$, then the marginal vector field

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z)\, p_{\text{data}}(z)}{p_t(x)} dz
$$

satisfies the continuity equation for the marginal $p_t(x)$. This proves the desired result.

## Flow Matching: Constructing the Loss Function

So far we have gained some insight into how to construct $u_t^{\text{target}}(x)$. We will use NN $u_t^\theta$ to apprx $u_t^{\text{target}}(x)$.

Let,

 $$
 X_0 \sim p_{\text{init}}, \quad \frac{d}{dt}X_t = u_t^{\text{target}}(X_t) \implies X_t \sim p_t(x_t) \quad (0 \le t \le 1).
 $$

 We can learn $u_t^\theta$ by minimizing the mean squared loss,

 $$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif[0,1]}, x \sim p_t} \left[\|u_{\theta t}(x) - u_t^{\text{target}}(x)\|^2\right] \\
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif[0,1]}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}\left[\|u_{\theta t}(x) - u_t^{\text{target}}(x)\|^2\right],
$$

The above is called the marginal flow matching loss.
What this says is, First, draw a random time $t \in [0, 1]$. Second, draw a random point $z$ from our data set, sample from $p_t(\cdot|z)$ (e.g., by adding some noise), and compute $u_{\theta t}(x)$. Finally, compute the mean-squared error between the output of our neural network and the marginal vector field $u_t^{\text{target}}(x)$. But here the problem is we do not know $u_t^{\text{target}}(x)$,  

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z)\, \frac{p_t(x | z)\, p_{\text{data}}(z)}{p_t(x)} dz.
$$

The above is not tracable. So what can we do? The idea is to realise we only need gradient of the loss function to update the parameters and ask ourselves what would happen if we replaced $u_t^{\text{target}}(x)$ with $u_t^{\text{target}}(x/z)$ in $\mathcal{L}_{\text{FM}}(\theta)$, 

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}\left[\|u_{\theta t}(x) - u_t^{\text{target}}(x|z)\|^2\right].
$$

This is called the conditional flow matching loss.
For the above can we get any bound? If its an upper bound it would be great, ie $\mathcal{L}_{\text{FM}}(\theta) \le \mathcal{L}_{\text{CFM}}(\theta)$. As you might have guessed, yes we can derive a bound, more specifically the two terms are equal upto a constant term, so thier gradients are equal.  

> ### Theorem 2
> The marginal flow matching loss equals the conditional flow matching loss up to a constant. That is,
>
> $$\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C,$$
>
> where $C$ is independent of $\theta$. Therefore, their gradients coincide:
>
> $$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta).$$
>
> Hence, minimizing $\mathcal{L}_{\text{CFM}}(\theta)$ with e.g., stochastic gradient descent (SGD) is equivalent to minimizing $\mathcal{L}_{\text{FM}}(\theta)$ with in the same fashion. In particular, for the minimizer $\theta^*$ of $\mathcal{L}_{\text{CFM}}(\theta)$, it will hold that $u_t^{\theta*} = u_t^{\text{target}}$.

**Proof:**

We expand the mean-squared error into three components and remove constants.

The marginal flow matching loss, $\mathcal{L}_{\text{FM}}(\theta)$, is defined as:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x)\|^2\right]

$$
Expanding the squared Euclidean norm $\|a-b\|^2 = \|a\|^2 - 2a^T b + \|b\|^2$:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t} \left[\|u_t^{\theta}(x)\|^2 - 2u_t^{\theta}(x)^T u_t^{\text{target}}(x) + \|u_t^{\text{target}}(x)\|^2\right]
$$

Separating the expectation and defining a constant $C_1 = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t} [\|u_t^{\text{target}}(x)\|^2]$ which is independent of $\theta$:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t} \left[\|u_t^{\theta}(x)\|^2 - 2u_t^{\theta}(x)^T u_t^{\text{target}}(x)\right] + C_1
$$

By using the sampling procedure of $p_t$ (which involves marginalizing over $z$ from $p_{\text{data}}$ and $p_t(\cdot|z)$), the expectation can be rewritten:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} \left[\|u_t^{\theta}(x)\|^2 - 2u_t^{\theta}(x)^T u_t^{\text{target}}(x)\right] + C_1
$$

Next, let us re-express the second summand, $\mathbb{E}_{t \sim \text{Unif}, x \sim p_t} [u_t^{\theta}(x)^T u_t^{\text{target}}(x)]$.
By definition of the expectation over $t$ and $x$:

$$
\mathbb{E}_{t \sim \text{Unif}, x \sim p_t} [u_t^{\theta}(x)^T u_t^{\text{target}}(x)] = \int_0^1 \int p_t(x) u_t^{\theta}(x)^T u_t^{\text{target}}(x) \, dx \, dt
$$

Substitute the definition of the marginal vector field $u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x | z) \frac{p_t(x | z)p_{\text{data}}(z)}{p_t(x)} dz$:

$$
= \int_0^1 \int p_t(x) u_t^{\theta}(x)^T \left[\int u_t^{\text{target}}(x | z) \frac{p_t(x | z)p_{\text{data}}(z)}{p_t(x)} dz\right] \, dx \, dt
$$

The $p_t(x)$ terms cancel, and by changing order of integration:

$$
= \int_0^1 \int \int u_t^{\theta}(x)^T u_t^{\text{target}}(x | z) p_t(x | z) p_{\text{data}}(z) \, dz \, dx \, dt
$$

This integral can be re-expressed as an expectation over the relevant distributions:

$$
= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [u_t^{\theta}(x)^T u_t^{\text{target}}(x | z)]
$$

We plug the conditional vector field $u_t^{\text{target}}(x|z)$ into the equation for $\mathcal{L}_{\text{FM}}$ to get:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} \left[\|u_t^{\theta}(x)\|^2 - 2\mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [u_t^{\theta}(x)^T u_t^{\text{target}}(x|z)] \right] + C_1
$$

By adding and subtracting $\|u_t^{\text{target}}(x|z)\|^2$ inside the expectation, and regrouping terms:

$$
= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} \left[\|u_t^{\theta}(x)\|^2 - 2u_t^{\theta}(x)^T u_t^{\text{target}}(x|z) + \|u_t^{\text{target}}(x|z)\|^2 - \|u_t^{\text{target}}(x|z)\|^2 \right] + C_1
$$

This allows us to form the squared difference $\|u_t^{\theta}(x) - u_t^{\text{target}}(x|z)\|^2$:

$$
= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} \left[\|u_t^{\theta}(x) - u_t^{\text{target}}(x|z)\|^2 \right] + \underbrace{\mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)} [-\|u_t^{\text{target}}(x|z)\|^2]}_{C_2} + C_1
$$

Recognizing the first term as $\mathcal{L}_{\text{CFM}}(\theta)$ and combining the constants $C_1$ and $C_2$ into a single constant $C$:

$$
= \mathcal{L}_{\text{CFM}}(\theta) + \underbrace{C_2 + C_1}_{=C}
$$

This concludes the proof.

We can train the $u_t^{\theta}(x)$, we can iterate using the below to get samples.

$$
dX_t = U_{\theta}^t (X_t) dt, \quad X_0 \sim p_{\text{init}}
$$

This procedure is called **Flow Matching**, which is summarized below.   


> ### Algorithm 1: Flow Matching Training Procedure
>
> **Require:** A dataset of samples $z \sim p_{data}$, neural network $u_t^\theta$
>
> 1. **for each** mini-batch of data **do**
> 2. &nbsp;&nbsp;&nbsp;&nbsp;Sample a data example $z$ from the dataset.
> 3. &nbsp;&nbsp;&nbsp;&nbsp;Sample a random time $t \sim \text{Unif}_{[0,1]}$.
> 4. &nbsp;&nbsp;&nbsp;&nbsp;Sample $x \sim p_t(\cdot|z)$
> 5. &nbsp;&nbsp;&nbsp;&nbsp;Compute loss
$ \mathcal{L}(\theta) = \|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2$
> &nbsp;&nbsp;&nbsp;&nbsp;
> 6. &nbsp;&nbsp;&nbsp;&nbsp;Update the model parameters $\theta$ via gradient descent on $\mathcal{L}(\theta)$.
> 7. **end for**

Things simplify we when assume the conditional probability $x \sim p_t(\cdot|z)$ is a gaussian, the SD-3, meta's movie gen are trainined using gaussian conditional path. Lets see what the loss looks like for this case.

## Gaussian Conditional Path

Let $\alpha_t$, $\beta_t$ be noise schedulers: two continuously differentiable, monotonic functions with $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$. The gaussian conditional path is defined as:  

$$p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$$

By the conditions we imposed on $\alpha_t$ and $\beta_t$,    

$p_0(\cdot|z) = \mathcal{N}(\alpha_0z, \beta_0^2 I_d) = \mathcal{N}(0, I_d)$, and $p_1(\cdot|z) = \mathcal{N}(\alpha_1z, \beta_1^2 I_d) = \delta_z $

We can sample $x$ from $p(\cdot|z)$ as follows,

$z \sim p_{data}$, $\epsilon \sim p_{init} = \mathcal{N}(0, I_d) \Rightarrow x = \alpha_t z + \beta_t \epsilon \sim p_t$

Intuitively, the above procedure adds more noise as $t$ goes to $t = 0$ from $T$, at $t = 0$ there is only noise. 

Now to construct $\mathcal{L}_{\text{CFM}}(\theta)$ loss we need $u_t^{\text{target}}(x|z)$, lets derive it.

> ### Theorem 3: 
>
>For $p_t(\cdot|z) = \mathcal{N}(\alpha_tz, \beta_t^2 I_d)$, with $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$.. The conditional target is given by:
>
>$$u_t^{\text{target}}(x|z) = \frac{\dot{\alpha}_t - \dot{\beta}_t}{\beta_t} \alpha_t z + \frac{\dot{\beta}_t}{\beta_t} x$$

**Proof**

Let $X_t$ be defined as:

$$
X_t = \alpha_t z + \beta_t X_0
$$

with $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$, then   

$$
X_t = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot|z).
$$

and at t = 1, $\alpha_1 = 1$, $\beta_1 = 0$

$$
X_1 \sim P_1 = P_{data}
$$

We conclude that the trajectories are distributed like the conditional probability path. Therefore 

The $X_t$ defined above is the ODE trajectory of 

$$
\frac{d}{dt} X_t = u_{t}^{\text{target}}(x|z) \quad \text{for all } x, z \in \mathbb{R}^d
$$

Now lets evalute $u_{t}^{\text{target}}(x|z)$ from the ODE,

$$
\frac{d}{dt} X_t = u_{t}^{\text{target}}(x|z) \quad \text{for all } x, z \in \mathbb{R}^d
$$

$$
\Leftrightarrow \dot{\alpha}_t z + \dot{\beta}_t x_0 = u_{t}^{\text{target}}(\alpha_t z + \beta_t x|z) \quad \text{for all } x_0, z \in \mathbb{R}^d
$$

rewritting $x_0 =  (x_t - \alpha_t z) / \beta_t$

$$
\Leftrightarrow \dot{\alpha}_t z + \dot{\beta}_t \left( \frac{x_t - \alpha_t z}{\beta_t} \right) = u_{t}^{\text{target}}(x_t|z) \quad \text{for all } x_t, z \in \mathbb{R}^d
$$

simplyfing notations $x_t$ as $x$

$$
\Leftrightarrow \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x = u_{t}^{\text{target}}(x|z) \quad \text{for all } x, z \in \mathbb{R}^d
$$

Which proves the theorem.

### Conditional Flow Matching Loss For Gaussian conditional Probability

As we have seen above we have $p_t(\cdot|z) = \mathcal{N}(\alpha_t z; \beta_t^2 I_d)$, we can sample from the conditional path via

$$
x_0 = \epsilon \sim \mathcal{N}(0, I_d) \Rightarrow x_t = \alpha_t z + \beta_t \epsilon \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot|z)
$$

Using the theorem 3,  $u_{t}^{\text{target}}(x|z)$ is given by

$$
u_{t}^{\text{target}}(x|z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

Plugging $u_{t}^{\text{target}}(x|z)$ in $\mathcal{L}_{\text{CFM}}(\theta)$, 

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}\left[\|u_{\theta t}(x) - u_t^{\text{target}}(x|z)\|^2\right].
$$

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)}\left[ \left\| u_t^\theta(x) - \left( \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x \right) \right\|^2 \right]
$$

Substitue $x = \alpha_t z + \beta_t \epsilon$

$$
= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)}\left[ \left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\|^2 \right]
$$

A special case of the above is when when we substitue  $\alpha_t = t$, and $\beta_t = 1 - t$.

Then we have $\dot{\alpha}_t = 1$, $\dot{\beta}_t = -1$, so that

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0,I_d)}[\left\|u_t^\theta(tz + (1 - t)\epsilon) - (z - \epsilon)\right\|^2]
$$


Stable Diffusion 3 and Meta’s Movie Gen Video are trained using this loss function.

> ### Algorithm 2: Flow Matching Training Procedure For Gaussian Conditional Path
>
> **Require:** A dataset of samples $z \sim p_{data}$, neural network $u_t^\theta$
>
> 1. **for each** mini-batch of data **do**
> 2. &nbsp;&nbsp;&nbsp;&nbsp;Sample a data example $z$ from the dataset.
> 3. &nbsp;&nbsp;&nbsp;&nbsp;Sample a random time $t \sim \text{Unif}_{[0,1]}$.
> 4. &nbsp;&nbsp;&nbsp;&nbsp;Sample noise $\epsilon \sim \mathcal{N}(0, I_d)$.
> 5. &nbsp;&nbsp;&nbsp;&nbsp;Set $x = tz + (1-t)\epsilon$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(General case: $x \sim p_t(\cdot|z)$)
> 6. &nbsp;&nbsp;&nbsp;&nbsp;Compute loss
> &nbsp;&nbsp;&nbsp;&nbsp;
> &nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{L}(\theta) = \|u_t^\theta(x) - (z - \epsilon)\|^2 \quad $(General case:$ $ $\mathcal{L}(\theta) = \|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2)$
> &nbsp;&nbsp;&nbsp;&nbsp;
> 7. &nbsp;&nbsp;&nbsp;&nbsp;Update the model parameters $\theta$ via gradient descent on $\mathcal{L}(\theta)$.
> 8. **end for**


## Appendix

### Continuity Equation
TD