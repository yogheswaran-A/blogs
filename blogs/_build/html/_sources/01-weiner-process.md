
# Understanding Weiner Process

## Motivation
I wrote this blog as I was trying to understand the math behind the diffusion model.Though one can understand the algorithm of differernt types of formulation under diffusion models such as herirachical VAE, Score based models, Rectiflow, flow models without knowing much about SDE, to understand in detail why they all fall under the same category I beleive a deeper understanding of weineer process, OU process is needed, especaily the stochastic calculas which is the key to Ito's lemma, forward and backward equations.     
In this section a brief informal intro to  wiener process and why stocastic calculas is provided and in the later section a formal definition is tried to be given.

### Weiner Process

Informally we can define stocashtic process to be a collection of random varibales $\{X_1 ,X_1,X_2,...\}$ indexed by $t$ (mostly time). At each time $t$ the value is defined by the value $X_t$.  For example consider a ball moving in space at time $t$, if the position($s_t$) of the ball after a time $dt$ is influenced by a random variable $\epsilon_t$, the position at time $t + dt$ would be random. The ball would be moving randomly in space, for each infinitesimally time $dt$ the change in position would be described by $\epsilon_t$. That is:     

$$
ds_t = \epsilon_t * dt
$$   

Notice that unlike normal ODE, it is hard to define the integration of such a process. At each step $dt$ the change is random, so you cannot form a classical Riemann sum in the usual way, each increment $\epsilon_t$ * $dt$ is a random kick and the path $s_t$​ is almost surely nowhere differentiable. To make sense of:   

$$
s_t  =  s_0 + \int ϵ_t dt
$$

we need to **stochastic calculas**. To see the probability  distribution of positions at a given time step, where the ball might end up (final  distribution), how the trajectory evolves for all of this we need stochastic calculas.     
Morever, when we model a ball’s random motion, we’re really looking at an uncountable collection of random variables, one for each instant in a continuous time interval. In other words, each possible trajectory is a function   $s ⁣:[0,T]→R$
and there are uncountably many such functions. Moreover, each trajectory itself involves uncountably many infinitesimal time steps. Because of this double infinity (uncountably many paths, each with uncountably many increments), the naïve, undergraduate definition of probability assign a number to every subset of outcomes breaks down. If one tried to give every possible subset of $R$ (or of the space of trajectories) a probability, one would immediately run into paradoxes like Banach–Tarski: for example, Banach–Tarski shows that you can partition a solid ball into finitely many non‐measurable pieces and reassemble them into two balls of the same volume as the original, so in this case assiging a probability of 1 to the initial solid ball won't work. To avoid such contradictions, modern probability theory restricts attention to a carefully chosen sigma‐algebra of measurable sets (for instance, the Borel sigma‐algebra on $R$). In that way, every event we care about, the ball is in Region A at time $t$, its maximum displacement up to time $t$ lies in some interval, etc., is guaranteed to be measurable, and every non‐measurable, paradoxical subset is simply excluded from the probability model. (We will start probability measure definition and other realted things to understad weiner process in the below sections). 

Let's get back to   

$$
ds_t = \epsilon_t * dt
$$     


here if we chose $\epsilon$ to be a gausian distribution with mean zero informally we will get a weiner process ($W_t$). 

A Wiener process $W_t$​ is a continuous-time stochastic process with the following properties:  

1) $W(0) = 0$. Weiner process starts with zero as the intial condition.
2) Time correlation is zero: The changes in the process over non-overlapping time intervals are independent.    

$$ C[W_{t+\tau} - W_t, W_t] = 0 \quad \text{for } \tau > 0  $$

$$
C[W_{t+\tau} - W_t, W_t] = \lim_{T\to\infty} \frac{1}{T} \int_{0}^{T} W_t * (W_{t+\tau} - W_t)   dt \quad = 0  \quad \text{for } \tau > 0
$$

3) Gaussian increments: The change $W_{t+u} − W_t$​ is normally distributed. 

$$ E[W_{t_{i+1}} - W_{t_i}] = 0 \quad \text{and} $$   
$$ \text{Var}[W_{t_{i+1}} - W_{t_i}] = t_{i+1} - t_i $$

4) Continuous paths: The process has continuous paths in time. There is no discontiuity in the paths. 

The stochastic equation one defined below is encountered more in computer science(rectified flow), physics and finance. We will see how to solve such processes.

$$ dX(t) = \alpha(X,t)dt + \beta(X,t)dW_t $$

where, $\alpha(X,t)$ represents the **drift term**, $\beta(X,t)$ is the **diffusion term**, and $W_t$ represents **weiner process**. This equation describes how a system's state $X(t)$ evolves under both deterministic forces and random fluctuations.

## Probability Spaces And Random Variables

As discussed above we need measure theoretic definition of probability to avoid getting into problems. Lets start with defining sigma algebra which is kinda similar to event space. 

---
**Definition of a σ-algebra** 

Let $\Omega$ be a non-empty set, and let $\mathcal{F}$ be a collection of subsets of $\Omega$.

We say that $\mathcal{F}$ is a **$\sigma$-algebra** if it satisfies the following three properties:

1.  **The empty set belongs to $\mathcal{F}$:**
    $\emptyset \in \mathcal{F}$

2.  **Closed under complement:**
    Whenever a set $A$ belongs to $\mathcal{F}$, its complement $A^c$ also belongs to $\mathcal{F}$.
    If $A \in \mathcal{F}$, then $A^c \in \mathcal{F}$

3.  **Closed under countable unions:**
    Whenever a sequence of sets $A_1, A_2, A_3, \ldots$ belongs to $\mathcal{F}$, their union also belongs to $\mathcal{F}$.
    If $A_1, A_2, A_3, \ldots \in \mathcal{F}$, then
    $\bigcup_{n=1}^\infty A_n \in \mathcal{F}$
---

The elements of a σ-algebra are called **measurable sets**, and the pair (**Ω**, **𝓕**) defines a **measurable space**. Here **Ω** is like sample space, *𝓕* is like the events.

Whenever we want to measure something( like probability) we want our event space *𝓕* which is the subset of **Ω** to be **σ-algebra** so that our measure won't be ill-defined (we will see a example below). By defining our event space like this we are making sure that we add the individual parts of the event space we get a non-overlapping parts, like adding individual pieces of sphere to make up a unit volume. 
For finite event spaces we normally use the power set $2^Ω$ as the event space which has all the properties defined above.    

Now lets define probability measure,

---
**Definition of a Probability Measure**

A **probability measure** **P** on an event space **𝓕** is a function that satisfies the following properties:

1.  **Maps events to the unit interval:**
    $P: \mathcal{F} \rightarrow [0, 1]$

2.  **Returns 0 for the empty set and 1 for the entire space:**
    $P(\emptyset) = 0$ and $P(\Omega) = 1$

3. **Satisfies countable additivity:**  
  For any countable collection of pairwise disjoint events $\{E_i\}_{i \in I}$ (i.e., $E_i \cap E_j = \emptyset$ for $i \neq j$), their probabilities add up:

$$ P\left(\bigcup_{i \in I} E_i\right) = \sum_{i \in I} P(E_i) $$
---

How does this new definition of sample space and probability measure help us? Lets look at The Banach–Tarski paradox, it roughly states that that:

*A solid sphere in 3D space can be decomposed into a finite number of disjoint subsets, which can be reassembled using only rotations and translations to form two identical spheres each the same size as the original.*

This seems to contradict our intuitive understanding of volume and conservation. How can one break one ball into parts and rearrange them into two identical balls? The sets involved in the Banach–Tarski paradox are non-measurable, they cannot be assigned a meaningful volume using any countably additive measure like Lebesgue measure.   

So how σ-algebras and probability measures help, Let's look at the definitions.    

1. The σ-algebra is closed under countable unions and complements.   
2. This closure ensures internal consistency.    
3.  Only subsets of Ω that live in this structure can be measured.   

The Banach–Tarski sets cannot be constructed within a σ-algebra like the Borel σ-algebra.And moe so,

4. The probability measure $P:F→[0,1]P:F→[0,1]$ is only defined on measurable sets (those in $𝓕$).

5. So, any event or set not in 𝓕 is simply not assigned a probability. It lies outside the scope of the theory.

This avoids contradictions like "duplicating" probability mass, because one can only measure events where countable additivity holds.

Banach–Tarski fails here because it relies on unmeasurable subsets (not in 𝓕). There's no way to define a probability measure on all subsets of ℝ³ that:

1. Is countably additive, and
2. Is rotation-invariant, and
3. Assigns the expected volume to regular sets,

without contradicting Banach–Tarski. Hope the above explanation gave some intuition.  Lets define random variable.   

---
A random variable $X$ is a measurable function $X: \Omega \rightarrow E \subseteq \mathbb{R}$ where:

1.  $X$ must be part of a measurable space, $(E, \mathcal{S})$ (recall: $\mathcal{S}$ defines a $\sigma$-algebra on the set $E$). For finite or countably infinite values of $X$, we generally use the powerset of $E$. Otherwise, we will typically use the **Borel set** for uncountably infinite sets (e.g., the real numbers).

2.  For all $s \in \mathcal{S}$, the pre-image of $s$ under $X$ is in $\mathcal{F}$. More precisely:

    $$ \{X \in s\} := \{\omega \in \Omega | X(\omega) \in s\} \in \mathcal{F} $$

---

Thanks