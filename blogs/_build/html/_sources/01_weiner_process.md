
# Understanding Weiner Process

## Motivation
I wrote this blog as I was trying to understand the math behind the diffusion model.Though one can understand the algorithm of differernt types of formulation under diffusion models such as herirachical VAE, Score based models, Rectiflow, flow models without knowing much about SDE, to understand in detail why they all fall under the same category I beleive a deeper understanding of weineer process, OU process is needed, especaily the stochastic calculas which is the key to ITO's lemma, forward and backward equations.     
In this section a brief informal intro to  wiener process and why stocastic calculas is provided and in the later section a formal definition is tried to be given.

## Introduction

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

$$ C[W_{t+\tau} - W_t, W_t] = 0 \quad \text{for } \tau > 0 $$   

$$ C[W_{t+\tau} - W_t, W_t] = \lim_{T\to\infty} \frac{1}{T} \int_{0}^{T} W_t * (W_{t+\tau} - W_t)   dt \quad = 0  \quad \text{for } \tau > 0$$   

3) Gaussian increments: The change $W_{t+u} − W_t$​ is normally distributed.    

$$ E[W_{t_{i+1}} - W_{t_i}] = 0 \quad \text{and}$$   

$$ \text{Var}[W_{t_{i+1}} - W_{t_i}] = t_{i+1} - t_i$$    

4) Continuous paths: The process has continuous paths in time. There is no discontiuity in the paths. 

The stochastic equation one defined below is encountered more in computer science(rectified flow), physics and finance. The ultimate goal of this post is to introduce ITO's lemma and use it to solve such process. At the end we solve a OU process.

$$ dX(t) = \alpha(X,t)dt + \beta(X,t)dW_t $$

where, $\alpha(X,t)$ represents the **drift term**, $\beta(X,t)$ is the **diffusion term**, and $W_t$ represents **weiner process**. This equation describes how a system's state $X(t)$ evolves under both deterministic forces and random fluctuations.

## Probability Spaces And Random Variables

As discussed above we need measure theoretic definition of probability to avoid getting into problems. Lets start with defining sigma algebra which is kinda similar to event space. 

**Definition of a σ-algebra** 

Let $\Omega$ be a non-empty set, and let $\mathcal{F}$ be a collection of subsets of $\Omega$.

We say that $\mathcal{F}$ is a **$\sigma$-algebra** if it satisfies the following three properties:
> 1.  **The empty set belongs to $\mathcal{F}$:**
    $\emptyset \in \mathcal{F}$
> 2.  **Closed under complement:**
    Whenever a set $A$ belongs to $\mathcal{F}$, its complement $A^c$ also belongs to $\mathcal{F}$.
    If $A \in \mathcal{F}$, then $A^c \in \mathcal{F}$
>3.  **Closed under countable unions:**
    Whenever a sequence of sets $A_1, A_2, A_3, \ldots$ belongs to $\mathcal{F}$, their union also belongs to $\mathcal{F}$.
    If $A_1, A_2, A_3, \ldots \in \mathcal{F}$, then
    $\bigcup_{n=1}^\infty A_n \in \mathcal{F}$


The elements of a σ-algebra are called **measurable sets**, and the pair (**Ω**, **𝓕**) defines a **measurable space**. Here **Ω** is like sample space, *𝓕* is like the events.

Whenever we want to measure something( like probability) we want our event space *𝓕* which is the subset of **Ω** to be **σ-algebra** so that our measure won't be ill-defined (we will see a example below). By defining our event space like this we are making sure that we add the individual parts of the event space we get a non-overlapping parts, like adding individual pieces of sphere to make up a unit volume. 
For finite event spaces we normally use the power set $2^Ω$ as the event space which has all the properties defined above.    

Now lets define probability measure,

**Definition of a Probability Measure**

> A **probability measure** **P** on an event space **𝓕** is a function that satisfies the following properties:
>1.  **Maps events to the unit interval:**
    $P: \mathcal{F} \rightarrow [0, 1]$
>2.  **Returns 0 for the empty set and 1 for the entire space:**
    $P(\emptyset) = 0$ and $P(\Omega) = 1$
>3. **Satisfies countable additivity:**  
>  For any countable collection of pairwise disjoint events $\{E_i\}_{i \in I}$ (i.e., $E_i \cap E_j = \emptyset$ for $i \neq j$), their probabilities add up:   
> $P\left(\bigcup_{i \in I} E_i\right) = \sum_{i \in I} P(E_i) $


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
> A **random variable** $X$ is a measurable function $X: \Omega \rightarrow E \subseteq \mathbb{R}$ where:
>1.  $X$ must be part of a measurable space, $(E, \mathcal{S})$ (recall: $\mathcal{S}$ defines a $\sigma$-algebra on the set $E$). For finite or countably infinite values of $X$, we generally use the powerset of $E$. Otherwise, we will typically use the **Borel set** for uncountably infinite sets (e.g., the real numbers).
>2.  For all $s \in \mathcal{S}$, the pre-image of $s$ under $X$ is in $\mathcal{F}$. More precisely:   
>$ \{X \in s\} := \{\omega \in \Omega | X(\omega) \in s\} \in \mathcal{F} $

What this essitially does is that, it helps us to define probability measure for non-numericals (like proability of number of heads less than 3). It helps us map these kind of statement(from the sample space $\Omega$) to a real numbers($E$). The second condition ensures that the numericals we are assiging to the events have well defined probability by making sure $s \in \mathcal{F}$. Also note that the **$\sigma (X) \in \mathcal{F} $**. What this means is that we can assign probability  to all the sets defined by the **$\sigma$-algebra** of $X$ since they are a subset of $\mathcal{F}$. 

## Stochastic Process

Stochastic Process is defined as: 

> Suppose that $(\Omega, \mathcal{F}, P)$ is a probability space, and that $T \subset \mathbb{R}$ is of infinite cardinality. Suppose further that for each $t \in T$, there is a random variable $X_t: \Omega \rightarrow \mathbb{R}$ defined on $(\Omega, \mathcal{F}, P)$. The function $X: T \times \Omega \rightarrow \mathbb{R}$ defined by $X(t, \omega) = X_t(\omega)$ is called a stochastic process with indexing set $T$, and is written as $X = \{X_t, t \in T\}$.

Lets break it down. The elements of the sample space $\Omega$ are infinite. For example, suppose our sample space is tossing of coins inifinite number of times, then $\omega$ one of the elements can be an infinite sequence of only heads.   
The second thing is that previously we said stocashtic process to be a collection of random varibales $\{X_1 ,X_1,X_2,...\}$ indexed by $t$, here $t$ is continous and is inifinite and $X_t$ takes both $t$ and $\omega$. So, $X_t ( \omega)$ takes a infinite element and maps it to real number[$\in \mathcal(F) $] so that we can measure the probability.      

Note that here $X_t$ is not restircted by time, meaning the probaility the random variable $X_t$ assigns to $\omega$ at time $t$ can depend on the future. For example let $\omega$ be 4 heads and thereafter tails $\{H,H,H,H,T,...\}$, so at $t = 4$, the probability the random varibale $X_4$ assings can depend of the future, meaning the occurences of tails. To restrict our time, to be causal we need something called adapted processes. Before definig it we need to see what filtration means.    

If we have want to our random variable to not depend on the future, how could one define it? what restriction can be put in place?    
1. Note that the events are defined by $\mathcal F$.
2. We need to make sure that the random variable $X_t$ depend only on $\{\omega_1,\omega_2,\omega_3,\omega_4,...\omega_t\}$
3. To make this happen we need to modify the event space such that at time $t$ the event space has only events upto time $t$ and not the future. 
4. The events($\mathcal F_t$) at time $t$ must have events till $t$, at time $t+1$ the events $\mathcal F_{t+1}$ should have all the events till time t and events at time $t+1$.

For the above we need a concept called filtration which is defined on our event space $\mathcal F$ and our index $T$, 

> A **filtration** $\mathcal{F}$ is an ordered collection of sub-$\sigma$-algebras $\mathcal{F} := (\mathcal{F}_t)_{t \in T}$ where $\mathcal{F}_t$ is a sub-$\sigma$-algebra of $\mathcal{F}$ and $\mathcal{F}_{t_1} \subseteq \mathcal{F}_{t_2}$ for all $t_1 \le t_2$. 

What this does are the above mentioned points, basically it breaks our $F$ into partitions such that each partition $F_t$ is a superset of $F_{t+1}$.

Now we can define **Adapted process**:   

> A stochastic process $X_t: T \times \Omega$ is **adapted** to the filtration $(\mathcal{F}_t)_{t \in T}$ if the random variable $X_t$ is $\mathcal{F}_t$ - measurable for all $t$. 

Thus $\sigma (X_t)$ is only $F_t$ measurable and $F_t$ does not depend on the future.

## Discrete Time Weiner Process

### Random walk

A random walk is the random walk on the integer number line $\mathcal Z$ which starts at 0, and at each step moves +1 or −1 with equal probability.    

$$X_{n+1}​=X_n​+ \epsilon_n $$
where $\epsilon$ 

$$
\epsilon = \begin{cases}
+1 & p = 0.5 \\
-1 & p = 0.5
\end{cases}
$$

The symmetric random walk $S_n$ is defined as:​
  
$$S_n = \sum_{i=1}^n X_i$$

### Scaled Symmetric Random Walk

The scaled symetric random walk for an positive integer $n$ is defined as:     

$$ 
W_n(t) = \frac{1}{\sqrt{n}} S_{\lfloor nt \rfloor} = \frac{1}{\sqrt{n}} \sum_{i=1}^{\lfloor nt \rfloor} X_i, \quad t \in [0, T] \quad (1)
$$

where $\{X_i\}$ are i.i.d. random variables with

$$
P(X_i = 1) = P(X_i = -1) = \frac{1}{2}, \quad \mathbb{E}[X_i] = 0, \quad \text{Var}(X_i) = 1.
$$

where $S_{nt}$ is the simple random walk defined above and $t$ is continous time. If $nt$ is a integer then $W_n(t)$ takes $S_{nt}$, else it will be an interpolation between the two consecutive $S_{nt}$. For example,lets look at $W_{10}(t)$. For $t = 0$, the $W_{10}(t)$ will be zero, at $t = 10$ it will be $\pm 1$, for $t < 10$ it will be a iterpolation between 0 and $\pm 1$.

Some properties of the Scaled Symmetric Random Walk:

**1. Independence of Increments**   
Let $0 = t_0 < t_1 < t_2 < \dots < t_m$ be points in time, and define the increments of the scaled walk:

$$
W_n(t_1) - W_n(t_0),\quad W_n(t_2) - W_n(t_1),\quad \dots,\quad W_n(t_m) - W_n(t_{m-1}).
$$

These increments are independent.

**Why?**

Each increment involves a sum over disjoint sets of the $X_i$'s:

$$
W_n(t_{i+1}) - W_n(t_i) = \frac{1}{\sqrt{n}} \sum_{j = \lfloor nt_i \rfloor + 1}^{\lfloor nt_{i+1} \rfloor} X_j.
$$

Since the $X_j$'s are independent and the intervals do not overlap, the increments of the scaled walk are also independent.

**2. Expectation and Variance of Increments**   
Let us now compute the expectation and variance of the increment $W_n(t_{i+1}) - W_n(t_i)$.

**2.1 Expectation**

$$
\mathbb{E}[W_n(t_{i+1}) - W_n(t_i)] = \frac{1}{\sqrt{n}} \sum_{j = \lfloor nt_i \rfloor + 1}^{\lfloor nt_{i+1} \rfloor} \mathbb{E}[X_j] = 0.
$$

**2.2 Variance**

$$
\text{Var}(W_n(t_{i+1}) - W_n(t_i)) = \frac{1}{n} \sum_{j = \lfloor nt_i \rfloor + 1}^{\lfloor nt_{i+1} \rfloor} \text{Var}(X_j).
$$  

Since $\text{Var}(X_j) = 1$ for all $j$, and the number of terms in the sum is approximately $n(t_{i+1} - t_i)$, we get:

$$
\text{Var}(W_n(t_{i+1}) - W_n(t_i))  = \frac{1}{n} \sum_{j = \lfloor nt_i \rfloor + 1}^{\lfloor nt_{i+1} \rfloor} \text{Var}(X_j) = \frac{\lfloor nt_{i+1} \rfloor - \lfloor nt_i \rfloor}{n} \quad \approx \frac{1}{n} \cdot n(t_{i+1} - t_i) = t_{i+1} - t_i.
$$

#### **A Interesting Case Arises As $n \to \infty$.**

This is where the scaled symmetric random walk truly becomes fascinating, as it converges to a continuous-time stochastic process known as **Brownian Motion** (or the **Wiener Process**), lets see how and some of it's properties.

**Continuity:**
As $n \to \infty$, the number of steps $\lfloor nt \rfloor$ for any given $t$ becomes infinitely large. The discrete steps of the random walk become infinitesimally small in time and magnitude (due to the $\frac{1}{\sqrt{n}}$ scaling). This effectively smooths out the random walk, and $W_n(t)$ converges in distribution to a continuous path. We no longer need to interpolate between discrete steps, the process itself becomes continuous.

**Independence of Increments (in the limit):**
This property is preserved in the limit. Brownian motion also has independent increments. This means that the future movements of a Brownian motion, given its current position, are independent of its past movements. This is a characteristic feature of many memoryless stochastic processes.

**Expectation in the Limit:**
As derived above, $\mathbb{E}[W_n(t)] = \mathbb{E}[\frac{1}{\sqrt{n}}\sum_{j=1}^{\lfloor nt \rfloor} X_j] = 0$. This holds true as $n \to \infty$. So, for Brownian motion $B(t)$, $\mathbb{E}[B(t)] = 0$. This reflects the no drift characteristic of standard Brownian motion, it's equally likely to move up (+1) or down (-1) over any given time interval.

**Variance in the Limit:**
As shown, $\text{Var}(W_n(t)) = \text{Var}(W_n(t) - W_n(0)) \approx t$. In the limit as $n \to \infty$, this approximation becomes exact. Thus, for Brownian motion $B(t)$, $\text{Var}(B(t)) = t$. This is a defining characteristic, the variance of Brownian motion at time $t$ is simply $t$. This means the spread of the process from its starting point grows linearly with time.

**Convergence to a Normal Distribution (Central Limit Theorem):**
This is the most crucial point for understanding the link between the scaled random walk and Brownian motion. The **Central Limit Theorem (CLT)** states that the sum of a large number of independent and identically distributed (i.i.d.) random variables, when properly normalized, will be approximately normally distributed, regardless of the original distribution of the individual variables.

In our case, $W_n(t) = \frac{1}{\sqrt{n}} \sum_{j=1}^{nt} X_j$ is a sum of $nt$ i.i.d. random variables $X_j$, each with mean $0$ and variance $1$. As $n \to \infty$, the number of terms $nt$ also goes to infinity. Therefore, by the CLT, $W_n(t)$ converges in distribution to a normal random variable.

Specifically, since $\mathbb{E}[W_n(t)] = 0$ and $\text{Var}(W_n(t)) \approx t$, as $n \to \infty$, $W_n(t)$ converges in distribution to a normal distribution with mean $0$ and variance $t$, i.e $\mathcal{N}(0, t)$.

## Weiner process

A stochastic process $\{W(t), t \ge 0\}$ is called a **Wiener process** (or **standard Brownian motion**) if it satisfies the following properties:

1. **Initial Condition:** $W(0) = 0$   
2.  **Independent Increments:**
For all $0 \le t_0 < t_1 < \dots < t_n$, the increments
$W(t_1) - W(t_0), W(t_2) - W(t_1), \ldots, W(t_n) - W(t_{n-1})$
are independent random variables.   
3. **Stationary Increments:**
For $s < t$, the increment $W(t) - W(s) \sim \mathcal{N}(0, t - s)$, i.e., it is normally distributed with mean $0$ and variance $t - s$.     
4. **Continuity:** The sample paths of $W(t)$ are almost surely continuous, meaning:

$$ \mathbb{P} \left( W(t) \text{ is continuous in } t \right) = 1 $$

Additional Notes:
* It is a **Gaussian process**, meaning any finite collection of random variables $(W(t_1), \dots, W(t_n))$ has a multivariate normal distribution.
* It has zero drift and unit volatility:

    $$\mathbb{E}[W(t)] = 0, \quad \text{Var}(W(t)) = t$$

One way to think of weiner process is to imagine watching the motion of a dust particle suspended in air under a microscope. The particle jiggles around in a random, constantly collides with air molecules. This chaotic, continuous, random motion is the physical inspiration for Brownian motion, and the Wiener process is the precise mathematical model of that behavior.

### Total Variation

Taking a look at total variation helps us  understand why weiner process is not differentiable.

For a function $f: [0, T] \to \mathbb{R}$, the **total variation** over $[0, T]$ is:   

$$ \text{TV}(f, [0,T]) = \sup_{\Pi} \sum_{i=0}^{n-1} |f(t_{i+1}) - f(t_i)| $$

where the supremum is taken over all partitions $\Pi = \{0 = t_0 < t_1 < \dots < t_n = T\}$ of the interval.

   
Total variation of a Wiener process (Brownian motion) over any interval $[0,T]$ is infinite almost surely.    
Proof:    
Let $\Delta t = t_{i+1} - t_i$, and remember that:
$W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t)$.
So this increment is a normal random variable with, Mean: $0$, Variance: $\Delta t$.      

Let $X \sim \mathcal{N}(0, \Delta t)$. We want:

$$\mathbb{E}[|X|]$$

Known Result: Expected Absolute Value of a Normal Variable
If $X \sim \mathcal{N}(0, \sigma^2)$, then:

$$ \mathbb{E}[|X|] = \sigma \sqrt{\dfrac{2}{\pi}} $$

In our case, $\sigma = \sqrt{\Delta t}$

$$
\mathbb{E}[|W_{t_{i+1}} - W_{t_i}|] = \sqrt{\Delta t} \sqrt{\frac{2}{\pi}} = \sqrt{\frac{2 \Delta t}{\pi}} 
$$

Suppose we break $[0, T]$ into $n$ equal intervals of length $\Delta t = T/n$. 

For each tiny interval, the expected absolute change is:

$$
\mathbb{E}\left[|W_{t_{i+1}} - W_{t_i}|\right] = \sqrt{\dfrac{2\Delta t}{\pi}} = \sqrt{\dfrac{2T}{n\pi}} 
$$

Summing this over all $n$ intervals:

$$
\text{Expected total variation} \approx n \cdot \sqrt{\dfrac{2T}{n\pi}} = \sqrt{\dfrac{n^2 \cdot 2T}{n\pi}} = \sqrt{\dfrac{2Tn}{\pi}} \to \infty \text{ as } n \to \infty
$$ 

As we use finer and finer partitions (smaller $\Delta t$), we are adding more and more small wobbles, and they add up to infinity.

**Comparison to Smooth Functions**,
For a smooth function, like $f(t) = t$ ( or any polynomial function):
* The absolute change in each small interval goes down linearly with $\Delta t$.
* So the total sum over $n$ intervals stays finite.

Brownian motion is different:
* In each small interval, the change is random.
* The size of the change shrinks like $\sqrt{\Delta t}$, which isn't fast enough to make the total sum converge.    

Imagine zooming in on a Brownian path. Unlike a smooth curve:
* One doesn’t see a line, just more noise

* The closer one zooms, the more erratic the behavior appears

* So every level of detail contributes more total movement

We are summing infinitely many tiny but not tiny enough movements, and they diverge Brownian motion has infinite total variation because it fluctuates so wildly and continuously, at every time scale, it adds up too much wiggle to be finite.
This is why classical calculus breaks down.

### Quadratic Variation.

The **quadratic variation** of $X$ over $[0,T]$ is:

$$
[X]_T = \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} \left( X_{t_{i+1}} - X_{t_i} \right)^2
$$

Where  $\Pi = \{0 = t_0 < t_1 < \dots < t_n = T\}$ is a partition of $[0,T]$ and  $|\Pi| = \max_i (t_{i+1} - t_i)$ is the mesh of the partition. The limit is taken in probability.

**For the smooth functions $f$, the **quadratic variation** is $[f]_T = 0$ for all $T \ge 0$**    

Proof:    
Taking any partition $\Pi = \{0 = t_0 < t_1 < \dots < t_n = T\}$.
Since $f(t) = t$, the increments are:

$$
f(t_{i+1}) - f(t_i) = t_{i+1} - t_i =: \Delta t_i
$$

Then the quadratic variation sum becomes:

$$
\sum_{i=0}^{n-1} (f(t_{i+1}) - f(t_i))^2 = \sum_{i=0}^{n-1} (\Delta t_i)^2
$$

Now suppose the mesh of the partition is $|\Pi| = \max_i \Delta t_i$. Then:

$$
\sum_{i=0}^{n-1} (\Delta t_i)^2 \le |\Pi| \sum_{i=0}^{n-1} \Delta t_i = |\Pi| \cdot T
$$

(because $\sum \Delta t_i = T$, the total length of the interval)

So:

$$
\sum_{i=0}^{n-1} (\Delta t_i)^2 \le T \cdot |\Pi|
$$

Now take the limit as $|\Pi| \to 0$:

$$
[f]_T = \lim_{|\Pi| \to 0} \sum_{i=0}^{n-1} (\Delta t_i)^2 \le \lim_{|\Pi| \to 0} T \cdot |\Pi| = 0
$$

So the quadratic variation of $f(t) = t$ over any interval is zero.    

Now lets compute it for weiner process, recall that weiner moition is higly irregular, the total variation is infinity, the paths are unpredictable. But if we compute the Quadratic variation of the brownian motion, we get something predictable and deterministic. 

**For the Wiener process $W$, the **quadratic variation** is $[W]_T = T$ for all $T \ge 0$ almost surely.**   

Proof: 
The quadratic variation for a partition is:

$$
Q_\Pi = \sum_{j=0}^{n-1} (W(t_{j+1}) - W(t_j))^2 
$$

This quantity is a random variable since it depends on the particular outcome path of the Wiener process (recall quadratic variation is with respect to a particular realized path).

To prove the theorem, we need to show that the sampled quadratic variation converges to $T$ as $|\Pi| \to 0$. This can be accomplished by showing $\mathbb{E}[Q_\Pi] = T$ and $\text{Var}[Q_\Pi] \to 0$, which says that we will converge to $T$ regardless of the path taken.

We know that each increment in the Wiener process is independent; thus their sums are the sums of the respective means and variances of each increment. So given that we have:

$$
\mathbb{E}[(W(t_{j+1}) - W(t_j))^2] = \mathbb{E}[W(t_{j+1}) - W(t_j)]^2 \quad - \quad 0\\
= \mathbb{E}[(W(t_{j+1}) - W(t_j))^2] - \mathbb{E}[W(t_{j+1}) - W(t_j)]^2 \quad \\
= \text{Var}[W(t_{j+1}) - W(t_j)] \\
= t_{j+1} - t_j
$$

We can easily compute $\mathbb{E}[Q_\Pi]$ as desired:

$$
\mathbb{E}[Q_\Pi] = \mathbb{E}\left[\sum_{j=0}^{n-1} (W(t_{j+1}) - W(t_j))^2\right] \\
= \sum_{j=0}^{n-1} \mathbb{E}[(W(t_{j+1}) - W(t_j))^2] \quad \\
= \sum_{j=0}^{n-1} (t_{j+1} - t_j) \\
= T \quad 
$$

From here, we use the fact that the expected value of the fourth moment of a normal random variable with zero mean is three times its variance. Anticipating the quantity we'll need to compute the variance, we have:

$$
\mathbb{E}[(W(t_{j+1}) - W(t_j))^4] = 3\text{Var}[(W(t_{j+1}) - W(t_j))]^2 = 3(t_{j+1} - t_j)^2 \quad 
$$

Computing the variance of the quadratic variation for each increment:

$$
\text{Var}[(W(t_{j+1}) - W(t_j))^2] \\
= \mathbb{E}[((W(t_{j+1}) - W(t_j))^2 - \mathbb{E}[(W(t_{j+1}) - W(t_j))^2])^2] \\
= \mathbb{E}[((W(t_{j+1}) - W(t_j))^2 - (t_{j+1} - t_j))^2] \\
= \mathbb{E}[W(t_{j+1}) - W(t_j))^4 - 2(t_{j+1} - t_j)\mathbb{E}[(W(t_{j+1}) - W(t_j))^2] + (t_{j+1} - t_j)^2] \\
= 3(t_{j+1} - t_j)^2 - 2(t_{j+1} - t_j)^2 + (t_{j+1} - t_j)^2 \\
= 2(t_{j+1} - t_j)^2
$$

From here, we can finally compute the variance:

$$
\text{Var}[Q_\Pi] = \sum_{j=0}^{n-1} \text{Var}[(W(t_{j+1}) - W(t_j))^2] \\
= \sum_{j=0}^{n-1} 2(t_{j+1} - t_j)^2 \\
\le \sum_{j=0}^{n-1} 2|\Pi|(t_{j+1} - t_j) \\
= 2|\Pi|T \quad
$$

As $\lim_{|\Pi| \to 0} \text{Var}[Q_\Pi] = 0$, therefore we have shown that $\lim_{|\Pi| \to 0} Q_\Pi = T$ as required.

**Quadratic variation** of Brownian motion measures the accumulated squared fluctuations of the path.
Even though the path is rough and random, the squared fluctuations add up deterministically:

$$
[W]_t = t
$$

Now if we step back and look at the big picture, this leads to one beautiful result. If we take almost any sample path of a Wiener process and keep dividing the time into smaller and smaller pieces, and then add up the square of the increments, the total will match the length of the time interval. Like, if you're checking over 5 seconds, the sum will be 5. Very clean. It grows at 1 unit per unit of time.

Honestly, that’s kind of mind-blowing. Because the path itself is totally random, jumping randomly, but somehow this squared sum still behaves in a very regular and predictable way. And what’s even more surprising is that the quadratic variation is not zero, even though the path is continuous. Usually, we think if something is continuous and smooth, it shouldn't change that much. But here, it’s continuous, just not differentiable anywhere. It looks smooth from far, but when we zoom in, it’s completely jagged.

Informally we will write the quadratic variation as:

$$ \boxed{ dW(t)dW(t)=dt }$$

Please note that $dW$ is normally distributed, so the above is formally valid only when we sum many of these small squared increments over time. 
 
**Also we can write,**

$$
\boxed{
dW(t)\,dt = 0
}
$$

Proof:   

$$[W(t),t]_T = 0 \Rightarrow dW(t)dt=0$$

Let $X(t)$ and $Y(t)$ be two processes. Their quadratic covariation over interval $[0,T]$ is defined as:

$$
[X,Y]_T = \lim_{|\Pi| \to 0} \sum_{i=1}^{n} \left(X(t_i) - X(t_{i-1})\right) \left(Y(t_i) - Y(t_{i-1})\right)
$$

for a partition $\Pi = \{0 = t_0 < t_1 < \dots < t_n = T\}$

For 

$$ X(t)=W(t), Y(t)=t $$

$$
[W, t]_T = \lim_{|\Pi| \to 0} \sum_{i=1}^{n} \left(W(t_i) - W(t_{i-1})\right) \left(t_i - t_{i-1}\right)
$$

Now notice:

* $W(t_i) - W(t_{i-1}) \sim \mathcal{N}(0, t_i - t_{i-1})$, and
* $t_i - t_{i-1}$ is deterministic.

So this is a sum of independent mean-zero random variables times deterministic numbers.
The expected value of each term is:

$$
\mathbb{E}\left[(W(t_i) - W(t_{i-1}))(t_i - t_{i-1})\right] = \mathbb{E}[W(t_i) - W(t_{i-1})] \cdot (t_i - t_{i-1}) = 0
$$

And the variance of the entire sum becomes small as $|\Pi| \to 0$, because each increment is independent and scaled by vanishingly small $(t_i - t_{i-1})$.

So:

$$
[W, t]_T = 0 \quad
$$

Therefore,

$$
dX(t)\,dY(t) = d[X, Y](t)
$$

$$
dW(t)\,dt = d[W(t), t] = 0
$$

because the process $[W(t), t] \equiv 0$, i.e., it's constant in time.

$$
\boxed{
[W(t), t]_T = 0 \quad \Rightarrow \quad dW(t)\,dt = 0
}
$$

Also note that, 

$$
\boxed{
dt\,dt = 0
}
$$

## ITO's Lemma

From this section onwarads, it's going to be a little bit informal.

Most of the SDE's we encounter(I have encountered) will be of the form:

$$Y(t)=f(t,X(t))$$

Where,

$$ dX(t) = \mu(t)dt + \sigma(t)dW_t $$

$Y(t)$ is a deterministic function of $X(t)$ which is not deterministic. We would like to know how $Y(t)$  changes for very small time differentials $dt$. So we actually want to be able to define the following equation:

$$dY_t=df(t,X_t)$$ 

ITO's lemma provides us a solution to answer this, it states that:

$$ 
\boxed{
 df(t,X(t)) = \left(\frac{\partial f}{\partial t} + \mu(t)\frac{\partial f}{\partial x} + \frac{\sigma^2(t)}{2} \frac{\partial^2 f}{\partial x^2}\right)dt + \frac{\partial f}{\partial x} \sigma(t)dW(t)
}
$$ 

given $f$ is to be twice differentiable with respect to $X_t$ and at least once differentiable with respect to $t$. 

Proof: 

The first step is to define the Taylor expansion for $f(t,X_t)$ in its general:

$$
f(t,X_t) \approx f(t_0,X_0) + \frac{\partial f(t_0,X_0)}{\partial t} \underbrace{(t-t_0)}_{\Delta t} + \frac{\partial f(t_0,X_0)}{\partial X_t} \underbrace{(X_t-X_0)}_{\Delta X_t} + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} \underbrace{(X_t-X_0)^2}_{\Delta X_t^2} \quad \\
$$

$$
= f(t_0,X_0) + \frac{\partial f(t_0,X_0)}{\partial t} \Delta t + \frac{\partial f(t_0,X_0)}{\partial X_t} \Delta X_t + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} \Delta X_t^2 
$$

Applying the limit: $[t→t_0,X_t→X_0]$

$$
\lim_{t \to t_0, X_t \to X_0} f(t,X_t) - f(t_0,X_0) = \lim_{t \to t_0, X_t \to X_0} \frac{\partial f(t_0,X_0)}{\partial t} \Delta t + \frac{\partial f(t_0,X_0)}{\partial X_t} \Delta X_t + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} \Delta X_t^2
$$

$$
df(t,X_t) = \frac{\partial f(t_0,X_0)}{\partial t} dt + \frac{\partial f(t_0,X_0)}{\partial X_t} dX_t + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} dX_t^2
$$

The next step is substituting $ dX(t) = \mu(t)dt + \sigma(t)dW_t $ into the equation:

$$
df(t,X_t) = \frac{\partial f(t_0,X_0)}{\partial t} dt + \frac{\partial f(t_0,X_0)}{\partial X_t} dX_t + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} dX_t^2 \\
= \frac{\partial f(t_0,X_0)}{\partial t} dt + \frac{\partial f(t_0,X_0)}{\partial X_t} (\mu(t)dt+\sigma(t)dW_t) + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} (\mu(t)dt+\sigma(t)dW_t)^2 \quad \\
= \frac{\partial f(t_0,X_0)}{\partial t} dt + \frac{\partial f(t_0,X_0)}{\partial X_t} (\mu(t)dt+\sigma(t)dW_t) \\
\quad + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} (\mu(t)^2 dt^2 + 2\mu(t)\sigma(t) dt dW_t + \sigma(t)^2 dW_t^2) \quad \\ 
= \frac{\partial f(t_0,X_0)}{\partial t} dt + \frac{\partial f(t_0,X_0)}{\partial X_t} (\mu(t)dt+\sigma(t)dW_t) \\
\quad + \frac{1}{2} \frac{\partial^2 f(t_0,X_0)}{\partial X_t^2} (\mu(t)^2 dt\,dt + 2\mu(t)\sigma(t) dt dW_t + \sigma(t)^2 dW_t\,dW_t) \quad \\
$$

Recall from before that,   
* $dW(t)\,dt = 0$
* $dt\,dt = 0 \quad \text{and,}$
* $dW(t)\,dW(t) = 0$   

Applying this we get,

$$
= \underbrace{\left(\frac{\partial f(t_0, X_0)}{\partial t} + \frac{\partial f(t_0, X_0)}{\partial X_t} \mu(t) + \frac{1}{2} \frac{\partial^2 f(t_0, X_0)}{\partial X_t^2} \sigma(t)^2 \right)}_{\text{deterministic}} dt + \underbrace{\frac{\partial f(t_0, X_0)}{\partial X_t} \sigma(t) dW_t}_{\text{stochastic}}
$$

$$ 
\boxed{
df(t,X(t)) = \left(\frac{\partial f}{\partial t} + \mu(t)\frac{\partial f}{\partial x} + \frac{\sigma^2(t)}{2} \frac{\partial^2 f}{\partial x^2}\right)dt + \frac{\partial f}{\partial x} \sigma(t)dW(t)
}
$$

Now lets look into OU process and ITO's lemma application.

## Ornstein-Uhlenbeck Process

The **Ornstein–Uhlenbeck (OU) process** is one of the most fundamental stochastic processes in continuous time. It models systems that evolve randomly, but with a natural tendency to drift back toward a long-term mean. Unlike standard Brownian motion, which wanders off indefinitely, the OU process is mean-reverting, it fluctuates around a central value, making it useful in modeling physical systems, finance, and in diffusion models.

Mathematically, the OU process is defined by the stochastic differential equation (SDE):

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

Here:

* $\mu$ is the long-term mean toward which the process is pulled,
* $\theta > 0$ is the rate of mean reversion,
* $\sigma$ controls the intensity of the randomness, and
* $W_t$ is standard Brownian motion (Wiener process).

The term $\theta(\mu - X_t) \, dt$ causes the process to drift back toward $\mu$ whenever it deviates, while $\sigma\,dW_t$ injects randomness into the motion. This balance between deterministic pull and random noise gives the OU process its characteristic wiggly but stable behavior.

### Ornstein–Uhlenbeck (OU) Process Solution

Lets solve the Ornstein–Uhlenbeck (OU) process, defined by the stochastic differential equation (SDE):

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

This is a linear stochastic differential equation, and it matches the form:

$$
dX_t = -aX_t\,dt + \sigma\,dW_t
$$

except with an inhomogeneous drift $\theta\mu$. So let’s proceed similarly.

$$
dX_t = \theta\mu\,dt - \theta X_t\,dt + \sigma\,dW_t = -\theta X_t\,dt + \theta\mu\,dt + \sigma\,dW_t
$$

Let’s apply the integrating factor method with $e^{\theta t}$. Multiply both sides by $e^{\theta t}$

Let $Y_t = e^{\theta t}X_t$. Then using Itô's product rule:

$$
dY_t = d(e^{\theta t} X_t) = e^{\theta t} dX_t + \theta e^{\theta t} X_t\,dt
$$

Now substitute for $dX_t$:

$$
dY_t = e^{\theta t}(-\theta X_t\,dt + \theta\mu\,dt + \sigma\,dW_t) + \theta e^{\theta t} X_t\,dt
$$

Now cancel the $-\theta X_t\,dt$ and $+\theta X_t\,dt$ terms:

$$
dY_t = \theta\mu e^{\theta t}\,dt + \sigma e^{\theta t}\,dW_t
$$

Integrate both sides

$$
Y_t - Y_0 = \int_0^t \theta\mu e^{\theta s}\,ds + \int_0^t \sigma e^{\theta s}\,dW_s
$$

Since $Y_0 = X_0$, we get:

$$
e^{\theta t} X_t = X_0 + \theta\mu \int_0^t e^{\theta s}\,ds + \sigma \int_0^t e^{\theta s}\,dW_s
$$

Solve for $X_t$

We divide both sides by $e^{\theta t}$:

$$
X_t = e^{-\theta t} X_0 + \mu(1 - e^{-\theta t}) + \sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s
$$

This is the classic Ornstein–Uhlenbeck process solution.


$$
\boxed{
X_t = e^{-\theta t} X_0 + \mu(1 - e^{-\theta t}) + \sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s
}
$$

Let’s see how the mean and variance of the Ornstein–Uhlenbeck process looks like.

$$
X_t = e^{-\theta t} X_0 + \mu(1 - e^{-\theta t}) + \sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s
$$

This expression consists of three terms:
* **Deterministic term**: $e^{-\theta t} X_0$
* **Drift toward mean $\mu$**: $\mu(1 - e^{-\theta t})$
* **Stochastic integral**: $\sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s$

### Mean of $X_t$
We take the expectation of $X_t$:

$$
\mathbb{E}[X_t] = \mathbb{E}[e^{-\theta t} X_0] + \mu(1 - e^{-\theta t}) + \sigma e^{-\theta t} \mathbb{E}\left[\int_0^t e^{\theta s}\,dW_s\right]
$$

Assuming $X_0$ is a constant or an initial random variable with a finite expectation, $\mathbb{E}[e^{-\theta t} X_0] = e^{-\theta t} \mathbb{E}[X_0]$.
The expectation of the stochastic integral $\int_0^t f(s)\,dW_s$ is zero when $f(s)$ is deterministic, $dW_t$ is brownian and has zero mean. Thus, $\mathbb{E}\left[\int_0^t e^{\theta s}\,dW_s\right] = 0$.

So:

$$
\mathbb{E}[X_t] = e^{-\theta t} \mathbb{E}[X_0] + \mu(1 - e^{-\theta t})
$$

**Mean:**

$$
\boxed{
\mathbb{E}[X_t] = \mathbb{E}[X_0]\, e^{-\theta t} + \mu(1 - e^{-\theta t})
}
$$

### Variance of $X_t$
Now we compute the variance. Since the first two terms ($e^{-\theta t} X_0$ and $\mu(1 - e^{-\theta t})$) are deterministic (or at least uncorrelated with the stochastic integral if $X_0$ is independent), and the stochastic integral has zero mean, we have:

$$
\operatorname{Var}(X_t) = \operatorname{Var}\left( \sigma e^{-\theta t} \int_0^t e^{\theta s}\,dW_s \right) \\
= \sigma^2 e^{-2\theta t} \operatorname{Var} \left( \int_0^t e^{\theta s}\,dW_s \right)
$$

We can write,

$$
\operatorname{Var} \left( \int_0^t e^{\theta s}\,dW_s \right) = \mathbb{E} \left[ \left( \int_0^t e^{\theta s}\,dW_s \right)^2 \right] = \int_0^t (e^{\theta s})^2 \, dWdW = \int_0^t (e^{\theta s})^2 \, ds = \int_0^t e^{2\theta s} \, ds \quad
$$

Recall that, $dW\,dW = dt = ds$ which is applied in the above equation.

Now, evaluate the definite integral:

$$
\int_0^t e^{2\theta s}\,ds = \left[ \frac{e^{2\theta s}}{2\theta} \right]_0^t = \frac{e^{2\theta t}}{2\theta} - \frac{e^{2\theta \cdot 0}}{2\theta} = \frac{e^{2\theta t} - 1}{2\theta}
$$

Substitute this back into the variance expression:

$$
\operatorname{Var}(X_t) = \sigma^2 e^{-2\theta t} \left( \frac{e^{2\theta t} - 1}{2\theta} \right) \\
= \frac{\sigma^2}{2\theta} e^{-2\theta t} (e^{2\theta t} - 1) \\
= \frac{\sigma^2}{2\theta} (e^{-2\theta t} e^{2\theta t} - e^{-2\theta t}) \\
= \frac{\sigma^2}{2\theta} (1 - e^{-2\theta t})
$$

**Variance:**

$$
\boxed{
\operatorname{Var}(X_t) = \frac{\sigma^2}{2\theta} (1 - e^{-2\theta t})
}
$$

**Mean:**

$$
\mathbb{E}[X_t] = \mathbb{E}[X_0]\, e^{-\theta t} + \mu(1 - e^{-\theta t})
$$

**Variance:**

$$
\operatorname{Var}(X_t) = \frac{\sigma^2}{2\theta} (1 - e^{-2\theta t})
$$

As $t \to \infty$, we get:
* $\mathbb{E}[X_t] \to \mu$ (since $e^{-\theta t} \to 0$ as $\theta > 0$)
* $\operatorname{Var}(X_t) \to \frac{\sigma^2}{2\theta}$ (since $e^{-2\theta t} \to 0$ as $\theta > 0$)

This shows that the OU process converges in distribution to a stationary Gaussian distribution:

$$
X_t \overset{d}{\to} \mathcal{N}\left(\mu, \frac{\sigma^2}{2\theta}\right)
$$

```python
# simulating OU process

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the OU process
theta = 0.7      # rate of mean reversion
mu = 0.0         # Long-term mean
sigma = 0.3      # intensity of the randomness
X0 = 1.0         # Initial value
T = 10.0         # Total time
dt = 0.01        # Time step
N = int(T / dt)  # Number of time steps

# Pre-allocate array for efficiency
X = np.zeros(N)
X[0] = X0

# Generate the OU process
for t in range(1, N):
    dW = np.sqrt(dt) * np.random.normal(0, 1)
    X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + sigma * dW

# Plot the result
plt.plot(np.linspace(0, T, N), X)
plt.title("Ornstein-Uhlenbeck Process Simulation")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()
```

![img](data/01-weiner-process/OU_simu.png)

## Reverse Time Equation

In the above OU process graph, we can see that after some time $T$ the final state is a sample from a normal distribution. Now we can ask an interesting question here, given that we have the final state(a sample from normal distribution), can we trace back and reach the initial state? That is can we traverse back in time to reach a state from the initial distribution?
Or more broadly for a stochastic process we have the forward equation given by :   

$$ 
dX(t) = \mu(t)dt + \sigma(t)dW_t 
$$

for some initial state $X_0$. If we let this proces continue for some time $T$ and let the state be now $X_T$. Can go back in reverse direction? Can we start from $X_T$ a state from the final distribution and go back to a state from the intial distribution? The answer to that is YES!!.

The reverse time equation allows us to do this, which is given by:

$$
dX(t) = \left(\mu(t) - \sigma^2(t)\nabla_X \log p(X(t),t)\right)dt + \sigma(t)d\hat{W}_t
$$

