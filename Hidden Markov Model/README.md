# Hidden Markov Model
The HMM is a _probabilistic, unsupervised_ machine-learning algorithm that is especially useful when dealing with sequences.

The algorithm discriminates _N_ hidden (latent) states, as well as _M_ observable states.
At any given time _t_, the state of all random variables depend only on the states at time _t - 1_.
This is the main operating assumption of the Hidden Markove model.

## Pros and Cons
### Pros
* Relatively simple
* Interpretable
* Extendable to many real life applications.

### Cons
* For certain situations it can be an oversimplification.

**Note: The hands-on implementation of this algorithm is explained in the notebooks.
However, we have also wrapped it up as a python package that is available in a [separate repository](https://github.com/OlegZero13/markeasy) under MIT license.**


## General formulation
### Notation
The notation that is used throughout the notebooks is given below:
* ![](https://latex.codecogs.com/svg.latex?T) - length of the observation sequence,
* ![](https://latex.codecogs.com/svg.latex?N) - number of states (latent variables),
* ![](https://latex.codecogs.com/svg.latex?M) - number of observables,
* ![](https://latex.codecogs.com/svg.latex?Q=\[q_0,q_1,...,q_{N-1}\]) - distinct states of the model,
* ![](https://latex.codecogs.com/svg.latex?A) - transition probability matrix,
* ![](https://latex.codecogs.com/svg.latex?B) - observation (emission) probability matrix,
* ![](https://latex.codecogs.com/svg.latex?\pi) - initial (latent) probability distribution,
* ![](https://latex.codecogs.com/svg.latex?\mathcal{O}=\[\mathcal{O}_0,\mathcal{O}_1,...,\mathcal{O}_{T-1}\]).

### Definitions
According to the definition, the elements of the transition matrix have the following probabilitic interpretation:
![](https://latex.codecogs.com/svg.latex?a_{i,j}=p(q_{j}^{(t+1)}|q_{i}^{(t)})), 
that is the probability of transitioning from one state to another at the next moment.
The matrix _A_ is an _N-by-N_ matrix.

At the same time
![](https://latex.codecogs.com/svg.latex?b_{j}(k)=p(\mathcal{O}_{k}^{(t)}|q_j^{(t)}))
that is an _M-by-N_ matrix.

Together with an _initial_ state probability ![](https://latex.codecogs.com/svg.latex?\pi=\[q^{(t=0)}]), the Hidden Markov Model is defined as a _collection_:
![](https://latex.codecogs.com/svg.latex?\lambda=(A,B,\pi)).

Consequently, the probability of the hidden state sequence

![](https://latex.codecogs.com/svg.latex?X=(x_0,x_1,...,x_{T-1}))

where

![](https://latex.codecogs.com/svg.latex?\forall_{t\in[0,T-1]}x_t\in{Q}).

with corresponding observations

![](https://latex.codecogs.com/svg.latex?\mathcal{O}=[o_0,o_1,...,o_{T-1}])

the joint probability is given by:

![](https://latex.codecogs.com/svg.latex?p(X,\mathcal{O})=\pi\cdot\prod_{t=0}^{T-1}a_{x_t,x_{t+1}}b_{x_{t},o_{t}}).


### Fundamental problems
There are three fundamental problems that are solvable with HMM:

#### 1. Finding score
Given a model and a sequence of observations, we want to determine the _score_ of the observed sequence with respect to the model.
In other words, given:

![](https://latex.codecogs.com/svg.latex?\lambda=(A,B,\pi))

and

![](https://latex.codecogs.com/svg.latex?\mathcal{O}=[o_0,o_1,...,o_{T-1}])

we want to find

![](https://latex.codecogs.com/svg.latex?p(\mathcal{O}|\lambda))

#### 2. Uncovering hidden variables
Given the model and the observations sequence (as before), we want to fund the most probable hidden sequence that caused the observations.
In other words, we need to find this:

![](https://latex.codecogs.com/svg.latex?p(X|\mathcal{O}))


#### 3. Training the model
Finally, given an observation sequence, together with the model hyper-parameters (_N_, _M_), we want to find the model that maximizes our chances of getting the desired observation sequence.
In other words:

![](https://latex.codecogs.com/svg.latex?\lambda=\arg\max_{A,B,\pi}p(\mathcal{O}|\lambda))

## Examples
### 1. Implementation and finding score
We will go step by step to implement the equations and obtain the score.

### 2. Uncovering hidden variables
We will further expand the library to account for this case.

### 3. Training the model
We will show how to train the model given an abstract example.

### 4. Morning hustle
In this example, we will use our custom library as a source and solve a problem of feeding the cat. We will also perturb the standard HMM by adding time-dependency of the states and use a Monte Carlo simulation to obtain the numerical results.


## References
* "A Revealing Introduction to Hidden Markov Models", Mark Stamp, Dep. of Computer Science, San Jose State University, October 17, 2018.





