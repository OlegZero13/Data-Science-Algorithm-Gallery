# Naive Bayes
The _Naive Bayes_ algorithm tries to _estimate_ the probability of certain event, given the probabilities of "what happened".
Typically these are known or can be calculated from the dataset.
The word _naive_ comes from the assumption that all features (inputs) are independent from each other.

## Pros and Cons
### Pros
* Simple (provided that features are truly independent).
* Requires no training - you calculate the "weights".
* Fast - uses simple multiplications.
* Interpretable - you literally deal with probabilities.
* Generic - useful for classification as well as regression problems.

### Cons
* May not be applicable (e.g. features are somewhat dependent).
* Requires to store the dataset for updates.
* May require additional steps (e.g. finding good probability density functions).

## General formulation
Let ![](https://latex.codecogs.com/svg.latex?\inline&space;x&space;=&space;(x_0,&space;x_1,&space;...,&space;x_{N-1})^T) denote some _input (feature) vector_
and let ![](https://latex.codecogs.com/svg.latex?\inline&space;y) denote the _output_ event we are after.
Note that ![](https://latex.codecogs.com/svg.latex?\inline&space;y) can be a discreet variable or contiuous, or even a vector.

Our goal is to find the _conditional probability_ ![](https://latex.codecogs.com/svg.latex?\inline&space;p(y|x)), which is the probability that ![](https://latex.codecogs.com/svg.latex?\inline&space;y) happens given that all ![](https://latex.codecogs.com/svg.latex?\inline&space;x) happened.

According to Bayes theorem:

![](https://latex.codecogs.com/svg.latex?p(y|x)&space;=&space;\frac{p(x|y)p(y)}{p(x)}).

Here ![](https://latex.codecogs.com/svg.latex?\inline&space;p(x|y)) is the likelihood function, which tells the probability of the particular conditions occur whenever ![](https://latex.codecogs.com/svg.latex?\inline&space;y) actually happened,
![](https://latex.codecogs.com/svg.latex?\inline&space;p(y)) is the _prior_ probability of the event - irrespectively of the conditions, and
![](https://latex.codecogs.com/svg.lates?\inline&space;p(x)) is the evidence probability.

All of these variables (right hand side) **can be calculated**.

### Multiple conditions
In case we have more than one input feature, again, assumming that they are **independent** (hence the word _naive_), we can "AND" them as conditions. Mathematically, that means multiplying of the probabilities:

![](https://latex.codecogs.com/svg.latex?p\left(\bigcap_{j&space;=&space;0}^{N-1}&space;x_j\right)&space;=&space;\prod_{j=0}^{N-1}p(x_j))

As this is applicable to both _likelihood functions_ and the _evidence probabilities_, we can simply:

![](https://latex.codecogs.com/svg.latex?p(y|x)&space;=&space;\frac{\prod_{j=0}^{N-1}p(x_j|y)}{\prod_{j=0}^{N-1}p(x_j)}p(y))

In case of we have more than one output event, we need to calculate the above formula for every event, and basically choose the most probable one:

![](https://latex.codecogs.com/svg.latex?y_k&space;=&space;\arg&space;\max_k&space;p(y|x),&space;\&space;y&space;\in&space;\{&space;y_k\})

## Examples
### Solution 1 - Iris dataset
Here we provide two examples for the algorithm.
The first one uses the well-known Iris dataset, and uses `scikit-learn` implementation as benchmark, taking advantage of Gaussian nature of the data distribution.

### Solution 2 - custom approach
The second example uses a more custom approach.
Here, we choose an artificial toy dataset from [Kaggle](https://www.kaggle.com/carlolepelaars/toy-dataset/version/1).
The dataset, contains four features and one output target.

In the next notebook we will analyze the data and build the model. As you will see, the dataset does not contain data that easily fall under any of supported cases by scikit-learn.
Therefore, we will build our model from scratch, only relying on numpy, scipy and pandas.
