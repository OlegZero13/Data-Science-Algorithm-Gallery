# K-Nearest Neighbours
The _KNN_ algorithm tries to estimate the target value (class or value) given similarties of its features with respect to examples already present in the dataset.
The name comes from the algorithm choosing _K_ points that are considered to be the _closest_ to a new point when evaluating the output.

## Pros and Cons
### Pros
* Simple.
* Requires no training - involves no optimization step.
* Interpretable - the logic behind it is clear.
* Generic - useful for both classification and regression problems.
* Easy to tune - _K_ and _distance_ are the only hyper-parameters.

### Cons
* The whole dataset is required for making predictions.
* As the datataset grows, so is the computational effort.


## General formulation
The recipe behind KNN is very simple.
Let's assume our dataset is defined through ![](https://latex.codecogs.com/svg.latex?(x_m,&space;y_m);&space;\quad&space;x_n&space;\in&space;\mathbb{R}^N,&space;y_n&space;\in&space;\mathbb{R},&space;m&space;\in&space;\{1,&space;2,&space;...,&space;M\}) pairs,
where ![](https://latex.codecogs.com/svg.latex?x_m) is an N-dimensional feature vector and ![](https://latex.codecogs.com/svg.latex?y_m) is the corresponding label, both referring to the _m_-th example.
Given a _distance_ ![](https://latex.codecogs.com/svg.latex?\text{d}(x_m,&space;x_u),&space;u&space;\notin&space;\{1,&space;2,&space;...,&space;M\}) is defined,
the algotihm perfoms the following steps:

1. It calculates the distance ![](https://latex.codecogs.com/svg.latex?\text{d}(x_m,&space;x_u)) between our test example ![](https://latex.codecogs.com/svg.latex?x_u) and all other examples in the dataset.
2. Having specified ![](https://latex.codecogs.com/svg.latex?K), it picks ![](https://latex.codecogs.com/svg.latex?K) examples for which the distance is the lowest.
3. If the problem is classification, the output target is the _mode_ of the targets from the selected subspace.
4. If the problem is regression, the _mode_ is replaced with an _average_.

Typically, the distance chosen is the so-called Minkowski distance:

![](https://latex.codecogs.com/svg.latex?\text{d}(x_m,&space;x_u)&space;=&space;\left(&space;\sum_{n=1}^N&space;\left|&space;x_n^{(m)}&space;-&space;x_u^{(u)}&space;\right|^p&space;\right)^{1/p})

If _p=2_, the distance becomes the Euclidean distance:

![](https://latex.codecogs.com/svg.latex?\text{d}(x_m,&space;x_u)&space;=&space;\sqrt{\sum_{n=1}^N&space;\left(&space;x_n^{(m)}&space;-&space;x_u^{(u)}&space;\right)^2&space;})

which is most commonly used.

The (hyper-)parameter _K_ is usually chosen between 3 and 9.
Usually, too low values render the algorithm errorours and more vulnerable to outliers.
Too high values, for a change, result also in lower precission due to admitting too many "not-so-closest" neighbours into the last step.
Finally, for practical reasons, odd values of _K_ are recommended.


## Examples
### Solution 1 - Iris dataset
We use this simple dataset of four-dimensional feature space to demonstrate the implementation.
We also compare this implementation againt the one provided by _scikit-learn_.
