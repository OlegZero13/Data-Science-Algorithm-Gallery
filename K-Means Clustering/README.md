# K-Means Clustering
As unsupervised learning algorithm, _K-Means Clustering_ does not look a the labels at all.
In fact is often used in situations, where data is unlabelled and we seak to understand high-level similarities or even create labels.
In its essence, it is used to formulate groups (clusters) over dataset by first deciding on the number of the clusters and then letting it define these groups based on data similarity.

## Pros and Cons
### Pros
* Simple.
* Interpretable.
* Practical - it is often used for gaining insight into the data itself.

### Cons
* The number of clusters needs to be decided a-priori.
* The distribution of the data (e.g. irregular shapes) may result in creation of counter-intuitive or meaningless clusters.
* Might appear counter-intuitive for higher dimensional data.


## General formulation
Let's assume that our dataset is defined through 
![](https://latex.codecogs.com/svg.latex?x_m\in\mathbb{R}^N)
, where
![](https://latex.codecogs.com/svg.latex?m) is an example number while
![](https://latex.codecogs.com/svg.latex?n)
referring to different features.

The algorithm defines the clusters through so-called _centroids_, which are abstract data points defined by averaging of those data points that are assigned to each cluster.
For it to function, we must have the following defined:
* We must know how to measure the _distance_ in the data space
![](https://latex.codecogs.com/svg.latex?\text{d}(x_m,&space;x_{m'})).
Similarly to the K-Nearest Neighbours, it could be the Minkowski distance or Euclidean distance.
* We must decide on the number of clusters
![](https://latex.codecogs.com/svg.latex?K)
a priori.
* Since the algorithm is peformed in iterations, we must know how to measure its progress and stop it when it no longer makes the progress.
For that, we can define the stop condition through the maximum number of iterations or define a tolerance factor
![](https://latex.codecogs.com/svg.latex?\varepsilon),
which will tell us that if for example the clusters haven't been modified more than this factor, it is basically done.

Given we agreed on the above definitions, the algorithm performs the following steps:
1. We randomly choose _K_ points (or select _K_ points from the dataset at random), to be our centroids.
2. Then, we measure the distance between every point in the dataset and the centroids.
3. We assign each data point to a cluster that is associated with a particular centroid based on the shortest distance.
4. Having all data points assigned, we compute the average of each point (feature wise) within each cluster.
5. The averages just computed define the new centroids for the next iteration (steps 2.-4.).
6. We interate, until we reach the stop condition (e.g. the movement of the centroids becomes small, the number of points in each cluster becomes constant for a few iterations or we simply apply a hard stop).

Within every iteration, the centroids will move as a result of updated points' reference to clusters.
However, with time, the movements will be so insignificant that we can say that the algorithm has _converged to a solution_.


## Examples
### 1. Principles and Custom Implementation
We use an artificially created dataset for that we can decide on its shape, which will make it easier to illustrate the algorithm.

### 2. What can go wrong?
The notebook discusses case where clusters can be ill-formulated when using of the K-Means algorithm.

We also comapre this implementation against the one provided by _scikit-learn_.

### 3. World's Map
In this notebook we try to use KMC algorithm to create artificial countries using population density world's map.
