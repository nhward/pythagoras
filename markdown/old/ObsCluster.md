This card visually explores the clusters present in the data, although it can only do this in 2 dimensions. 
In addition, a decision trees attempts to reveal any rules that explain the clusters.
The number of clusters is inherited from the previous card.

There are 5 methods available that will cluster the data to the given number of clusters:

 1. Agglomerative - all points belong to one of K clusters. [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) (bottom up.) 
 1. Divisive - all points belong to one of K clusters. [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) (top down.)
 1. Mixture - all points belong to one of K clusters. [Model based clustering](https://en.wikipedia.org/wiki/Model-based_clustering) using mixtures of normal distributions.
 1. Partition - all points belong to one of K clusters. K-Means and K-Medoids. See [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering).
 1. Density - some points can be left unallocated. Density-based spatial clustering of applications with noise, [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)

#### Data preparation
In order to dimensionally reduce AND in order to calculate distances, it is convenient to just deal with numeric predictor variables. All non-numerical predictor variables will need to be transformed into numerical ones prior to this card.
The Decision tree that attempts to explain the clustering can access more than just the numeric variables. Even so, geometry, character and high-cardinality factor variables are withheld from the Decision tree.

#### Dimensional reduction
The card employs a method for constructing a low dimensional embedding of high-dimensional observation distances.
`t-SNE` is an implementation of Barnes-Hut t-Distributed Stochastic Neighbour Embedding see [here](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

#### Visualisation
The visualisation consists of a pair of charts. The left chart is a 2 dimensional scatter plot of the data colour coded for their allocated clusters.
The right chart is a decision tree chart. If no leaves are visible, then (at that complexity) no decision tree is found. The bottom of the chart shows the classification accuracy. Bear in mind that, due to class imbalance, accuracy may need careful interpretation. 
The reverse side of the card reveals the observation allocation in a tabular form with the same colour coding as the scatter chart.

#### Distance

All calculations of distance assume some distance metric. The most obvious and useful is the Euclidean distance. The alternative is the Manhattan distance.

-   *Euclidean*: This is the distance of "flat" space. Euclidean distances are root sum-of-squares of differences.

-   *Manhattan*: Also known as the taxi-cab distance. This is the distance of "grid" space in which movement is constrained to be on a grid and diagonal movement is not possible. Manhattan distances are the sum of absolute differences.

In order to give each variable equal weighting in these distance metrics, the variables are standardised (i.e. transformed to have a mean of 0 and standard deviation of 1.


#### Actions
Below are some actions that follow from a better understanding of the cluster allocation.

 - Visually assess the quality of clustering.
 - Hypothesise the meaning of the clusters.
 - Add each observation's allocated cluster as a predictive (factor) variable.
 - Whether different methods give rise to significantly different clustering.
 - Identify observations are poorly clustered (and why)
