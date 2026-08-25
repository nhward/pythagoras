This card explores the model structure of the data; specifically multivariate normal distributional models. This is called mixture modeling and assumes that each cluster is an independent multi-dimensional Normal (or Gaussian) distribution. This card is one of five main types of clustering:

-   Topological
-   Hierarchical
-   Partition
-   **Model-based**
-   Density-based

#### Goal

The goal of this card is to assess the cluster memberships of a set of K different generating probability distributions.

#### Expectation Maximisation

Using the Expectation Maximisation algorithm (EM), and a number of mixture model types, the range of cluster numbers (K) is trialled. The model and k that gives the highest Bayesian Information Criteria (BIC) metric is the best solution.

<details>

<summary>Click for the available model names</summary>

**univariate mixture**

-   "E": equal variance (one-dimensional)
-   "V": variable/unqual variance (one-dimensional)

**multivariate mixture**

-   "EII": spherical, equal volume
-   "VII": spherical, unequal volume
-   "EEI": diagonal, equal volume and shape
-   "VEI": diagonal, varying volume, equal shape
-   "EVI": diagonal, equal volume, varying shape
-   "VVI": diagonal, varying volume and shape
-   "EEE": ellipsoidal, equal volume, shape, and orientation
-   "VEE": ellipsoidal, equal shape and orientation (\*)
-   "EVE": ellipsoidal, equal volume and orientation (\*)
-   "VVE": ellipsoidal, equal orientation (\*)
-   "EEV": ellipsoidal, equal volume and equal shape
-   "VEV": ellipsoidal, equal shape
-   "EVV": ellipsoidal, equal volume (\*)
-   "VVV": ellipsoidal, varying volume, shape, and orientation

**single cluster**

-   "X": univariate normal
-   "XII": spherical multivariate normal
-   "XXI": diagonal multivariate normal
-   "XXX": ellipsoidal multivariate normal

</details>

In order to give each variable equal weighting in these distance metrics, the variables are standardised (i.e. transformed to have a mean of 0 and standard deviation of 1.

#### Actions

Below are some actions that follow from a better understanding of the model (mixture) cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.
