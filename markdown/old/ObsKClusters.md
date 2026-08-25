---
output:
  html:
    meta:
      css: ["@tabsets"]
      js: ["@tabsets"]
---
This card explores the number of clusters (K) present in the data. This is done in the context of a number of clustering methods and a variety of ways of using/assessing the cluster.

### Distance

All calculations of distance assume some distance metric. The most obvious and useful is the Euclidean distance. The alternative is the Manhattan distance.

-   *Euclidean*: This is the distance of "flat" space. Euclidean distances are root sum-of-squares of differences.

-   *Manhattan*: Also known as the taxi-cab distance. This is the distance of "grid" space in which movement is constrained to be on a grid and diagonal movement is not possible. Manhattan distances are the sum of absolute differences.

In order to give each variable equal weighting in these distance metrics, the variables are standardised (i.e. transformed to have a mean of 0 and standard deviation of 1.
Since distance is best understood as a numerical quantity, all non-numerical variables will need to be transformed into numerical ones prior to this card. Any remaining factor/text variables are dropped in this analysis.

# Cluster Methods {.tabset .tabset-fade .tabset-pills}

The clustering methods are discussed in separate tabs below:

## Agglomerative

### Agglomerative hierarchical clustering

Agglomerative clustering begins with each observation being its own cluster. Nearby observations are gradually merged into bigger clusters. The process continues until finally there is a single cluster. 
The hierarchy can be "cut" at each number of clusters in the chosen range, K, and assessed using clustering metrics.
Each metric locates an optimum K. These optimum Ks are treated as a vote for the number of clusters. These votes are shown in the bar charts and tabulated in the tables.
This approach assumes that the metrics (see below) are all equally valid, which may not be the case. However, if the majority K value is large enough the conclusion can be treated with some confidence. 
If other clustering methods give different values for K you will need to investigate deeper.

<details>

<summary>Click for the available index names</summary>

-   Ball_Hall
-   Banfeld_Raftery
-   C_index
-   Calinski_Harabasz
-   Davies_Bouldin
-   Det_Ratio
-   Dunn
-   Gamma
-   G_plus
-   GDI 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53
-   Ksq_DetW
-   Log_Det_Ratio
-   Log_SS_Ratio
-   McClain_Rao
-   PBM
-   Point_Biserial
-   Ray_Turi
-   Ratkowsky_Lance
-   Scott_Symons
-   SD_Scat
-   SD_Dis
-   S_Dbw
-   Silhouette
-   Tau Trace_W
-   Trace_WiB
-   Wemmert_Gancarski
-   Xie_Beni

</details>

### Actions

Below are some actions that follow from a better understanding of the hierarchical cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods.
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.

## Divisive

### Divisive hierarchical clustering

Divisive clustering starts with all observations being a single cluster. This is gradually broken into smaller clusters. The process is repeated until each observation is its own cluster.
The hierarchy can be "cut" at each number of clusters in the chosen range, K, and assessed using clustering metrics.
Each metric locates an optimum K. These optimum Ks are treated as a vote for the number of clusters. These votes are shown in the bar charts and tabulated in the tables.
This approach assumes that the metrics (see below) are all equally valid, which may not be the case. However, if the majority K value is large enough the conclusion can be treated with some confidence. 
If other clustering methods give different values for K you will need to investigate deeper.

<details>

<summary>Click for the available index names</summary>

-   Ball_Hall
-   Banfeld_Raftery
-   C_index
-   Calinski_Harabasz
-   Davies_Bouldin
-   Det_Ratio
-   Dunn
-   Gamma
-   G_plus
-   GDI 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53
-   Ksq_DetW
-   Log_Det_Ratio
-   Log_SS_Ratio
-   McClain_Rao
-   PBM
-   Point_Biserial
-   Ray_Turi
-   Ratkowsky_Lance
-   Scott_Symons
-   SD_Scat
-   SD_Dis
-   S_Dbw
-   Silhouette
-   Tau Trace_W
-   Trace_WiB
-   Wemmert_Gancarski
-   Xie_Beni

</details>

### Actions

Below are some actions that follow from a better understanding of the hierarchical cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods.
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.

## Partition

### Partition clustering

Partitioning is a method of assigning points to a set of central values. The size of the set is K, and assessed using clustering metrics.
Each metric locates an optimum K. These optimum Ks are treated as a vote for the number of clusters. These votes are shown in the bar charts and tabulated in the tables.
This approach assumes that the metrics (see below) are all equally valid, which may not be the case. However, if the majority K value is large enough the conclusion can be treated with some confidence. 
If other clustering methods give different values for K you will need to investigate deeper.

There are two main methods for partition clustering. These both utilise the idea of each point belonging to a cluster defined by a nearby centre.

- **k-medoids** Medoids are a type of centre point that are always members of the data set. Medoids are used in the **k-medoids** clustering algorithm which is also known as Partition Around Medoids (PAM). In this algorithm, data is partitioned into clusters by finding medoids that minimize the sum of dissimilarities between points and their nearest medoid. The **k-medoids** algorithm is an alternative to **k-means**, but is more robust to outliers and noise.

- **k-means**  In K-means, each cluster is represented by its arithmetic mean of data points assigned to the cluster. A **centroid** is a data point that represents the centre-of-mass (the mean) of the cluster, and it might not necessarily be a member of the dataset. **K-means** is sensitive to the starting values of the algorithm.

Here are some advantages and disadvantages of the two methods:

K-medoids:

-   is convergent in a predetermined number of steps
-   is less sensitive to outliers than k-means

K-means:

-   works better for clusters that are (somewhat) hollow
-   computes faster but is less repeatable than k-medoid
-   is sensitive to its initial (or assumed starting) centres

Both k-means and k-medoids assume that the cluster is defined by its compactness. In contrast, other types of clustering allow for the clusters to have members defined by their closeness to each other rather than to the centre. This latter style allows for linear, planar, hyper-planar and dounut shaped clusters (amongst others.) Generally the more dimensions (more variables) the data has the less useful clustering will be to reveal much about the structure of the data.

<details>

<summary>Click for the available index names</summary>

-   Ball_Hall
-   Banfeld_Raftery
-   C_index
-   Calinski_Harabasz
-   Davies_Bouldin
-   Det_Ratio
-   Dunn
-   Gamma
-   G_plus
-   GDI 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43, 51, 52, 53
-   Ksq_DetW
-   Log_Det_Ratio
-   Log_SS_Ratio
-   McClain_Rao
-   PBM
-   Point_Biserial
-   Ray_Turi
-   Ratkowsky_Lance
-   Scott_Symons
-   SD_Scat
-   SD_Dis
-   S_Dbw
-   Silhouette
-   Tau Trace_W
-   Trace_WiB
-   Wemmert_Gancarski
-   Xie_Beni

</details>

### Actions

Below are some actions that follow from a better understanding of the partition cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.

## Topology

### Topological clustering

A set of multidimensional points are turned into a topological manifold by agglomerating the points together in a piecemeal fashion. We imagine a distance, ϵ and then merge all points within a distance ϵ of each other. One way to visualize this is to draw a ball with radius ϵ/2 around each point and then connect all points with intersecting balls. Luckily, the Topological Data Analysis (TDA) does not need to determine a reasonable value of ϵ. Instead, a sequence of them are tested.

Initially the points are all individuals "islands" surrounded by a void. As the points are agglomerated (ϵ increasing), these island expand and merge into groups of observations. Islands can merge into the shape of lagoons trapping a void in its centre. Topologically speaking this is a hole. The birth of these islands and voids are recorded and displayed in the Persistence charts.

> The central idea, is that topological features (islands and/or voids) are significant (ie. not noise) if they persist for a long time during the agglomeration process.

There are two main forms of visualisation; a scatter chart and a bar chart. Their interpretations are not obvious. See below for an example. In addition, we can use a silhouette calculation to find the aggregation that best discriminates the clusters. A line chart of silhouette versus agglomeration, helps to understand how reliable such an assessment might be.

Want to read more? Search for "Topological Data Analysis" or start [here](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/)

The dimension = 0 component of the topology is utilised for clusters as dimension = 1 is for detecting holes in the data. The relative lengths of the cluster durations (persistence) is used to derive likelihoods for each K.

### Actions

Below are some actions that follow from a better understanding of the topological structure.

-   Distinguish between clusters due to structure and those due to noise.
-   Verify and investigate why any data gaps exist.
-   Utilise the appropriate number of clusters in subsequent cluster analysis.
-   Verify and investigate why any observational clusters exist.

### Example

Suppose we were to analyse the (artificial) data shown below. ![Circles data](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Circles%20Plot-1.png){width="80%"}

We would hope to discover that Topological Data Analysis says that there are two clusters and there are two "holes" in the data. Let's see what the charts tell us...

Below is the Persistence scatter chart ![Persistence scatter chart](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Plot%20Circle%20Persistence%20Diagram-1.png){width="80%"}

An interpretation of the above chart confirms there are two (off diagonal) black points that suggests two clusters. The two red points (also off diagonal) suggests two data holes.

![Persistence bar chart](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Circles%20Barcode%20Diagram-1.png){width="80%"}

An interpretation of the above chart confirms there are two long black bars that suggests two clusters. The two long red bars suggests two data holes.

------------------------------------------------------------------------

Obviously real data is not as clear cut as this but this example does explain the subtleties of interpreting TDA information.

## Mixture

### Mixture-model clustering

Mixtures modeling attempts to represent data as being from different generating probability distributions. Different mixture models can be investigated. 
Each mixture model locates an optimum K using expectation maximisation (EM). These optimum Ks are treated as a vote for the number of clusters. These votes are shown in the bar charts and tabulated in the tables.
This approach assumes that the mixtures models (see below) are all equally valid, which may not be the case. However, if the majority K value is large enough the conclusion can be treated with some confidence. 
If other clustering methods give different values for K you will need to investigate deeper.

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

### Actions

Below are some actions that follow from a better understanding of the model (mixture) cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.

## Density

### Density based clustering

Density clusters are structures of points that form a chain of close connectedness. In contrast, other types of clustering allow for the clusters to have members defined by their closeness the centre rather than closeness to their neighbours.
In this analysis we employ dbscan to find clusters for a chosen level of closeness.
The closeness parameter (Epsilon) can be varied (along a feasible range) to generate a sequence of cluster counts (K) using density based clustering. 
These Ks are treated as a vote for the number of clusters. These votes are shown in the bar charts and tabulated in the tables.

This approach assumes that the Epsilon values are all equally valid, which may not be the case. However, if the majority K value is large enough the conclusion can be treated with some confidence. 
If other clustering methods give different values for K you will need to investigate deeper.


## Aggregate
This is the an aggregation of the results of the earlier tabs into a single stacked bar chart. 
If you have little preference for which clustering method to employ, this chart makes its easy to decide on the number of clusters as it combines all methods. 
If the conclusion, K, is the same for all methods, there is little point in choosing a specific method.

