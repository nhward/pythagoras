This card explores the hierarchical structure of the data; specifically two methods: **Agglomerative** and **Divisive** hierarchy. There are five main types of clustering:

> 1.  Topological
> 2.  **Hierarchical**
> 3.  Partition
> 4.  Model-based
> 5.  Density-based

#### Goal

The goal of this card is to recommend the number of clusters (K) and to assign membership to hierarchy clustered observations. There is also an attempt to explain the clusters.

#### Agglomerative

Agglomerative clustering begins with each observation being its own cluster. Nearby observations are gradually merged into bigger clusters. The process continues until finally there is a single cluster.

#### Divisive

Divisive clustering works in reverse and starts with all observations being a single cluster. This is gradually broken into smaller clusters. The process is repeated until each observation is its own cluster.

Both k-means and k-medoids assume that the cluster is defined by its compactness. In contrast, other types of clustering allow for the clusters to have members defined by their closeness to each other rather than to the centre. This latter style allows for linear, planar, hyper-planar and dounut shaped clusters (amongst others.) Generally the more dimensions (more variables) the data has the less useful clustering will be to reveal much about the structure of the data.

#### Distance

All calculations of distance assume some distance metric. The most obvious and useful is the Euclidean distance.

-   *Euclidean*: This is the distance of "flat" space. Euclidean distances are root sum-of-squares of differences.

-   *Manhattan*: Also known as the taxi-cab distance. This is the distance of "grid" space in which movement is constrained to be on a grid and diagonal movement is not possible. Manhattan distances are the sum of absolute differences.

In order to give each variable equal weighting in these distance metrics, the variables are standardised (i.e. transformed to have a mean of 0 and standard deviation of 1.

#### Number of clusters

The number of clusters (K) is an important quantity to document. For hierarchical clusters, the indices for assessing the quality of clustering are calculated for a range of K and the results are put to a majority vote.

This assumes that the indices are all equally valid (which may not be the case), however if the majority is large enough the conclusion can be treated with some confidence. However if other clustering methods give different values for K you will need to investigate deeper.

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

#### Actions

Below are some actions that follow from a better understanding of the hierarchical cluster structure.

-   Quantify the suitability of different numbers of clusters.
-   Compare the recommended number of clusters to those of other methods.
-   Hypothesise the meaning of any clusters.
-   Add each observation's allocated cluster as a predictive (factor) variable.
