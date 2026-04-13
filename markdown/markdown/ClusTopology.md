This card explores the topology of the data; specifically any holes and clusters. This is one of five main types of clustering:

1.  **Topological**
2.  Hierarchical
3.  Partition
4.  Model-based
5.  Density-based

#### Goal

The goal of this card is to assess the topological data analysis.

A set of multidimensional points are turned into a topological manifold by agglomerating the points together in a piecemeal fashion. We imagine a distance, ϵ and then merge all points within a distance ϵ of each other. One way to visualize this is to draw a ball with radius ϵ/2 around each point and then connect all points with intersecting balls. Luckily, the Topological Data Analysis (TDA) does not need to determine a reasonable value of ϵ. Instead, a sequence of them are tested.

Initially the points are all individuals "islands" surrounded by a void. As the points are agglomerated (ϵ increasing), these island expand and merge into groups of observations. Islands can merge into the shape of lagoons trapping a void in its centre. Topologically speaking this is a hole. The birth of these islands and voids are recorded and displayed in the Persistence charts.

> The central idea, is that topological features (islands and/or voids) are significant (ie. not noise) if they persist for a long time during the agglomeration process.

There are two main forms of visualisation; a scatter chart and a bar chart. Their interpretations are not obvious. See below for an example. In addition, we can use a silhouette calculation to find the aggregation that best discriminates the clusters. A line chart of silhouette versus agglomeration, helps to understand how reliable such an assessment might be.

Want to read more? Search for "Topological Data Analysis" or start [here](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/)

#### Actions

Below are some actions that follow from a better understanding of the topological structure.

-   Distinguish between clusters due to structure and those due to noise.
-   Verify and investigate why any data gaps exist.
-   Utilise the appropriate number of clusters in subsequent cluster analysis.
-   Verify and investigate why any observational clusters exist.

#### Example

Suppose we were to analyse the (artificial) data shown below. ![Circles data](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Circles%20Plot-1.png){width="80%"}

We would hope to discover that Topological Data Analysis says that there are two clusters and there are two "holes" in the data. Let's see what the charts tell us...

Below is the Persistence scatter chart ![Persistence scatter chart](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Plot%20Circle%20Persistence%20Diagram-1.png){width="80%"}

An interpretation of the above chart confirms there are two (off diagonal) black points that suggests two clusters. The two red points (also off diagonal) suggests two data holes.

![Persistence bar chart](https://reed-math241.github.io/blog/posts/2021-05-10-finding-holes-in-data-using-the-tda-r-package/intro_TDA_files/figure-html5/Circles%20Barcode%20Diagram-1.png){width="80%"}

An interpretation of the above chart confirms there are two long black bars that suggests two clusters. The two long red bars suggests two data holes.

------------------------------------------------------------------------

Obviously real data is not as clear cut as this but this example does explain the subtleties of interpreting TDA information.
