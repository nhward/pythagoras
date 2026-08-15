#### 🧭 Introduction
This card is a dataset summary that explores the characteristics of each predictor variable and builds a dissimilarity matrix which is visualised through a hierarchy graph.

The variable characteristics employed in assessing dissimilarity are:

-   The cardinality
-   The centrality (if numeric), maximum scaled to 1 (mean or median depending on use of robust statistics)
-   The spread as a proportion of the centrality (standard deviation or Median Absolute Deviation (mad) depending on use of robust statistics)
-   The skew
-   The name (using q-grams and cosine distance)
-   The sequence of missing values (using cosine distance)
-   The sequence of value (using correlation distance i.e. $1 - |corr|$)

The various dissimilarity matrices are unified using a weighted mean. Correlation has a five-fold weighting over the other characteristics. 

#### 🔄 Flip side

The back of the card documents:

 * the dissimilarity matrix. Zero means identical.

#### ⚙️ Settings

Use the Tour Guide button to learn about the settings of this card.

 * Employ robust statistics for correlation, central tendency and spread
 * Size of q-grams
 * Hierachical clustering technique
 * Hierarchical chart layout
 * Limit the maximum number of observations to process

#### 🎯 Goals

 1. Primary: Judge whether the variables form groups that suggest shared strategies for managing them. 
 1. Secondary: Discover surprises that suggest further investgation. 

The actions implied by the dissimilarity are:

 -   Investigate variables that have (almost) the same name. What makes them different otherwise?
 -   Apply similar preprocessing to variables that are similar.
 -   Feature engineer the data to have few similar variables.
 -   Change the names of variables that sound similar but are quite dissimilar.
 -   Drop variables that are effectively duplicates.

***
All cards downstream of this card will receive the upstream data. This card does not change the data (only its appearance).  