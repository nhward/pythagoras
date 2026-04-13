This card is a summary of a dataset that explores the characteristics of each variable and builds a dissimilarity matrix which is visualised through a hierarchy graph.

#### Goal

The goal of this card is to judge whether the variables form groups that suggest strategies for managing them. The variable characteristics employed in assessing dissimilarity are:

-   The cardinality
-   The centrality (if numeric), maximum scaled to 1 (mean or median depending on use of robust statistics)
-   The spread as a proportion of the centrality (standard deviation or Median Absolute Deviation (mad) depending on use of robust statistics)
-   The skew
-   The data type(s) (using q-grams and cosine distance)
-   The name (using q-grams and cosine distance)
-   The sequence of missing values (using cosine distance)
-   The sequence of value (using correlation distance i.e. $1 - |corr|$)

The various dissimilarity matrices are unified using a weighted mean. Correlation has a four-fold weighting. The q in q-grams can be varied using the "*Q-gram size*" setting.

#### Actions

The actions implied by the dissimilarity are:

-   Investigate variables that have (almost) the same name. What makes them different otherwise?
-   Apply similar preprocessing to variables that are similar.
-   Feature engineer the data to have few similar variables.
-   Change the names of variables that sound similar but are quite dissimilar.
-   Drop variables that are duplicates.
