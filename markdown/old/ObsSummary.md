This card summarises explores the observations of a dataset with respect to duplicates, independence and observation missingness.

#### Goal

The goal of this card is to judge whether the data is suitable for data analysis. The potential issues this card addresses are:

 - How many observations are there?
 - What is the distribution of duplicate (and near-duplicate) observations?
 - What is the distribution of missing values in observations?
 - Are the observations independent of each other or are there spatial and/or temporal effects?
 - Are there enough observations (i.e. sufficiently tall?)

The last issue relates to the ratio of observations to variables (specifically predictor variables). A wide dataset is likely to suffer from over fitting.
"Near-duplicates" refer to observations that are the same except for 1,2 or more values. As the number of mismatches grows the nearness diminishes.

#### Actions

The downstream actions implied by the summary issues are:

 * Drop certain observations.
 * Employ dimensional reduction to make the dataset **taller**.
 * Employ pivoting to change the width and height of the dataset.
 * Add an weighting column to the dataset to deal with observation duplication.

#### Examples

> An observation with 70% missingness should be dropped.

> A dataset with 30% duplicated observations should be cleaned up.

> A dataset with some necessary duplicated observations can be redesigned as a dataset with observation weighting.
