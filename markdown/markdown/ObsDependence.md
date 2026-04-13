This card is a summary of a hypothesis test regarding the dependence and independence of observations. This is performed for any target and each predictor variable.

The Ljung–Box test may be defined as:

**H<sub>0</sub>**: The data is not correlated (i.e. the autocorrelations in the population from which the sample is taken are 0, so that any observed correlations in the data result from randomness of the sampling process).

**H<sub>a</sub>**: The data exhibits serial correlation.

By its nature this test is only applicable to numeric variables. We have nothing to say at this stage about the dependence/independence of categorical variables.

When a variable is sorted in ascending or descending order, it is to be expected that the observations will not be in a random order. It follows that the actual sort order that is in place for a dataset sets the context for the hypothesis test. The choices for ordering the data are:

-   variables with the role **Sequential**
-   variables with the role **Stratifier**
-   composite variables with the role **Identity**
-   the natural order of the data

#### Goal

The goal of this card is to disprove the **H<sub>0</sub>** hypothesis or fail to disprove the **H<sub>0</sub>** hypothesis for each predictor variable. In other words to disprove a random ordering or to fail to disprove a random ordering ordering of each variable. Note that a variable with auto-correlation cannot be observationally independent. Whereas a variable with no autocorrelation may or may not be observationally independent/dependent.

The dataset sort order is crucial to testing this hypothesis correctly. Typical causes of observational dependence (i.e lack of independence):

-   Time series data where where non-contemporary observations are more different than contemporary ones . The data needs to be sorted by the **sequence** role (e.g time) to detect this.
-   Spatial data where distant observations are more different than nearby ones. The data cannot be sorted by both latitude and longitude at the same time so this is harder to detect.
-   Data that is stratified into groups that are also distance-based clusters. The data needs to be sorted by the **stratifier** role variable to detect this.

Suppose you believe you have tabular data (i.e. data with independent observations) and yet find a variable whose Ljung-Box statistic demonstrates this to be untrue. 
The variable that is dependent must be correlated with the row order. How can this be reconciled? There are two (related) possibilities:

 1. It is possible that the data's natural order has been (accidentally or purposefully) sorted on this variable. Choose a non-predictor/non-target sort order for the data.
 1. It is possible that the variable is not a predictor. If the variable has high cardinality it may be that the variable role is actually **Sequence**. If the variable has low cardinality it may be that the variable role is actually **Stratifier** or **Sensitive** or **Treatment**.

#### Actions

The actions implied by the summary of the hypothesis test are:

-   Adopt time series analysis or spatial analysis or spatio-temporal analysis over tabular data analysis.
-   Adopt time series resampling or spatial resampling or spatio-temporal resampling over tabular data resampling.
-   Reassign predictor variables to non-predictor roles.
