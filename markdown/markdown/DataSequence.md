This card enables the homogeneity of observations to be examined variable by variables. Data that is homogeneous has observations that are similar throughout when viewed in a relevant sequence order. A relevant sequence might to time-order or id-order. We expect relevant sequence variables to have unique values, with the possible exception of longitudinal-study data.

This chart has something in common with control charts. It employs, not consecutive observations but rather, consecutive groups of observations. The reason for this relates to the central limit theorem.

The distribution of any variable can be bell shaped, uniform, saw-tooth, well anything really. We cannot know for sure what that distribution is. It follows that we cannot know if an individual observation is typical or atypical. The solution is to get the variable to fit a distribution we recognise. The central limit theorem allows us to make this transformation. By sampling groups and taking group means, these can be treated as having a Normal distribution.

The "*Maximum number of bins"* settings allows the number of groups to be set provided there are at least 5 observations per bin.

$$size = {N \over bins/chart}$$

There is also a setting for the choice of sequence variable. The default is the natural order (which may be arbitrary, random or meaningful.) If the sequence is truly random, then this card should show nothing of interest. If the sequence is meaningful to the data it may show patterns that can be interpreted.

Since some variables have a wide range, the chart will automatically use a logarithmic transform when necessary.

For continuous data each group a box plot is drawn. For discrete data each group is drawn as a stacked bar showing the proportions of each level.

#### Goal

The goal of this card is to visualise the relevant statistics of consecutive groups of observations. What might be detected and interpreted is:

-   A trend of increasing (or decreasing) median values
-   Fluctuating median values (and small spread) caused by a lack of independence of the observations.
-   Synchronised patterns between the variables, suggesting correlation.
-   Cycles that suggest the data has seasonality
-   The spread of colours in the charts is an indication of cardinality

#### Actions

The downstream actions implied by the data-sequence issues are:

-   Determine the nature of the heterogeneity? Median, Spread, Proportions, Outliers, Missing values?
-   Speculate on the cause of any heterogeneity.
-   Switch to time series analysis / perform time series sampling
-   Experiment with different sequence variables. Speculate on whether they appear to be random orders.

#### Examples

> A variable's data-quality improves over time as experience in measurement and record keeping improves.

> In banking, missing values are highest in the most recent observations as these are still being completed.

> A wine laboratory's data fluctuates widely since it processes wine in vineyard batches that are mainly a single grape variety.

> Medical data may have several possible sequencing variables: Date of treatment(s), Date of initial diagnosis, Date of appointments. Is this longitudinal data? Are operations randomised or do similar operations occur in batches?

