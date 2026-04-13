This card is a summary of a dataset that explores the variables through their data type, distributional statistics, chart, cardinality and missing values.

#### Goal
The goal of this card is to judge whether the data is suitable for data analysis. The potential issues this card addresses are:

 - How many variables are there and how many are of each data type?
 - Are there variables with the same name?
 - Are there variables whose values are copies of other variables
 - Are there variables whose values are entirely non-negative? i.e. >= 0
 - Are there variables whose values are entirely constant?
 - Are there variables with the wrong data types e.g. ordinal-factors?
 - Are there variables that are highly correlated to others?
 - Are there variables that are linear combinations of others?
 - Do sets of variables cluster with respect to scale and spread?
 - What does the distributional shape suggest about a variable's measurement process?

#### Actions

The actions implied by the summary issues are:

 * Change the names of certain variables
 * Modify the data type of certain variables
 * Assign certain variable to a different role
 * Drop certain variables
 * Extract new variables from the existing variables

#### Examples
> A predictor variable is highly correlated to another predictor. Replace the second with the difference between the two.

> A variable with zero spread can be dropped.

> A variable with a bimodal distribution may be measuring two distinct kinds of observations e.g. houses and offices. Alternatively, two different scales of measurement may be responsible e.g. fahrenheit & centigrade.

