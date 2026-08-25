This card assesses variables (particularly predictors) from the point of view of cardinality. The cardinality of a variable is the number of different (i.e. unique) values that it contains across the observations. A variable with low cardinality is said to be discrete. In contrast, a variable with high cardinality is said to be continuous.

The thresholds between low & not-low and between high & not-high are not universally agreed. The default values of 15 and 50 are a good starting point. The thresholds can be changed in the card's settings.

Generally we *expect* decimal measurements to be continuous (i.e. high cardinality). We also *expect* factor-levels to be discrete (i.e. low cardinality). There are exceptions for certain variable roles. A character-based observation-identifier variable will legitimately have high cardinality. A decimal-based observation-weighting variable can legitimately have low cardinality.

#### Goal

The goal of this card is to expose variables to a cardinality assessment. This assessment is conducted from the point of view of the **predictor** role. The potential assessments of a variable are (in ascending importance):

1.  Fine as a continuous predictor (numeric / cardinality is high)
2.  Fine as a discrete predictor (categoric / cardinality is low)
3.  Curious as a continuous predictor (numeric / cardinality is between high and low)
4.  Suspicious as a continuous predictor (numeric / cardinality is low)
5.  Difficult as a discrete predictor (categorical / cardinality is between high and low)
6.  Problematic as a discrete predictor (categorical / cardinality is high)

The last issue category (labelled problematic) should not be ignored.

#### Actions

The downstream actions implied by the important cardinality issues are:

-   Check the data type and derivation of the variable
-   Assign the variable to a non-predictor role
-   Lower the cardinality of the variable e.g. by coalescing levels
-   Drop the variable
-   Extract other variables from the variable (each with lower cardinality)
-   Choose a method that accommodates the cardinality

#### Example

> A variable that contains a paragraph of written comments about makes and models of automobile is bound to have problematic cardinality. Since each paragraph is likely to be unique there will be high cardinality for a discrete variable.\
> This would need to be replaced (through feature engineering) by variables such as: "Topics", "SentenceCount", "WordCount", "Sentiment", etc which might have predictive value. These replacement variables should not generate cardinality issues. The exact replacement variables will depend on the data and upon the analysis goal.
