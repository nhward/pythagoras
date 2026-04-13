Correlation between the predictors is a useful way to characterise the data. This information is in the form of a matrix of values. For many types of correlation this is a symmetric matrix. However, there are some types of correlation where the matrix is asymmetric.

It is entirely optional whether the target variable is included. We expect many variables to be correlated with with the target. It is only surprising if we find no predictors are correlated with the target.

There are several types of correlation:

<center>

|                            | Linear? | Robust? | Symmetric? | Data type |
|----------------------------|---------|---------|------------|-----------|
| **Pearson's Rho**          | yes     | No      | Yes        | Numeric   |
| **Spearman Rank**          | No      | Yes     | Yes        | Numeric   |
| **Kendall's Tau**          | No      | Yes     | Yes        | Numeric   |
| **Predictive Power Score** | No      | No      | No         | Any       |
| **Distance correlation**   | No      | No      | Yes        | Numeric   |
| **Latent correlation**     | No      | No      | Yes        | Numeric   |
| **Xi Correlation**         | No      | No      | Yes        | Numeric   |
| **Mutual information**     | No      | No      | Yes        | Numeric   |

</center>

#### Goal

The goal of this card is to assess what the predictor correlation tells us about the data.

-   Whether sets of variables are correlated together.
-   Whether changing between robust and non-robust types of correlation indicate the presence of outliers.
-   Which variables have negligible (statistically insignificant) correlation.
-   Which pairs of variables are definitely dependent and which might be independent

Generally predictor correlation is never a good thing. It tends to make machine learning either slower to converge or gives the model instability. Methods that ensemble many such unstable models together can successfully overcome this instability.

#### Actions

Below are some actions that follow from a better understanding of the correlation structure of the predictors.

-   Gain understanding of the data.
-   Design ensemble strategies where appropriate.
-   Feature select for independence (i.e. remove one of a pair of highly correlated predictors.)
-   Feature engineer the correlation out of the data (e.g. use dimensional reduction or simply replace a correlated pair with their mean and their range)
-   Deduce whether the relationships are linear or non-linear by comparing linear with non-linear correlation types.
-   Deduce whether outliers are present by comparing robust with non-robust correlation types.
-   Acknowledge the limitations of the data and the model.

These are all manual actions that cannot and should not be automated.
