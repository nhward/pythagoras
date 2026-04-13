This card enables the type of "missingness" to be examined and assessed for each variable.

The "missingness" of a variable is the proportion of missing values that it contains across the observations. The type of missingness affects how we should deal with it.

#### Goal

The goal of this card is to assign to each variables that has missing values what the type of missingness is. 
The possible types of missingness are:

-   **Random** A predictor's missingness has no pattern with respect to the non-missing values of other predictors. This is also known as *Missing-completely-at-random (MCAR)*
-   **Patterned** A predictor's missingness has a detectable pattern with respect to the non-missing values of other predictors. This is also known as *Missing-at-random (MAR)*
-   **Because** A predictor's missingness is due to the values that have been made missing. This is also known as *Missing-not-at-random (MNAR)*

There is a secondary goal of creating shadow variables to allow for informative missingness.

##### Shadow variables

It is possible to add extra binary predictors that supply information about whether a value is/was missing. This is a form a feature extraction. 
Once imputed, a missing value would be indistinguishable from a non-missing value unless a shadow variable keeps this information available. 
Since only the predictor role can normally have missing values, only predictor variables will be considered for shadow variables.

##### Informative missingness

The presence of missing values can potentially assist in predicting the target. When this occurs we have "informative missingness". It is important to create shadow variables so that this relationship can be exploited during modeling. Feature selection can later exclude the shadow variables if they prove uninformative.


##### Random Type

Values that are randomly missing are reliably unbiased by virtue of being randomised.

This type of missingness means any attempts at partial deletion, or imputation, have a strong case for being *unbiased*. This means the any statistics from the data should not be biased due to the missing values. However the statistics confidence interval may be widened as a result.

##### Patterned Type

Values whose missingness is patterned can be said to have a probability of missingness that can be predicted to an extent that is statistically better than guessing. This is a set of classification problems. The target is each variable's binary missingness; the predictors are the other variable values. One can also add the sequence number of observations, in case missingness is a time-dependent process.

This type of missingness means any attempts at partial deletion have a strong case for being *biased*. This means that statistics from the data might be biased if partial deletion is employed. The use of imputation does not present a strong case for bias. Imputation is the preferred solution to patterned missingness.

##### Because Type

Values that are missing because of the unseen values. For example,

> Blood-test results can be missing because no diagnostic justification for the test has yet arisen. Were such tests undertaken unnecessarily they would usually yield results in the normal range. A possible manual imputation would involve substituting a typically "Normal" result for each missing test. Substituting a mean result would be a poor choice.

We cannot detect the last type of missingness since the information needed is withheld from us due to being missing. This must be manually analysed through a reasoning process involving what might motivate values to be withheld in the data collection process.

This type of missingness means that statistics from the data might be *biased*, since this type of missingness is a form of sampling bias. This is the worst type of missingness and also the most difficult to detect. Sometimes a reasonable manual imputation strategy exists for this type of missingnes.

#### Actions

Once each variable's type of missingness are distinguished between *Random*, *Patterned* and *Because* the following actions might be possible:

-   Design manual-imputation strategies for some variables (esp. *Because* types)

-   Design statistical-imputation strategies for *Random* and *Patterned* type variables.

-   Design partial-deletion strategies for *Random* type variables.

-   Create shadow variables **before** any imputation 
