This card both performs imputation and crudely assesses its accuracy.

#### Goal
The goal of this card is to explore imputation as a means of resolving any current or future missing values. Three types of imputation are explored in this card:

 1. **K-nearest-neighbours** (KNN) imputation - *fast*
 1. **Bag imputation** - *slow*
 1. **Simple imputation** (_Median_ for numeric and ordinal variables and _Mode_ for categorical variables) - *very fast*

The first two employ all the (other) predictors to predict the missing values. Note that only Predictors (variables of `role = predictor`) 
play any part in imputation. The laster method only needs to consider a single variable at a time.
Ideally, the predictor variables are independent of each other and cannot assist in predicting a missing value. 
Any performance improvement over simple imputation must exploit predictor dependencies.

The assessment is biased (to over estimate performance) because, currently, the training data doubles as the test data leading to over-fitting.

#### Interpretation

The imputation performance table shows columns: Predictor, Missing, R2 and Accuracy

 * The *Predictor* columns keeps the same order across the imputation styles.
 * The *Missing* column shows the percentage of values that are missing.  
 * The *R^2^* column shows the performance of imputation involving numeric (and ordinal) values. For **KNN** and **Bag Imputation**, the R^2^ value is actually the [correlation](https://en.wikipedia.org/wiki/Correlation) between actual and predicted. For the **Simple imputation** style, the R^2^ value is the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination).
 * The *Accuracy* column shows the performance of imputation involving categorical values.

Weightings, if supplied, are not utilised in the imputations.

#### Actions

The actions implied by the card are:

 * Check the imputation accuracy 
 * Analyse the speed/performance trade-off of imputation methods
 * Increase the earlier excessively-missing-observations threshold to have an easier missing value problem to solve
 * Choose and implement the imputation methodology if the accuracy is acceptable
 * Revert to full partial-deletion through the use 0% as the earlier excessively-missing-observations threshold
 * Leave the missing values in place and plan to utilise methods that tolerate missing values (mainly tree based methods)

You are allowed to specify an imputation strategy for the data, even when there is no evidence of missing values. This is because future data might have missing values that would otherwise prevent predictions being made.
If you perform observation removal for excessively missing cases, (see the _Missingness Map_ card) you can prevent future data being predicted that is excessively missing. Otherwise, you risk allowing predictions where most or all predictors are missing which can lead to meaningless results.
