This card enables the variable sparsity to be assessed. **In this context**, a variable is sparse when most all the values are the same.

This situation is also known as Near Zero Variance (NZV). A sparse variable can be categorical or quantitative. Bear in mind that this is not the common usage of the term "sparseness."

> In data science, "data sparseness" refers to a situation where a large portion of data within a dataset is either missing or set to zero.

If you normalise the variable by subtracting its most common value, we then return to the usual meaning of the "Sparseness."

In the extreme, when all the values are identical, the variable has zero spread (i.e. zero variance if numeric). A constant predictor variable is of no value to modeling and is a problem to linear regression. This situation is mopped up by the sparsity assessment below.

A sparse variable needs to be detected because it is one that is either constant (i.e. zero variance) or is **likely** to become constant when resampled. Resampling necessarily happens during model tuning and/or model assessment. Sparsity is a bad characteristic for some variable roles.

Sparse variables should not be confused with low variance. Not all low-variance variables are sparse but all sparse variables have low variance.

##### Detection methods

1.  Sparse variables (i.e. variables that might become constant when resampled) can be detected using repeated resampling simulation.

2.  A heuristic method for detecting sparse predictors is to use two measures of the predictor which will predict whether it will behave as a sparse variable. This heuristic is shown in this card's chart. The red zone in the lower-right corner is the area that means the variable is sparse. The limit values of 95/5 and 10% are reasonable default thresholds. These default thresholds can be changed in the card's settings.

### Goal

The goal of this card is to expose variables to a near-zero-variance heuristic assessment. The interpretation of the assessment with the variable's role

-   Predictor variables often cannot tolerate sparsity.
-   Target variables cannot tolerate sparsity.
-   Other variable roles sometimes can tolerate sparsity.

### Actions

The downstream actions implied by the presence of sparse variables are:

-   Drop the variable
    -   Late dropping: remove any constant variables during model tuning and assessing. This means that the number of predictors can fluctuate during the (resampling happening during the) modeling process. - **this is what the card's button can engineer**
    -   Early dropping: exclude these predictors ahead of the resampling so that the number of predictors is uniform throughout the (resampling happening during the) modeling process. - **this is what the card's button can engineer**
-   Reassign the variable's role
-   Prevent resampling the rare values into the test set
-   Do nothing and employ a method that accommodates constant predictors

Before dropping a predictor variable it is important to confirm that it is not a strong predictor of the target variable.

### Example

A predictor variable with 800 observations has two distinct values and 799 of them are a single value. To be identified as sparse, first the frequency of the most prevalent value over the second most frequent value (called the “frequency ratio”) must be above **freqCut** (e.g. 95/5). Secondly, the “percent of unique values,” the number of unique values divided by the total number of samples (times 100), must also be below **uniqueCut** (e.g. 10%). The frequency ratio is 799/1 = 799 and the unique value percentage is 1/800 = 0.00125

In practice, when this variable was resampled using 5-fold cross validation, four of the folds were found to be constant for this variable.

The sparsity heuristic (with the default thresholds) identifies this variable as being an issue.
