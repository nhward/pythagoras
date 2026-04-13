Missing values are common in many data related analyses. In a separate but related card, the **type of missingness** was investigated. In this card we look at the **Map of Missingness**. This card enables the "missingness" of variables and observations to be examined and assessed.

High missingness is a bad characteristic. The threshold at which missingness is no longer acceptable is not universally agreed. The default value of 50% is a good starting point. The threshold can be changed in the card's sidebar for either variables or observations. Setting this to zero is equivalent to partial deletion.

> Sometimes variables that are excessively missing have a simple explanation. Such explanations may suggest a manual imputation that fixes many of the missing values. Bear in mind the subtle difference between *Not Applicable* and *Not Available*.
>
> Sometimes missingness is not constant across time. Data collection processes often improve over time.
>
> Sometimes the observations that are only a few hours old are still incomplete.
>
> It is common for software systems (e.g. banking systems) to collect more personal information as they evolve more complexity over the years. This is evident in a declining missing value rate over time since early records will not have collected the more modern information. For example, email addresses before 2000.

#### Goal

The goal of this card is to assess what the missing data tells us about the processes that generates the data. The interpretation of the assessment varies with the variable's role:

> -   **Predictor** variables often are missing values, however most methods demand that these observations are dealt with by some combination of a) dropped (partial deletion), b) manually supplied, c) automatically imputed.
>
> -   **Target** variables can only tolerate having any missing values in so far as the analysis allows for semi-supervised learning. Otherwise drop these observations.
>
> -   **Identifier**, **Weighting**, **Partition**, **Stratifier**, **Sequence**, **Sensitive**, **Latitude**, **Longitude**, **Treatment**: Other variable roles cannot tolerate having any missing values. Drop these observations.

#### Actions

Below are some actions that follow from a better understanding of the structure of the missing values.

-   Gain understanding of the causes of missing values.
-   Reassign the variable's role
-   Design manual imputation strategies for some variables.
-   Design statistical imputation strategies for other variables.
-   Create shadow variables for appropriate variables.
-   Drop variables that are excessively missing. Drop variables responsibly. Check the variable's importance.
-   Drop observations that are excessively missing. This may result in too few usable observations if the threshold is set too low. Buttons exist for the removal of excessively missing variables and observations. If both are selected, variables will always be processed first so as to minimise the loss of observations.

Bear in mind that if observations whose missingness is greater than 50% are to be dropped, then this also applies to future cases being predicted. This has the advantage of preventing the model producing dubious results where most or all predictors are missing.

#### Example

> A variable records patient results of a rare blood test. The variable is 55% missing because most patients have not needed to have this test. The fact that the test has not been required suggests that patient's results would have been in the normal range. Manually replacing these missing values with a normal result would bring the missingness down. Note that this is not a the same as replacing the missing values with a mean test result due to sampling bias.
