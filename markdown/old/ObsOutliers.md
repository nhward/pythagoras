Data can have mistakes. Some erroneous observations stand out as outliers, others do not stand out in any obvious way. All that we can say with confidence is that erroneous observations are more likely among the outliers than elsewhere. *In a mining analogy, we seek a suitable rock strata to mine in.*

The hunt for erroneous data becomes the hunt for outliers. However we must be clear that we seek outliers not for their own sake but for the possibility of being erroneous and hoping that a number of these will suggest an explanation about many such cases. *Continuing the analogy, we are looking for the occasional nugget in our diggings and hoping for a seam.*

A search for outliers begins by asking exactly what it is that observations are supposed to be outliers of. There are several possibilities:

-   **Outliers to what is physically or logically possible.** *A house must not have 0 bathrooms.*
-   **Outliers to that which is probable.** *A large house, with 10 occupants, having only a single bathroom.*
-   **Outliers to some assumed "model" of their behaviour.** *Houses with more bathrooms tend to also have more bedrooms.*
-   **Outliers to some assumed rule of the membership of the data.** *A house with 200 bathrooms is probably a hotel rather than a house.*
-   **Outliers to quality norms.** *The housing data from a particular charity may contradict historic council records.*

> A common source of outliers are missing value place-holders (e.g. -999) that have not been identified as such.

#### Goal

The goal of this card is to employ a variety of outlier detection methodologies. Some may be inappropriate and misleading for certain datasets. The supported outlier techniques are:

-   **Mahalanobis distance** which suits single cluster numeric data
-   **Cook's distance** which suits linear relationships amongst numeric data
-   **Local outlier factors** which suits clustered numeric data
-   **One class SVM** which suits non-linear relationships amongst numeric data
-   **Isolation forest** which suits non-linear relationships amongst data

Among these, only **Isolation forest** can tolerate missing values and categorical variables. The **Cook's distance** demands a target variable and can only be used if this role is allocated.

The combined assessment of all these methodologies is a better indication of true outliers than any individual method alone. Any outlier observations must be manually examined to find explanations. This is a task that requires human intervention. Charts cease to help beyond identifying the possible outliers.

#### Actions

Ultimately, an explanation about the identified outliers may lead to:

-   fixing the data collection/extraction/transcription processes.
-   rules for correcting values.
-   greater understanding of the data.
-   excluding zero, one or more sets of observations for justifiable reasons (beyond just being outliers.)
-   the use of "robust"" modeling methodologies to mitigate any remaining outliers.

These are all manual actions that cannot and should not be automated.
