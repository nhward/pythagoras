Data generation processes can drift over time. These can be due to a number of causes which broadly fall into two categories.

-   Measurement drift - that which we measured in the past and, repeated afresh, is now giving different measurements. For example, the ratio of urban-to-rural housing might have changed from 85% to 95% over forty years.

-   Concept drift - the things being measured in the past are not the same things that are being measured now. For example, the concept of a "tiny-house" might not have existed 40 years ago.

> The guaranteed consequence of data drift is the decaying performance of aging models.

An excellent way to assess decaying performance is to monitor model residuals over time. However, this is not always possible and requires the passage of time. This card focusses on what we can assess about drift in our curated data (i.e. the train / test data) given that it spans a reasonable time-period.

#### Goal

The goal of this card is to assess whether the data is changing significantly over the span of time in which data was collected. This investigation turns all problems into weak **times series** ones. Like all time series problems we need to ensure the observations are in the correct sequence.

What is it about predictors and targets that might change over time?

-   The occurrence of values missing from observations.
-   The outlier-rating of observations
-   The means of continuous variables.
-   The spread of continuous variables.
-   The proportions of categorical variables.
-   The covariance structure of continuous variables.

By analysing sets of contiguous observations, the combined statistics tend to have Normal distributions due to the central limit theorem.

Discovering data drift is not unusual and is a feature of all data sets, only the time-frame affects how serious the problem is relative to the lifespan of the model. Social media data sets can expect a drift time-frame of months. Population health can expect a drift time-frame of decades.

#### Actions

Suppose you discover that the predictors and target have drift, what should you do about it?

-   Gain understanding of the causes of drift.
-   Discard the early data in the model building process.
-   Retrain the model periodically with additional fresh data.
-   Add time as a predictive variable.
-   Acknowledge the limitations of the data and the model.

These are all manual actions that cannot and should not be automated.
