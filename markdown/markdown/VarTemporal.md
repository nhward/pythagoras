---
output:
  html:
    meta:
      css: ["@tabsets"]
      js: ["@tabsets"]
---
This card explores the time-series character of the data, in particular, _regular interval_ data. This means that the time elapsed between observations is the same throughout.
It is a requirement that the data be convertible to a **tsibble**. This means it must have its _Sequence_ role assigned. The combination of each _Sequence_ and the _Identifier_ must be unique.

#### Example - Longitudinal data

The sequence variable is recorded as a date when a patient has their appointment. 
This is not unique since many patients have appointments on the same day. 
The identifier variable is the patient id. Together the patient id and date ensures each observation has a unique index value.

If a patient visits twice in a single day, the data set would not be able to be converted into a **tsibble**. This could be fixed by changing the visit date to visit date&time.

# Charts {.tabset .tabset-fade .tabset-pills}
Each of the following charts explores a different aspect of time series data.

## Line chart

The **Line chart** is a series of line charts that span the range of dates covered by the data. Each line is a separate identifier across the (time) sequence. In this context an Identifier + Sequence is the fully unique key. When in full-screen mode a legend explains the lines.

## Seasonal chart

The **Seasonal chart** shows the distributions of over a "full" season. What constitutes a full season depends upon the data interval.

| Interval   | Season   |
|:----------:|:--------:|
| Year       | Year     |
| Quarter    | Year     |
| Month      | Year     |
| Week       | Year     |
| Day        | Week     |
| Hour       | Day      |
| Minute     | Day      |
| Second     | Day      |

The goal is to wrap  the data around each season and chart the distributions that each interval sees.

## Tiled chart

## Deconstruction
The **Deconstruction chart** breaks the time series data into components:

 * **Trend:** This is the smooth change over time. This does not have to be a straight line. Gentle curves are permitted. 
 * **Seasonality:** This component of the time series into a repeating pattern. This is limited to a single seasonality although there are situations in which there may be yearly and weekly patterns. Choose the longer period if in doubt.
 * **Noise:** This component is what is left over after the previous two components have be taken out of the original data. The goal of time series modeling is to make this component have a small amplitude. To check this, bear in mind that the three components need not have the same Y scale. A good interpretation needs to consider the noise scale compared to the others.

## Correlation

The "measurement" variables (which can have Predictor or Target roles) are correlated with their earlier values. 
This assumes the observations are in the sequence and have regular time-steps. This is called **Autocorrelation**. 
Autocorrelation does not tolerate missing values or interval gaps in the data.

Two forms of correlation can be charted:

* _Full autocorrelation:_ This is simply the normal Pearson's correlation based. The chart's X axis is the number of lags performed.
* _Partial autocorrelation:_ This the *additional* Pearson's correlation introduced by the l'th lag assuming all earlier lag-variables will be used as extra predictors. 

Partial autocorrelation tends to favour smaller lags, than full autocorrelation.

A threshold correlation, that marks where autocorrelation becomes significantly non-zero, can be drawn on the charts. As the lags get larger partial autocorrelation typically falls below this threshold and rarely peaks above it again.

#### Example
For daily interval data, the partial correlation chart might show the bars fall below the significance threshold after 5 lags (i.e. 5 lags).
In contrast the full correlation chart might show the correlation rise above the significance threshold on day 7 suggesting a weekly pattern exists. There may even be a small peak appearing at lag 365 suggesting a yearly pattern.

## Strength
The strength chart is related to the time series deconstruction. It is a scatter chart where the x axis is the "strength" of the trend, and the Y axis is the "strength" of the seasonality. Each participant in a longitudinal study (or groups similar one) can each provide a point in the scatter chart.
This helps to identify outlier participants.

## Spectral


#
