This card employs box-plots and violin plots (and a blend of these) to explore the distributions of many numeric variables at once. By specifying a "facet" variable, the effect of a low-cardinality variable is superimposed in the charts. There exists a special case in which the facet variable has a cardinality of 2. In this situation the violin charts have a split personality.

Since variables are not always in a similar scale, the settings permit normalising the data. Be careful to analyse the quartiles in the context of the current normalisation. With full normalisation, the common y-axis is Z-scores. With no normalisation the common y-axis is a raw numeric scale.

#### Goal

The goal of this card is to make the data more interpretable for data analysis by providing a broad overview. The back side of the card tabulates this information.

#### Actions

Below are some actions that follow from a better understanding of the box/violin plots.

-   Whether the boxplot outliers look plausible or implausible. Note that the chart ones are based on 1.5 multiplier.
-   Whether (unnormalised) variables have different means using the 95 percentile notches setting of the (pure) boxplots.
-   What shapes the distributions take.
-   How the conclusions change with the introduction of a third "facet" variable.

#### Technical note

**Plotly** boxplots are fixed at a multiplier of 1.5 - the settings will not affect the charts.

The tabulation allows the multiplier setting to be reflected in the revised tabulated outliers.
