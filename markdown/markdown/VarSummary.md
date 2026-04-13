Ideally this card would create a multi-dimensional chart showing the data in all dimensions simultaneously. Sadly no such chart has been invented. When it is, a hyper-sentient species will enjoy looking at at it.

While we wait to evolve, this card is the next best thing. It allows the pairwise charting of variables as a single grid of chart combinations. The nature of the variable types involved in each sub-chart determines the appropriate chart to display. By specifying a "facet" variable, the effect of a low-cardinality variable is superimposed in the charts.

#### Goal

The goal of this card is to make the data more interpretable for data analysis by providing a broad overview. Since this is an expensive chart to produce there is necessarily a compromise between speed and completeness to be found.

|                                                                      |
|:--------------------------------------------------------------------:|
| The "*Maximum chart-grid size*" setting limits the size of the grid. |
|   Since the grid is symmetrical, the duplicate calls are omitted.    |

#### Actions

Below are some actions that follow from a better understanding of the pair-wise study of the variables.

-   Whether relationships exist, specifically, whether two variables are dependent or independent.
-   Whether the data shows gaps.
-   Whether the outliers look plausible or implausible.
-   What shapes the distributions take.
-   How the conclusions change with the introduction of a third "facet" variable.

#### Technical note

The grid of charts is generated using **Plotly** and is executed asynchronously using **shiny::ExtendedTask**.
