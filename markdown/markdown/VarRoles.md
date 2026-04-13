This card allows the assignment of roles to variables. Unassigned variables are treated as unnecessary variables."),

#### Goal

The goal of this card is to make the data more interpretable for data analysis by knowing how each variable is expected to be used in the analysis. The allowed variable-roles this card supports are:

-   **Target** - the variable that is being predicted (also known as the **outcome** role)
-   **Predictor** - variables used to make the predictions (which is also true of **Treatment**)
-   **Identifier** - one or more variables to uniquely identify each observation
-   **Weighting** - an importance or repeat count of an observation (also known as **case_weights**)
-   **Partition** - a factor that divides observations into test or train splits
-   **Stratifier** - a factor that groups similar observations together but is not meaningfully predictive
-   **Sequence** - variables (usually data/time) that give an order to the observations but are not directly predictive themselves
-   **Sensitive** - factor variables that record features that should not be predictive and can be reviewed for fairness
-   **Treatment** - a factor variable that may have a _causal_ influence over the **Target**
-   **Geometry** - spatial variables that record points, lines, and polygons (or collections thereof)

Only the **Predictor** role needs to be filled. Not every variable needs to be assigned to a role. Unassigned variables are not used.
Currently a variable cannot not have more than one role. If this proves to be problematic, create a duplicate variable which can be assigned an different role.
When tabular data is first loaded, everything is a predictor. The process of importing spatial data (shape files) will automatically assign the special geometry variables to the **Geometry** role.

**Predictor**, **Sequence**  and **Identifier** are the only roles that can be assigned to 0, 1 or several variables.
Roles have requirements affecting data type and cardinality. Not meeting these requirements will generate validation messages.
The assignment of the **Target** role affects whether the analysis is supervised or unsupervised. A factor target implies classification; a numeric target implies regression.

#### Examples

> For a cluster analysis, nothing should be assigned to the **Target** role.
> For a Kaggle challenge the test/train partition is given in advance. Assign the 2-level factor to the **Partition** role.
> For a prediction model that should not favour males over females, the sex variable can be assigned to the **Sensitive** role.
> For a spatial model, the numeric position variables can be assigned to the **Latitude** and **Longitude** roles.
> For an observation-weighted model, the **Weighting** role needs to be assigned to a non-negative numeric variable.
