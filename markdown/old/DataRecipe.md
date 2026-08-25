This card is a summary of a recipe for preparing a dataset. A recipe can both learn from the data and also change that data and any future data. A recipe is a sequence of steps. Each step can change the data context for the step that follows. Recipes are designed to be used repeatedly within a resampling strategy.

#### Goal

The goal of this card is to explore what steps the recipe has accumulated up to this point in its specification. A recipe manages the steps that are undertaken, the order of these steps and the training of these steps. Recipe steps are also used to assign variables to roles.

These roles are:

-   **Target** - the variable that is being predicted (also known as the **outcome** role)
-   **Predictor** - variables used to make the predictions (which is also true of **Treatment**)
-   **Identifier** - one or more variables to uniquely identify each observation
-   **Weighting** - an importance or repeat count of an observation (also known as **case_weights**)
-   **Partition** - a factor that divides observations into test or train splits
-   **Stratifier** - a factor that groups similar observations together but is not meaningfully predictive
-   **Sequence** - variables (usually data/time) that give an order to the observations but are not directly predictive themselves
-   **Sensitive** - factor variables that record features that should not be predictive and can be reviewed for fairness
-   **Treatment** - a factor variable that may have a *causal* influence over the **Target**
-   **Geometry** - spatial variables that record points, lines, and polygons (or collections thereof)

Only the **Predictor** role needs to be filled. Not every variable needs to be assigned to a role. Unassigned variables are not used. Currently a variable cannot not have more than one role. If this proves to be problematic, create a duplicate variable which can be assigned an different role. When tabular data is first loaded, everything is a predictor. The process of importing spatial data (shape files) will automatically assign the special geometry variables to the **Geometry** role.

#### Actions

The actions suggested by this card are:

-   Adding steps that, while not required by the current data, may be required for unseen future data
-   Assigning roles correctly, so that relevant recipe steps act upon the correct variables
-   Re-assigning variable data types, so that relevant recipe are applicable to the correct variables.
-   Checking the order of recipe steps.

#### Example

Sometimes, historic data can be missing-value free (through over-zealous data cleaning) and therefore not require missing values to be "fixed". Adding a imputation recipe step to handle missing values means that future observations with missing values will not need to be rejected.
