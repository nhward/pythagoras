#### 🧭 Introduction

The purpose of the **Role Assignment card**  is to make the dataset explicitly interpretable by defining how each variable is intended to be used.

The card is a visual drag-and-drop interface used to define and validate the statistical role of each variable in the analysis. Move each variable name to the appropriate role slot.

Subsequent analysis should begin again with a fresh role assignment as the assignments shape the analysis.

> _"Every model makes assumptions. This card makes them explicit."_

The following roles are supported:

 - **Target** — the variable being predicted (the outcome)
 - **Predictor** — variables used to predict the target
 - **Treatment** — variables with a potential causal effect on the target (distinct from purely predictive variables)
 - **Identifier** — variables that uniquely identify observations (optionally combined with Sequence)
 - **Weighting** — a numeric variable representing observation importance (case weights)
 - **Partition** — a categorical variable defining train/test splits
 - **Stratifier** — a low-cardinality grouping variable used for segmentation (e.g. faceting), but not intended to be predictive
 - **Sequence** — an ordered variable (typically time) defining observation order
 - **Sensitive** — a variable used for fairness auditing; should not drive predictions
 - **Geometry** — spatial data (points, lines, polygons) for mapping and spatial analysis
 - **None** — variables excluded from downstream analysis

An easy to read introduction to variable roles is available [here](https://medium.com/@nhward60/a-murder-mystery-for-your-dataset-assigning-roles-to-variables-593983a43214).

We operate under the following presumptions:

 * Each variable is assigned to, at most, one role.
 * Since we have a **None** role, all variables must be assigned to, at least, one role.
 * Although multiple spatial variables are theoretically possible, there are limitations in the handling of spatial data so only one **Geometry** variable is allowed.
 * Although multiple dependent variables are theoretically possible, most modeling techniques do not support this, so only one **Target** variable is allowed.
 * When multiple variables are assigned to the **Identifier** role their contents are merged using a separator character.

#### 🔒 Constraints

Role Requirements

 - Only the **Predictor** role is required to be filled
 - On initial load, Spatial imports automatically assign geometry variables to **Geometry**. All other variables are assigned to **Predictor**.

Role cardinality:
 
 - **Predictor**, **Identifier**, and **None** roles → may belong to multiple variables
 - All other roles → at most one variable

Additional constraints (validated automatically):
 
 - data type
 - missing values
 - cardinality

Violations are indicated via validation messages.

#### 🧪 Example Use-Cases

 - Regression → assign a numeric variable to Target
 - Classification → assign a categorical variable to Target
 - Clustering → leave Target unassigned
 - Predefined train/test split → assign to Partition e.g. [Kaggle](HTTPS://www.kaggle.com)
 - Fairness analysis → assign variables to Sensitive e.g. "Sex"
 - Spatial modeling → assign spatial variables to Geometry
 - Weighted models → assign a non-negative numeric variable to Weighting
 - Time series → assign a time variable to Sequence
 - Longitudinal analysis → assign Sequence + Identifier

#### 🔄 Flip side

The back of the card documents:
 
 * the most recently committed assignments

#### ⚙️ Settings

Use the Tour Guide button to learn about the settings of this card.  

 * Separator character when combining multiple variables into a unique identifier. 
 * Limit the maximum number of observations to analyze for assessing cardinality and missing values.

#### 🎯 Goals

 1. Primary: Identify which single roles should be assigned to each variable.
 1. Secondary: Validate that each variable's data-type, cardinality, and missing values are appropriate for the assigned role.


#### 🛠️ Actions

 - If a variable does not fit any role, assign it to **None**
 - Understanding roles helps anticipate valid downstream analyses
 - Role constraints guide sensible assignments (e.g. singleton roles)
 - If no Predictors are defined, most analyses will not be meaningful
 - If a variable needs multiple roles, duplicate it and assign each copy separately

 ***
 All cards downstream of this card will receive any changes enacted by this card. These changes will only relate to variable roles. 