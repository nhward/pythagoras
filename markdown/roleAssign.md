#### 🧭 Introduction

The **Role Assignment card** is a visual drag-and-drop interface used to define and validate the statistical role of each variable in the analysis. 

The reverse side of the card displays the most recently committed assignments. 

An accessible introduction to variable roles is available [here](https://medium.com/@nhward60/a-murder-mystery-for-your-dataset-assigning-roles-to-variables-593983a43214).

Subsequent analysis should begin again with a fresh role assignment as the assignments shape the analysis.

> _"Every model makes assumptions. This card makes them explicit."_

#### 🎯 Goal

The purpose of this card is to make the dataset explicitly interpretable by defining how each variable is intended to be used.

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


#### 🧠 Presumptions

We operate under the following presumptions:

 * Each variable is assigned to, at most, one role.
 * Since we have a **None** role, all variables must be assigned to, at least, one role.
 * Although multiple spatial variables are theoretically possible, there are limitations in the handling of spatial data so only one **Geometry** variable is allowed.
 * Although multiple dependent variables are theoretically possible, most modelling techniques do not support this, so only one **Target** variable is allowed.
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

#### 🛠️ Actions

 - If a variable does not fit any role, assign it to **None**
 - Understanding roles helps anticipate valid downstream analyses
 - Role constraints guide sensible assignments (e.g. singleton roles)
 - If no Predictors are defined, most analyses will not be meaningful
 - If a variable needs multiple roles, duplicate it and assign each copy separately

#### 🧪 Example Use-Cases

 - Regression → assign a numeric variable to Target
 - Classification → assign a categorical variable to Target
 - Clustering → leave Target unassigned
 - Predefined train/test split (e.g. Kaggle) → assign to Partition
 - Fairness analysis → assign variables (e.g. sex) to Sensitive
 - Spatial modelling → assign spatial variables to Geometry
 - Weighted models → assign a non-negative numeric variable to Weighting
 - Time series → assign a time variable to Sequence
 - Longitudinal analysis → assign Sequence + Identifier