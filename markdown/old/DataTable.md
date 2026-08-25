This card enables all variables and observations to be examined. You are able to step through the pages, search for observations, filter individual variables, and set the viewing order of the observations.This card is best viewed in full-screen. There is a search facility to locate observations with matching data items.

#### Goal

The goal of this card is to locate unexpected values. The issues might be:

 - Numeric values like **-99** that might be place-holders for missing values.
 - Strings like **?** or **-** that might be place-holders for missing values.
 - Whether the observation identifiers show a progression. Are there gaps?
 - What the extreme values of each variable are. Are these outliers?
 - Whether variables have duplicate values.
 - Values whose numbers of decimal places are different to the norm.

The last issue requires the appropriate setting of the 'digits' parameter in the sidebar.

#### Actions
      
The downstream actions implied by the data-table issues are:

 * Revise the missing-value place-holders
 * Extract a predictor from the observation identifier
 * Drop the observations that are excessively missing
 * Detect outliers and investigate any explainable cause
 * Investigate changes in precision and investigate any explainable cause

#### Examples

> A variable is sorted and the lowest values are shown to be -999. These need to be converted to NA

> From pages 31 to 73 the value of a variable is constant at 70. An instrument fault might be the cause.

> High values of a variable have zero decimal places. Low values have three decimal places. Is there a change in units (mm to m)?

> The observation identifier seems to end in the observation year - extract the last four digits as a new variable.

