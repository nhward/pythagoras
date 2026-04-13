This card explores the diversity of observations of a dataset with respect to duplicates and near duplicates.

Near duplicates are observations that differ (i.e. mismatch) by a low number of variables. A near duplicate is not one whose mismatches are close, rather, it relates to the number of mismatches regardless of their magnitude.
A true duplicate is an observation that is an exact match to another observation. Its differences are zero. In contrast, a near duplicate may be
an exact match to 8 out of 9 variables and be recorded against the "1 difference" tally.

For this assessment all variables are examined **except** those with _no role_, _geometry role_, _weighting role_, and the _identifier role_ (roles = None, Geometry, Weighting, Identifier).

The advanced settings allow the decimal data to be compared at a lower accuracy. This enables close numeric values to be exact matches. Integer data is not affected.
The process of rounding is performed to a fixed number of significant places, rather than decimal places, due to different scales that variables can have. 
This has the effect of increasing the tally of each of the chart's bars.

#### Goal

The goal of this card is to visualise the shape of the chart of the tally of near duplicates v differences as a signature of the observation diversity. 
High dimensional data is likely to be extremely diverse and show no bars in the chart. Any near-duplicates showing with high dimensional data is a cause for concern and suggests many of the dimensions are redundant.

The potential issues this card addresses are:

 - What is the tally of (near) duplicates?
 - What is the shape of the near-duplicate observations?
 - What are the row numbers of the (near) duplicate observations?

#### Actions

The downstream actions implied by the summary issues are:

 * Drop duplicate observations.
 * Employ dimensional reduction.
 * Employ pivoting to change the width and height of the dataset.
 * Add an weighting column to the dataset to deal with observation duplication.

#### Examples

> A dataset with 30% duplicated observations should be cleaned up.

> A dataset with some necessary duplicated observations can be redesigned as a dataset with observation weighting.

> A dataset with 100 variables is showing near duplicates at the 3 & 4 differences level. This is curious and merits investigation.
