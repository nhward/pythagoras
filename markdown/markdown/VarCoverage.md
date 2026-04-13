This card dialogue examines the coverage of the target with respect to the predictors of the dataset.

#### Goal

The goal of this card is to judge whether the data is suitable for data analysis. The main topic here is called "coverage". Coverage relates to how well the data is represented in observations that are "intersectional". An example will clarify the meaning of these terms:

> Suppose some 10,000 observations of people, some of whom have disabilities. The data is fairly equally represented, 5017:4983, between males and females. The data might show there are 24 males who are wheelchair bound, but only 3 females. If you have reason to expect that these numbers should be the same, this is an anomaly to be investigated. On the other hand, if you expect differences you can check that they are in the direction you expect. 
Suppose we further analyse the data by occupations. If we discover there are no observations in the data set for the intersection female/disabled/military then one of two possibilities exist:
> 
> 1. The subsample is intrinsically empty i.e. mutually exclusive.
> 1. The dataset is not yet large enough to include members of this subsample.

#### Independence

If the various categorical predictors are independent of each other, the proportions of their intersections should be calculable as the product of their respective proportions. When this calculation does not match reality, it suggests that the variables are not independent.

The potential issues this card addresses include the following. 

 1. Are there intersections of categories whose proportions are over represented (assuming independence)? 
 1. Are there intersections of categories whose proportions are under represented (assuming independence)? 
 1. Are there intersections of categories that are completely unobserved (as an extreme form of under representation?)
 
 This applies to the following roles (which are intrinsically categorical):
 - the **Sensitive** role.
 - the **Stratifier** role.
 - the **Treatment** role.
 - the **Partition** role.
 
We focus specifically on these roles as these are the ones for which we are likely to have clear expectations of independence.

#### Actions

The actions implied by the summary issues are:

 * Employ observation weighting to restore proportions to expectations.
 * Try _fairness_ techniques in the modeling (e.g. different classification thresholds for Males & Females)
 * Postpone analysis while more data is collected.
 * Investigate the anomalous proportions to better understand the data.

#### Examples

> In a patient dataset, one hospital is over represented in the presentation of a rare type of cancer?  Having identified the issue let's investigate the cause. Is the imbalance locational or diagnostic?

> A dataset is used to create a classification model. One subset of observations is suffiently rare (and classifications sufficiently inaccurate) that this subset is excluded from being predicted.
