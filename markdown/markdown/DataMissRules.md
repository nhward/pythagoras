Association analysis is often used to analyse shopping basket patterns.
Here, we are treating missing values in an observation as items in a
basket. This enables us to apply association rules analysis to this
problem.

The association rules analysis focuses solely on the missing values and does not 
consider the values of the non-missing data as explanatory.

#### 🛒 What Is Association Rules Analysis?

Association rules analysis is a data mining technique used to find
interesting relationships or patterns between items in large datasets —
especially transactional data.

A classic use case is in market basket analysis, where the goal is to
find rules like:

> If a customer buys bread and butter, they are likely to also buy jam.

These rules help businesses understand buying behaviour, recommend
products, or optimise store layouts.

##### 🔗 What Is an Association Rule?

An association rule has the form:

`IF [Item A and Item B] THEN [Item C]`
The bracketed terms are addressed as LHS and RHS. This is how they are labelled in the tabulated rules.

Or more formally:

`A, B → C`

This means: if items A and B are purchased together, then item C is
likely to be purchased as well.

##### 📊 Three Key Metrics: Support, Confidence, and Lift

To know whether a rule is meaningful, we measure it using **support**
and **confidence**. Lift is the ratio of these.

✅ 1. Support

Support tells us how frequently the items in the rule appear together in
the dataset.

Definition: Support of A → C = (Number of transactions containing both A
and C) / (Total number of transactions)

Example: *There are 1,000 transactions. • 100 transactions contain both
bread and jam.*

So: *Support(bread → jam) = 100 / 1000 = 0.10 (or 10%)*

Interpretation: *10% of all customers bought both bread and jam
together.*

✅ 2. Confidence

Confidence tells us how often the consequent (right-hand side) appears
when the antecedent (left-hand side) is present.

Definition: *Confidence of A → C = (Number of transactions containing
both A and C) / (Number of transactions containing A)*

Example:*200 transactions contain bread, 100 of those also contain
jam*

So: *Confidence(bread → jam) = 100 / 200 = 0.50 (or 50%) 

Interpretation: Among people who bought bread, 50% also bought jam*


✅ 3. Lift

Lift tells us how much more likely the right-hand side (consequent) of a rule is to occur when the left-hand side (antecedent) is present — compared to if the two were independent.

Lift answers this question:

>Is the rule actually useful, or is it just describing something that happens frequently anyway?

Definition: *Lift of A → C = Confidence(A → C) / Support(A → C)*

Example: *Let’s say in a grocery store dataset...*

 - 10% of transactions include bread
 - 5% include butter
 - 4% include both bread and butter

So:

 - Support(bread) = 0.10
 - Support(butter) = 0.05
 - Support(bread and butter) = 0.04
 - Confidence(bread → butter) = 0.04 / 0.10 = 0.40
 - Lift(bread → butter) = 0.40 / 0.05 = 8

Interpretation: *Customers who buy bread are 8 times more likely to also buy butter than if the two items were bought independently. That’s a strong association, and this rule is likely interesting and useful.*

***

#### Goal

The goal of this card is to uncover patterns that explain why certain variables are missing together. To do this we need to make a mapping between a basket of shopping and missing data.
An observation serves as a basket. The variables are the the goods on sale. If a value is missing then that item is not in our basket.
While that seems sensible, we shall do the opposite. We shall deem the item to be in our "missing basket" if it is missing. That way, if an observation is complete (i.e. nothing missing) then our basket is empty. We shall ignore all empty baskets from our calculations.

#### Actions

Below are some actions that follow from a better understanding of the
association rules of the missing values.

-   Gain understanding of the dominant patterns of missing values.
-   Objectively assess the strength of the patterns.
-   Hypothesise causes for these patterns.
-   Feed-back improvements to the data collection process.

#### Example

> When variable A & B are missing together it is a virtual certainty that variables C, D, E are also missing. 
> C,D are occasionally missing without A or B being missing.
> E is rarely missing without A or B being missing.
>
> This suggests that the cause of A & B being missing is the same cause of E being missing and probably of C & D.
