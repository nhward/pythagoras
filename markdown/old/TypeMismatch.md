#### Levels of measurement

Levels of measurement, also called scales of measurement, tell you how precisely variables are recorded. In data science, a variable is anything that can take on different values across different observations in your data set (e.g., height or test scores).

Traditionally there are 4 levels of measurement:

1.  Nominal: the data can only be categorised
2.  Ordinal: the data can be categorised and ranked
3.  Interval: the data can be categorised, ranked, and distance can be calculated
4.  Ratio: the data can be categorised, ranked, distance can be calculated, and has a natural/well-defined zero.

Depending on the level of measurement of the variable, what you can do to analyse your data may be limited. For example, it is unwise to use distributional transforms (e.g. log transforms) on data that does not have a well defined zero.

This view of levels of measurement is somewhat insufficient for our purposes as it fails to consider a great deal of real-life data that cannot be allocated to any of these four levels of measurement.

#### Raw Data Types

This card is a visualises mismatches of variable data types and provides an automated means of fixing many of these. The raw data types considered are:

| Type | Description | Fix | Example |
|---------------|---------------------------|---------------|---------------|
| Decimal | Numbers that derive from a measuring process | Auto | 12.5 km/h |
| Integer | Whole numbers that derive from a counting process | Auto | 2 bedrooms |
| Nominal | Words that describe the state of something | Depends on cardinality[^typemismatch-1] | green |
| Ordinal | Words that rank something | Manual | XXL |
| Text | Sets of many words that rarely repeat | Depends on cardinality[^typemismatch-2] | "Able was I ere I saw Elba" |
| Date/time | Ways of describing points in time (Date/POSIXdt/POSIXlt) | Auto | 15/Sept/2013 09:45 UTC |
| Geometry | Ways of describing points, lines and polygons | Depends on file format | point: lat=56.8776262, long=34.1426748 |
| Logical | Two-state nominal | Auto | True/False |
| Cyclic | Periodic nominal | Manual | days of the week |
| Other | Anything else | Manual |  |

The table above records some data-type fixes as being "manual". This means the card dialogue being discussed here will not be able to fix these variables automatically. A manual assessment must be performed in an separate card dialogue.

After using the auto-fix button, any data issues (that appear in red) that still remain in the chart will be those whose resolution will require manual assessment elsewhere.

#### Why does data type matter?

To answer this we need to consider what most modelling methods demand of predictors - that they are all encoded as numeric [^typemismatch-3].

We recognise the fundamental differences between **nominal**, **ordinal** and **text**. Ultimately these must be converted into numbers, but (and this is the key point) the process of doing so is quite different for each.

-   Ordinal variables can be decomposed into a set of decimal values using orthogonal polynomials via dummy encoding.
-   Low-cardinality nominal variables can be decomposed into a set of binary variables.
-   Medium-cardinality nominal variables can be "mean-encoded."
-   High-cardinality nominal variables can be "frequency-encoded."
-   Text variables can be vector encoded/embedded.
-   Text variables can be decomposed into sentiment, topics words (see nominal above), word frequencies (See Integers above), and other relevant NLP techniques
-   The special case of **cyclic** data requires a harmonic encoding employing both sine and cosine components to fully capture the periodicity.

It is also possible to lower the cardinality of nominal variables by aggregating low frequency levels into a single "other" category.

In a similar way, we need to recognise the fundamental differences between **decimal** numbers and **integers**. The former often has a symmetrical Gaussian distribution whilst the latter often has a skewed Poisson distribution.

-   Decimal variables which result from a measurement process (potentially with units attached)
-   Integer variables which result from a counting process

This distinction may be important to a downstream process that will make assumptions about their population distributions. An example of this is Gaussian Processes regression. Another approach is to explicitly employ poisson regression where the variables demand this.

Dates and Times are complex variables that can be encoded in interesting ways. It is common to encode these into some decimal representation of time since some (artificial) origin. We can take this as obvious. It is also possible to encode these into:

-   Year - (whose calendar?) - integer
-   Month - Cyclic
-   Business quarter - Cyclic
-   Day (of month or year) - Cyclic
-   Season (bear in mind we do not all live in the northern hemisphere) - Cyclic
-   Hour (12 or 24 hour) - Cyclic
-   Minute - Cyclic
-   Second - Cyclic
-   Time of day (morning/afternoon/evening) - Cyclic
-   Local date/time or universal date/time

Returning to the original question "Why does identifying the raw data type matter?"; we can reply that assigning the correct raw data type will make their downstream use more structured and guide us to use the correct encoding for each data type.

Bear in mind that normalisation (i.e. centering and scaling) that might occur downstream will transform an integer variable into decimal ones. This will disguise some of the data-type evidence we are investigating here.

#### Goal

The goal of this card is to discover data type mismatches and visualise these in a sankey chart. In this context a mismatch is when the observed raw data type does not agree with the expected data type. The red connectors are mismatches. The green ones are matches. Any mismatches that point to *Other* are for manual resolution. Many mismatches can be fixed by using the "Fix" button. A second press will remove the fix.

#### Actions

The actions suggested by this card are:

-   Fixing the easy mismatches using the button.
-   Identifying variables that need manual resolution.
-   Manually fixing the mismatches downstream for things like *Ordinal* and *Cyclic*

#### Examples

Some typical examples of mismatches are:

-   A *text* variable contains digits and all observations of this variable can be translated into valid numbers. The expected data type is *decimal* or *integer.*

-   A *text* variable with low cardinality. The expected data type is *nominal* or *Ordinal.*

-   A *factor* variable has a high cardinality (say 150). It is impractical to use dummy encoding on this variable. The expected data type is *text*.

-   A *decimal* variable only contains whole numbers. The expected data type is *integer.*

-   A *logical* variable that only contains two states. The expected data type is *integer*.


##### Footnotes

[^typemismatch-1]: Cardinality refers to the number of distinct levels a nominal variable has.
[^typemismatch-2]: Cardinality refers to the number of distinct levels a nominal variable has.
[^typemismatch-3]: A notable exception to this are decision trees which natively work with nominal variables.
