This dialogue allows the basic modification of variables. Variables can be renamed, removed and have their data types changed (within reason). Each change involves setting up the change and pressing the "Commit" button. The button bounces when it is ready and able to be used. The set of changes will appear in code as an extended "recipe." Mistakes can be reversed in bulk by pressing the "reset" button.

The changes fall into five main categories:

1. Remove variables - This provides the first opportunity to drop variables.
1. Convert variable types - Converting the data type (see below)
1. Rename variables - Changing the variable's names
1. Reorder variables - For ordered factors, this involves specifying the exact rank sequence
1. Expand variables - Pivoting a variable containing a list of elements into numerous additional binary variables

#### Goal

The goal of this dialogue is clear the list of errors from the table that is visible in full-screen mode. Doing so is to make the data suitable for data analysis. Unresolved warnings and info messages can be tolerated. A particular error may take a sequence of changes to resolve. For example to turn a character variable into an integer involves firstly converting to numeric and then subsequently to integer. 

The important change is data-type conversion. Variables can have the wrong data type due their form of storage. For example, CSV files do not distinguish between text and factors, or between text and dates. Sometimes, missing value place-holders (e.g. "missing") can turn a column of numbers into a column of numbers-and-text. In this situation the data type drops to the lowest common denominator which is "character". The allowed data-type conversions this card supports are:

 - From text/character variables to numerical
 - From numeric/decimal variables to integer
 - From text/character variables to factor
 - From nominal variables to ordinal (i.e. ordered factors.)
 - From text/character variables to date
 - From nominal variables to text/character
 - From integer variables to numeric/decimal
 - From ordinal variables to cyclic 
 
Data type conversions not listed above are not available (e.g. decimal to factor) since their use rarely, if ever,  leads to a more useful dataset.
It is not uncommon for raw data sets to have no ordinal variables, in that those that might exist are initially recorded as text/character variables. It is recommended that text variables of long length / high cardinality are not converted to factor variables. To be a good candidate for a factor variable, the cardinality (i.e. the number of distinct values) should be low. See the "cardinality threshold" slider control in the sidebar.

Since there is no *cyclic* data type native to R, we shall convert ordinal variables directly to their harmonic equivalents as a pair of decimal variables (i.e. sine and cosine components)  

#### Examples

> A string variable with a cardinality of 5 should be changed to a factor variable.

> A factor variable with factor levels: 'neutral','disappointed','pleased' can be changed to an ordinal variable ordered as: 'disappointed','neutral','pleased'.  

> A factor variable that holds the complete contents of various books should be changed to a text/character variable.

> An ordinal factor variable with levels 'Cheap', 'Normal', 'Expensive', 'No price shown' should be changed to factor since the levels have no definite ranking (due to the last level.)

> A text variable composed of comma delimited shopping items can be migrated to a set of binary variables - one for each possible item.
