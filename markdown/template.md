#### 🧭 Introduction


#### 🔄 Flip side

The back of the card documents:

 * Tabulates the count of each placeholder per variable.

#### ⚙️ Settings

Use the Tour Guide button to learn about the settings of this card.

 * Limit the maximum number of observations to show

#### 🧪 Example

If a column’s observed range is −1000 … 1200:  
-99 is inside the range → not flagged when `Replace extrema only` is ON.

If the range is −99 … 1200:  
-99 equals the minimum → is flagged when `Replace extrema only` is ON.  
(With `Replace extrema only` OFF, -99 is matched anywhere it appears.)

#### 🎯 Goals

 1. Primary: Help you verify whether candidate placeholders truly represent missingness (a domain-aware judgment).
 1. Secondary: Make it easy to fix the data. For each detected placeholder, the card provides a button to replace those values with an unambiguous missing value (NA/NaN/NaT). The chart and table refresh to verify the efficacy of the operation. 

***
All cards downstream of this card will receive any changes enacted by this card. At most, these changes will produce additional missing values. 
