#### 🧭 Introduction

A Missingness Heatmap is a visual tool for exploring tabular data. It highlights true missing values and also flags candidate missing-value placeholders (e.g. -99, --, N/A).
These candidates are configured in the sidebar and are typically data-type specific.
	
 - Placeholders = values used to stand in for “no data” when a value is required e.g. integers: -99, -999; floats: -99.0; strings: --, N/A; datetimes: 0001-01-01.

Since charts can become slow when given excessive amounts of data, the chart randomly samples a portion of the data for the charts. Any replacing of placeholders will be performed on the entirity of the data, regardless of the `Maximum Observations to chart` setting.

The flip side (back of the card) tabulates the count of each placeholder per variable.

#### ⚙️ Variations

##### 🔤 Case sensitivity (strings)   
Toggle between case-insensitive ("NA" matches "na", "Na", "nA") and case-sensitive matching.
Controlled via the card’s `Case Sensitive` setting.

##### 📉📈 Numeric “extrema only”   
When `Only replace extreme numeric values` is ON, a numeric placeholder (e.g. -99) is only flagged if it sits at the edge of the observed distribution for that variable (i.e., equals the current minimum or maximum).
When OFF, any occurrence of the placeholder is matched regardless of position.

ℹ️ “Edge of the range” means the value is the observed min or max for that column.

#### 🧪 Example

If a column’s observed range is −1000 … 1200:  
-99 is inside the range → not flagged when `Replace extrema only` is ON.

If the range is −99 … 1200:  
-99 equals the minimum → is flagged when `Replace extrema only` is ON.  
(With `Replace extrema only` OFF, -99 is matched anywhere it appears.)

#### 🎯 Goals

- ✅ Primary: Help you verify whether candidate placeholders truly represent missingness (a domain-aware judgment).
- 🛠️ Secondary: Make it easy to fix the data. For each detected placeholder, the card provides a button to replace those values with an unambiguous missing value (NA/NaN/NaT). The chart and table refresh to verify the efficacy of the operation. 

All cards downstream of this card will receive any changes enacted by this card.  
Use the Tour Guide button to learn about the settings of this card.