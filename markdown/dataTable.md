#### 📊 Introduction

Having direct access to the data is a vital part of data analysis. The small-card version is a head/tail abbreviated table. The full screen version gives access to all the observations. It also allows observation filtering.

There are a minimium of operations performed on the data:
 - Convert to Pandas if possible
 - Geometries made displayable via `Well Known Text` (WKT) format
 
#### ⚙️ Variations

##### 🟦 Bounding boxes (Geometries)  
Converts each geometry to a bounding box representation.  
Controlled via the card’s `Summarised as bounding boxes` setting.  

##### 🔢 Rounding (Decimal)  
Rounds decimal data, primarily to fit more columns into the available width.  
Controlled via the card’s `Decimal places to show` setting.  

#### 🎯 Goals

- ✅ Primary: Gives you direct access to the tables values. Columns are documented with their variable type since this is not always apparent.
- 🛠️ Secondary: Provides an opportunity to export the data as a CSV file. The file's name is based on the card's unique namespace.

All cards downstream of this card will receive the upstream version of the data. This card does not change the data (only its appearance).  
Use the Tour Guide button to learn about the settings of this card.  