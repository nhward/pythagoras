#### 🧭 Introduction

Having direct access to the data is a vital part of data analysis. The small-card version is a head/tail abbreviated table. The full screen version gives access to all the observations. It also allows observation filtering.

The raw data is minimally altered:

 - Convert to Pandas if possible
 - Geometries made displayable via _Well Known Text_ (WKT) format


#### 🔄 Flip side
There is no flip side. Going full-screen gives access to the entirety of the data.

#### ⚙️ Settings

Use the Tour Guide button to learn about the settings of this card.  

 * Bounding boxes (Geometries) 
 Converts each geometry to a bounding box representation.  
 * Rounding (Decimal)  
 Rounds decimal data, primarily to fit more columns into the available width.  
 * Limit the maximum number of observations to show (on the flip side)

#### 🎯 Goals

 1. Primary: Gives you direct access to the tables values. Columns are documented with their variable type since this is not always apparent.
 1. Secondary: Provides an opportunity to export the data as a CSV file. The file's name is based on the card's unique namespace.

***
All cards downstream of this card will receive the upstream data. This card does not change the data (only its appearance).  
