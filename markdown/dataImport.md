#### 🧭 Introduction

All analysis or modeling tasks begin with loading some raw data. The data is expected to be in a rows (observations) and columns (variables) table, or at least be able to be transformed into tabular data.

Data can be imported through any of three importation styles:

  1. User's local file system
  1. Dataset located within certain python packages
  1. URL resource

In this card, we assume that the data is addressed as a single file. The importation of files and URLs supports a variety of file types. These include, but are not limited to, the following file extensions:

- CSV: A comma, semi-colon, or tab-delimited file.
- XLS: A spreadsheet worksheet.
- RDS: An R serialization file containing a data frame.
- GPKG: A spatial data file.

If you have a different file extension, feel free to give it a try to see if it can be imported.

#### 🔄 Flip side

The back of the card documents:

- Dataset Info: Class and dimensions
- Memory usage: Bytes in use
- Variable listing: Column, Non-Null count, Null count, Data type
- Categories: Count, unique, top, frequency

#### ⚙️ Settings
Use the Tour Guide button to learn about the settings of this card. 

* The between-column separator character.
* The worksheet position (for multi-sheet spreadsheets). 

#### 🎯 Goals

1. Primary: Facilitate data importation
1. Secondary: Help verify whether you have the required dataset.

***
All cards downstream of this card will receive the data imported by this card. All variables are assigned to the __Predictor__ role. Dates and date-times that do not declare their time zones are assigned to UTC.

