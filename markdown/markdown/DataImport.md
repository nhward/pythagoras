All analysis or modeling task begin with identifying and loading the raw data. In this document, we assume that the data is stored in a single file. The importation process supports a variety of file types, including, but not limited to, the following:

- RDS: An R serialization file containing a data frame.
- CSV: A comma, semi-colon, or tab-delimited file.
- XLS: A spreadsheet with one or more worksheets.
- SHP: A shapefile (actually a set of related files).
- GPKG: A spatial data file.

If you have a different file extension, feel free to give it a try to see if it can be imported.

After importation, depending on the file type, the data may require further modifications. For instance, data imported from CSV files may contain ordinal factors that are not represented properly.

When a dataset is imported, it is expected that each variable maintains a consistent data type. However, the presence of missing value placeholders can disrupt this consistency, resulting in a reduction of the data type to the lowest common denominator: character.

Dates and times that do not declare their time zones are assigned to UTC.

#### Goal

The primary goal of this document is to facilitate data importation and help identify various potential issues that can arise during the process. These issues can be addressed later in the analysis. Key issues to look for include:

- Whether variable names are present and clearly defined.
- Choosing appropriate data types (e.g., character vs. factors).
- Identifying missing value placeholders.
- Resolving ambiguous date formats.
- Handling time format and time zone effects[^1].
- Assessing the Coordinate Reference System (CRS) for geospatial data[^1].
- Addressing locale-specific formats for decimals and separators.
- Ensuring the correct order of ordinal factors.

####  Actions

The following actions can be addressed using this framework:

- Detecting a header row in a CSV file that provides variable names.
- Specifying particular placeholders for missing values.
- Allocating geometry variables (from shapefiles) to the **Geometry** role.
- Assigning other variables to the **Predictor** role.
- Specifying a special decimal character.[^2]
- Indicating a specific date format.[^1][^3]
- Designating a specific CRS.[^1][^4]
- Employing a specific numeric format.

###### Footnotes
[^1]: Work in progress.
[^2]: This assumes that a single decimal character can be applied uniformly to all values of all numeric variables. If multiple formats exist, this must be resolved before importing the data.
[^3]: This assumes that a single date format can be uniformly applied to all values of all date variables. If multiple formats exist, this must be managed prior to data import.
[^4]: This assumes that a single Coordinate Reference System (CRS) can be uniformly applied to all values of all geometry variables. If multiple CRSs exist, this must be managed prior to data import.
