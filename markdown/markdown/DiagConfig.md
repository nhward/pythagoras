#### Configuration

The configuration to be examined is the configuration of the computer where the bulk of the processing takes place.
This is typically some cloud server. Your own computer will only be running a browser with which to display the web pages.

Each tab reports upon a different aspect of configuration

#### Platform
This includes *R version*, *operating system*, *architecture*, *user interface*, *language*, *character set collation & type*, *time zone*, *date*
*RStudio version*, *Pandoc version*, *Quarto version*

Note: that Windows is not case sensitive with respect to file names, whereas Linux and MacOS are. 
There are also differences between threading approaches between the operating systems.

 * Windows uses multisession
 * Direct hosting in RStudio uses multisession
 * All other configurations use multicore - which has the best parallel performance.
Generally speaking Linux is a good server operating system for hosting R software.

#### Packages
This table documents the packages and their characteristics. In particular is the **Source** column which distinguished packages that are authorised CRAN ones and those that are loaded from *Bioconductor* or *Github*
There are facilities for searching, sorting and paging through the many packages.

#### External
This listing recods more obscure aspects of the software configuration. 
In particular the _lapack_ and _BLAS_ entries dictate the efficiency of matrix calculations.

#### Settings
The settings listing records the values of R options that relate to the operation of Shiny. 
The settings are not able to be changed here or elsewhere. 
**DevMode** should not be found to be set during "production" use.
