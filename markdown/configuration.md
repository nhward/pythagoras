#### 🧭 Introduction

This card analyses the environment in which the analysis is taking place. This can be useful in explaining differences in the reproduction of past results. If you refresh the card at a later time there may be additional packages loaded.

 - The **Summary** tab provides a small set of high level property values.
 - The **URL** tab provides evidence of the server in which the software is running. A host name of "localhost" or "127.0.0.1" means the software is running locally.
 - The **Packages** tab provides a sorted list of packages that have been loaded, up to this point, along with their version numbers.
 - The **Folders** tab counts the files that are in significant folders


#### 🔄 Flip side
The back of the card documents:

- Pythagoras modules
- Important Python packages 
- Number of logical CPU cores

#### ⚙️ Settings
There are no settings.

#### 🎯 Goals

 1. Primary: Document the package versions in use
 2. Secondary: Provide evidence about changes


#### 🛠️ Actions
Keep a copy of this information when a crucial analysis has been completed. This might explain imperfect reproduction in future.

If the python libraries can be restored to their original ones, the package infrastructure will be identical. Any remaining differences will be due to changes in the data and/or changes in the Pythagoras. 

The in-use package list can change during an analysis session as additional "cards" are loaded progressively. 
***
This card does not require a data set. This card does not change any dataset. Any cards downstream of this card will receive the upstream version of the data.