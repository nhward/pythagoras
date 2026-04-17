#### 🧭 Introduction

This card analyses the environment in which the analysis is taking place. This can be useful in explaining differences in the reproduction of past results. 

 - The **Summary** tab provides a small set of high level property values.
 - The **URL** tab provides evidence of the server in which the software is running. A host name of "localhost" or "127.0.0.1" means the software is running locally.
 - The **Packages** tab provides a sorted list of packages that have been loaded, up to this point, along with their version numbers.
 - The **Folders** tab counts the files that are in significant folders

#### 🎯 Goals

- ✅ Primary: Document the package versions in use
- 🛠️ Secondary: Provide evidence about changes


ℹ️ If you refresh the card at a later time there may be additional packages loaded.
ℹ️ To replicate a past analysis, the package version numbers might need to match.
ℹ️ This card does not require a data set.
ℹ️ This card does not change a dataset. Any cards downstream of this card will receive the upstream version of the data.   

The flip side (back of the card) documents:

- Pythagoras modules
- Important Python packages 
- Number of logical CPU cores

#### 🛠️ Actions

Keep a copy of this information when a crucial analysis has been completed. This might explain imperfect reproduction in future.
If the python libraries can be restored to their original ones, the software infrastructure will be identical. Any remaining differences will be due to changes in the data and/or changes in the Pythagoras. 
