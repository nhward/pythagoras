The monitor card uses the _reactlog_ package to record and then visualise the sequence of events that trigger during a session.
_reactlog_ is short for Reactivity Log which is a file of events and timestamps that are generated and stored on the server.

Monitor acts as a debugger for your Shiny app's reactive behaviour. 
It creates a dependency graph to reveal how user inputs and reactive expressions interact. 
This helps a developer identify overly sensitive reactive elements that trigger too often and slow down your app. 
By analysing the graph, a developer can optimise the code and ensure a smooth user experience.

#### Status bar

<img alt = "Status Bar" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f11ec974f29fd9539a_AD_4nXcOZLsi_rcpUbP8xmmIprFPcXgIDOH_cRmWW3Y9OLt-DXYiRmT7h0KFKEQ7npLelq6ykhKUKzGnYnKGJE0sPpBGO7fQYYLJ8UxsosoNUB0YbxuD0oTnwH7AqoWg5wp8WUYK7XKpmq3p8QfVokdy0XouFcva.png" 
style="width:100%;">

#### Progress Bar: Visualising Reactive Flow

<img alt = "Progress Bar" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f1c9e74189d53b4d2e_AD_4nXdKAa6Vh45auT_VEa-XvD0EbaPA1jXT9BxyH8tPqniFHjNlKs0L-WwBbhnX9UX6H_bDdv0zIT8XBScbPNM_hEuEPTGn7U1fY3_0WkH6yk6Db-jCY1V2WPjwfpB0QDV-aIyHFKC6a5jSZKl8FM48DP61ssI.png" 
style="width:100%;">

- *Current Location (Dark Gray Rectangle)*: This marker indicates your app's current position within the reactive execution sequence.  
- *Reactive Endpoint Marks (Short Green Lines)*: These lines highlight points where reactive computations have reached completion (e.g., outputs or observations).  
- *Idle Periods (Tall Green Lines)*: These sections represent periods where your app is dormant, waiting for something to trigger a change in reactive values.  
- *User-Defined Marks (Tall Gray Lines)*: These markers allow you to designate specific points of interest within the reactive flow for future reference.  

#### Current Step Details:

<img alt = "Current step" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f14274b3e4717f569f_AD_4nXeqi-dcO6UzpS9PO3Ij3XwcIyRPYUdeExlfBqJ2EFQ0QGOJiCO6loi9CNMEOyPoxvBIG2rGhfnONwBzegqQrEqyo-tSokd-vpITFeeL7ykXis7-tSVHRgGsa8_03UwQwPVkrFBnHDSSRct_EltxPI60Cv0.png" 
style="width:100%;">

The current step section provides in-depth information about the specific reactive step you're currently viewing:

- *Step Description*: This section clarifies the current stage within the reactive life cycle, helping you understand what computations are currently being executed.  
- *Time Information*: This displays the elapsed time since the very first recorded step, offering a sense of the overall execution timeline.  
- *Session ID*: For multi-user Shiny apps, the session ID identifies the specific user session associated with the current step.  

#### Search Bar:

<img alt = "Current step" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f2885888dd0cbdae00_AD_4nXf5Zw9_-ftiqFQN6wKn3X5Z6_pDeUpGAx5hq02MNZByK59e_NBk7noaR1a2euXz_h_O90W9MdGDRwydCpZPtXapNDWG_Ujlsvojp86OQ8ma491F8HgVR3IE0d5av73rowXIF8pJyoTpMrUG_rVFDt6xMq0.png" 
style="width:40%;">

The search bar helps you navigate potentially complex reactive graphs with ease:

- *Search by Name or ID*: By entering at least three characters, the search bar highlights all matching elements within the graph, along with their associated family trees (related computations). This allows you to quickly locate specific reactive expressions or outputs of interest.  
- *Partial Matching*: You can leverage partial matching for module names. For example, searching for "input" might highlight elements within the "details" module, helping you narrow down your search based on specific functionalities.  

#### Navigation Buttons: Exploring the Reactive Timeline

<img alt = "Navigation buttons" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f209a7851abec24882_AD_4nXfZu5O8qxxe8sBmfw3cPeigwcw3Wsr9Di3uZ6oUOJ8hsO4JNzye5WUHgjsoEzl04OP3BWCFOKjQ-HIs9ZbcQRM6H_ZHFvW_UcfrQVpm1UJ0WojsrOiiInkprbm0HMI_ljLSQ9dcoMsTEs7rlAhNmN35fpSq.png" 
style="width:50%;">

The _reactlog_ status bar provides a set of navigation buttons that allow you to move forward and backward through the timeline of your app's reactive computations. 
These buttons offer different levels of granularity:

- *One-step navigation (arrow keys)*: Move a single step forward or backward within the displayed graph.  
- *Reactive endpoint jumps*: Jump directly to the next or previous point where a reactive endpoint calculation finishes.  
- *Idle step navigation*: Move to the next point where your app became idle.  
- *User-defined mark jumps (Home/End)*: Navigate to the nearest user-defined mark placed within your app.  

#### Dependency Graph

Dependency graph can be complex for simple apps and simple for complex apps. It provides a useful high-level view of your app reactivity and can provide good insight on fixing the reactivity of your app. This helps in avoiding unnecessary reactive and observe calls.

<img alt = "Dependency graph" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f25048d29b7cddbe92_AD_4nXc4BYYcr_1Db361Ie7P8LCogGq30FJUcGhBgFFtBW5gxfnNItTFNXZMAuW_ETnFZUpVzK_wyTw2fiVl6ODjjlT8Y4JlFZbVeu_U4So8ldwgf4JOUsS9-txm8efunGqmRQTDh3Am4eU2dp8AwIYZdlpd4U3K.png" 
style="width:100%;">

### Analysis Goals

 * _Find Chatty Reactives_: Look for nodes with many outgoing arrows (frequently re-evaluated expressions).
 * _Analyse Dependencies_: Check for long chains (potential bottlenecks for calculations).
 * _Focus on Suspicious Areas_: Use filtering and highlighting to zoom in on problem areas.
 * _Code Optimisation_: By focusing on path with most time taken, you can prioritize optimisation efforts by re-factoring code, optimizing state management, etc
 * _Jump to Profiling next_: Combine with code profiling for detailed execution times.
