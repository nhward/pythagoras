---
output:
  html:
    meta:
      css: ["@tabsets"]
      js:  ["@tabsets"]
---

Profiling is a developer tool that provides information about the code hotspots. A hotspot is a line of code that consumes a disproportionate amount of the computer's processing ability. 
The profiler display is in the form of an interactive graphic that expands to show more detail. 
The hotspot code lines are ranked and the worst lines are displayed initially.

#### 🎯 Goal

The goal of a profiling is to :

 - Detect what the hotspots are.
 - Decide, for each hotspot, whether the line is something that can be improved or not.

## Two styles {.tabset .tabset-fade .tabset-pills}
The profiler comes in several flavours: 

### R-Profiling

There is an R-based tool that uses the _profvis_ package. _Profvis_ is an R package specially designed for performance profiling in Shiny apps. It acts like a code execution profiler. _Profvis_ captures how long different parts of your code take to run using a **flame graph**. A developer can optimize the code and make your Shiny app run faster.

#### Flame graph

 - *Horizontal Axis:* The horizontal axis of the flame graph represents the passage of time, typically from left to right. The width of the graph is determined by the total time spent executing the code.
 <img alt="Horizontal axis" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f349695fde87bcc104_AD_4nXcITopw38oDI_v5js-o2ucGX8wM9d5uXcHjN3OPRF29AWqrgNxyZqzaPgr5aZt0o6SsiUGrgKgQyGzPldUnSPbEI0x6BvNclVwZte8pEjcsHMB2EI3A5tdUvbiv0FDKaEUp3rgXLtUU8NT-VU6JYPND59mA.png" style = "width:100%">
 
 - *Vertical Bars:* Each vertical bar in the flame graph represents a function call stack in your code. The height of the bar indicates the depth of the call stack, with the top-most bar representing the outermost function call.
 <center>
 <img alt="Vertical bar" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f27e2bfd273b70db99_AD_4nXfCffLt6l5Q6Kf55QV536LhnBQMdffxyoSYb178L1D3buhf0DTQ3Mnlc2DNoUARqRs1Jr-bdhj2gkOtDLaZe7WEWw9IXzM7H1hk8U7LMTW93OZDflA4sas0wWmBwIzLbOEfKYIuhdUfWER5rKMiwGPaRtHV.png" style="width:15%">
 </center>
 
 - *Colors:* Different colors are used to distinguish between different functions or code blocks. Colors often depict different code categories (e.g., user code, Shiny functions, R base functions).
 - *Width of Bars:* The width of each bar corresponds to the amount of time spent executing the function. Wider bars indicate that more time was spent in that particular function.
 <img alt="Width of bars " src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f155e77902e8dae345_AD_4nXeU4_hdCyE5KGFcvR1pau_RVT3g0XmDQ-_WY0a-tigu-8c5uf4hGeD9vKkF_y2UhwqkMUrsgxpPkQ45hN7Mthw1bebVBW2Ycl5avd0jXwEFZZBjb9OiVcOGO_IS8xv6uC6eybSojtt8WPtQMFA-4Bj25Ok.png" style="width:100%">

#### Code viewer

Just above the flame graph you will see code viewer. Code viewer also provides insights like memory consumption and time taken at line level.

<img alt="Code viewer" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f15ef262350ed1f1e6_AD_4nXdkYMKAzhLW1pcox4DTlSzHlzhDVmOzjZW_j_vojosdqA-hb_qlD3pT9jEt_q57INqsGe2SuxPgDvvaVTV9Zr4L5LrDfyOWym0BPhd4K48jhE4b9dRg-2aG7I85mkXkrRpvCGvYAuWSZXhbyQ72iUOyyeJk.png" style="width:100%">

 - *Clicking on Bars:* Clicking on a bar representing a function call often reveals additional information. This might include the function name, self time (time spent within the function), and total time (including time spent in functions it calls).
 <center>
 <img alt="Click on bar" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f12e9d53ec133f5173_AD_4nXfJgJ_QGptJ9aiNQyRLe-FjkU78bUpUKcEGuS2VuyseiPxYmnVN2QSlcgTnVcpUYlgq29PbIGfXvCrSLJ50BFe7g5z63CB9drJJKbMqk92P6TmRoP5LxiiQR0UbNuROlwuD_fc8w_LakatB5yak40JNLNk.png" style="width:30%">
 </center>
 
 - *Clicking on Cells:* Single click on cell will move you to the specific function, double click on cell will redirect you to code editor in a separate window, additionally it will expand the bar to provide you a more clear view.
 - *Data Viewer:* In addition to the flame graph, _profvis_ also offers a data view accessible by clicking the Data tab. Here, you'll find a hierarchical table showing the profile in a top-down manner. Click on the code column to expand the call stack, and check the Memory and Time columns for resource allocation insights.
<img alt="Data viewer" src="https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f2c1fb3a4e81a73918_AD_4nXfKWH4TeuZjLdvcrv_xnrdeaacAS-cYDAgVAx2roxtd3c4Pw47PmNg8e53e39P5ooG-H6K-7xpQhegqyybIv-Ex2yh_2L_cQFrs4S-xdit4zQA363SkIzQcMVJEjRRNN7MKsbUS1HNMOMVRRoOlbXl0EXc.png" style="width:100%">

#### Analyzing the Flame Graph

 - *Identifying Hotspots:* Look for sections with the widest bars. These represent functions that consume the most execution time and are potential bottlenecks.
 - *Understanding Call Stack:* Trace the path downwards from the top to identify the sequence of function calls leading to the bottleneck.
 - *Zooming In:* Use interactive features to zoom in on specific sections for a closer look at nested function calls.
 - *Comparing Runs:* You can compare flame graphs from different app versions to identify performance improvements or regressions.

#### Interpreting the Information:

 - *Time Spent in Functions:* The width of a bar at a specific level reveals the time spent executing that particular function.
 - *Self Time vs. Total Time:* Some profvis implementations differentiate between "self time" (time spent within the function itself) and "total time" (including time spent in functions it calls). This helps identify functions that contribute significantly to bottlenecks through nested calls.
 - *Code Optimization:* By focusing on functions with the widest bars, you can prioritize optimization efforts by re-factoring code, leveraging vectorization, or exploring alternative algorithms.

### Web-Profiling

There is a java script tool called _shiny.tictoc.js_ produced by Appsilon.
_shiny.tictoc_ helps you measure the performance of your app. 
It provides a simple visualization that helps you measure how much time it takes to recalculate each output of your app and how long do server side computations take.
If this card is included in Ziggarto, the _shiny.tictoc.js_ will be loaded and will introduce timing events throughout the session. 
This can affect performance. The **Refresh** button will update the visualisation with the latest cumulative timings.

_shiny.tictoc_ outputs looks like this:

<img alt = "output looks like this" src=https://cdn.prod.website-files.com/654fd3ad88635290d9845b9e/66c4c6f1f5db54c2b36704cb_AD_4nXcWs0k7S8Pq2LeEDxSr5xSwzJI2rzVE5Y5rQWvFwdelQn9I2prz7_cdaDSBepsrmPfNErCTLAgbJmdLxxwpI5Is19XZoSxxkvd4BoHHXKzvZWiOsGuN_y1Jxkn9x1WYkCXaxxcu_G4JCn6nzxzBjTi5d5_1.png style="width:100%;">

 - *X-axis* shows timeline.
 - *Red* is for outputs (it looks at the shiny:recalculating and shiny:value events)
 - *Green* is for custom message handlers (looks at the shiny:message event)
 - *Blue* is for general server side calculations (shiny:busy and shiny:idle events)

##

#### Causes of bottlenecks are:

 - A large task is performed by a single line of code. There is no inefficiency here, just a large task. 
 - A frequently invoked line of code can be why it appears to use so much compute resource.
 - A section of code may perform well for small datasets but its performance degrades significantly with large datasets. This degradation could be related to the number of observations (rows) or to the number of variables (columns) or both. This applies to statistical analyses and to data-science algorithms.
 - A chart may be trying to display too much information, especially if each point has a hover reaction built into the chart. 
 - Combinations of things (e.g. combinations of variables) rapidly produce huge tasks by virtue of the rate at which combinations accumulate. It is best to avoid these scenarios.

#### 🛠 Actions

##### Developer

 - Rewrite the slow code to be faster - improve its efficiency.
 - Parallel-code is code that tries to use more than one CPU. There are many architectures for doing this. Some operating systems do not support certain architectures and the code must revert to a slow sequential style of execution. Sometimes, a parallel architecture is so poorly implemented that it runs slower than sequential code.
 
##### User

 - Avoid long data sets. Keep the number of observations low by down-sampling the data.
 - Avoid wide data sets. Remove uninformative variables as soon as you can determine their value/importance.
 - Avoid selecting too many variables in the charts.
 - Avoid using algorithms that do not scale.
