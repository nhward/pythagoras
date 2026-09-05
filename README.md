

# <img src="app/www/tetractys.png" alt="tetractys" width="40" height="40"> Pythagoras

*A modular, workflow-driven data science environment built with Shiny for Python.*

---

## Overview

**Pythagoras** is an experimental framework for building interactive, modular data science workflows using **Shiny for Python**. It reimagines the typical notebook or pipeline approach as a sequence of **self-contained “cards”**, each responsible for a specific analytical task.

Rather than forcing a rigid pipeline, Pythagoras allows users to:

- Construct linear workflows dynamically
- Reorder steps via drag-and-drop
- Insert or remove analytical components at runtime
- Inspect and extract the underlying code used at each step

The result is a system that addresses both:
- the **data problem** (cleaning, transforming, modeling)
- the **workflow problem** (how analysis is structured, communicated, and reproduced)

---

## Core Concepts

### 🧩 Cards

A **card** is the fundamental unit of computation and interaction.

Each card:
- Receives data from the previous card
- Optionally transforms or augments that data
- Passes the result downstream

Cards are implemented as Shiny modules with a consistent interface and UI.

#### Standard Features

Every card can support:

- 🔄 **Flip view** (front = visualization, back = summary/metadata)
- ⚙️ **Settings sidebar**
- 🧾 **Code extraction** (view and collect executable snippets)
- 📖 **Documentation** (markdown-driven modal window)
- 🧭 **Guided tours** (Shepherd-based walkthroughs)
- 🖥️ **Full-screen mode** (higher resolution and greater detail)
- 🟰 **Consistent styling and controls**
- 🧲 **Drag handle** (for reordering)

---

### 🔗 Workflow = Ordered Cards

A workflow is defined by:

1. **Initial dataset**
2. **Ordered linear sequence of cards**
3. **User interactions within each card**

This is a key architectural choice.

---

### 🔀 Dynamic Reordering

Cards can be:
- Cards are organized into sections
- Reordered via drag-and-drop (using Sortable.js) within a section
- Inserted into a section from a library of available cards
- Removed from a section

This enables:

- Rapid experimentation
- Multiple analytical paths
- User-defined workflows rather than prescribed ones

---

### 📦 Data Flow Model

Each card server consumes an upstream reactive source and returns its own reactive
output. Reordering cards changes only those source connections; Shiny then propagates
invalidations through the resulting graph. The values are structured data wrappers
(e.g. `proxy_data`) that include:

- The progressively cleaned dataset (Pandas / GeoPandas)
- A **RoleMap** describing variable roles
- A name
- A list of cleaning tasks
- A scikit pipeline (ready for a later resampling strategy)

Available roles:

- `target`
- `predictor`
- `identifier`
- `partition`
- `weighting`
- `stratifier`
- `treatment`,
- `sensitive`
- `geometry`
- `sequence`

Cards validate, chart, and transform data according to these roles.

---

## Architecture

### 🧠 Reactive Model

Pythagoras leverages Shiny’s reactive system:

- Data flows through reactive dependencies
- Cards recompute only when required
- Expensive operations can be suspended

---

### 🧱 Module System

Each card is implemented as a function:

```python
def instance():
    return Card(...)
```

This allows:

* Lazy instantiation
* Dynamic loading
* Clean separation between definition and execution

Cards inherit from base classes:

* ABC: Abstract base class
* Module: implements shiny modules
* Card: implements a common look and feel for the cards

⸻

🧩 UI Composition

Cards are rendered within a grid container:

* Responsive layout (CSS grid)
* Minimum card width enforced
* Multiple columns depending on viewport

Drag-and-drop ordering is handled via:

* Sortable.js
* Custom JS bindings
* Shiny input events (CardOrder)

⸻

Cards (available and planned)

📊 Data Prep

* [Data importation](app/www/markdown/data_import.html) 
* [Data tabulation](app/www/markdown/data_tabulation.html)
* [Role assignment](app/www/markdown/role_assignment.html)
* [Variable modification](app/www/markdown/var_modify.html)


🧹 Data cleaning

* [Duplicate Observations](app/www/markdown/obs_duplicates.html)
* [Data homogeneity](app/www/markdown/data_homogeneity.html)
* [Variable dissimilarity](app/www/markdown/var_dissimilar.html)

∅ Missing Values

* [Missing Placeholders](app/www/markdown/miss_placeholders.html)
* [Missingness types](app/www/markdown/miss_type.html)
* [Informative Missingness](app/www/markdown/miss_informative.html)
* [Missingness sets](app/www/markdown/miss_sets.html)
* [Excessive missingness](app/www/markdown/miss_map.html)
* [Missingness rules](app/www/markdown/miss_rules.html)
* [Learned imputation ](app/www/markdown/miss_impute.html)


🧠 Preprocessing

* [Variable transforms](app/www/markdown/var_transform.html)
* Feature roles
* Partitioning
* Weighting
* Stratification

Miscellaneous
* [Configuration](app/www/markdown/sys_configuration.html)
* [System log](app/www/markdown/system_log.html)
* [Data journey](app/www/markdown/data_provenance.html)

⸻

Code Extraction

Each card can expose the code it uses.

Users can:

* View code in a modal
* Collect snippets across cards
* Assemble a reproducible notebook

This bridges the gap between:

* interactive analysis
* reproducible pipelines

⸻

State Management

The full state of an analysis consists of:

* Input data
* Card sequence
* Card-specific settings

Future work includes:

* Saving/loading workflows
* Bookmarking state
* Session persistence

⸻

Testing

The project uses:

* Unit tests for core logic
* UI tests (Playwright) for interaction

Challenges addressed include:

* reactive timing
* DOM updates
* dynamic UI insertion
* drag-and-drop behavior

⸻

Design Philosophy

Pythagoras is built around a few key ideas:

1. Analysis is a sequence of small steps

The order of operations matters and each step should be visual and interactive.

2. State is explicit

There is no hidden global state — everything flows through the cards.

3. Interactivity and reproducibility must coexist

Users should be able to explore and extract code.

4. Structure should not limit exploration

Users can reorder, insert, and remove steps freely.

⸻

Limitations & Considerations

* Complex reactive timing (especially with dynamic UI)
* Browser/server coordination (JS + Shiny)
* Performance with many active cards
* Shinylive compatibility (no runtime filesystem access)

⸻

Future Directions

* Workflow saving/loading
* Card marketplace / plugin system
* Improved visual consistency
* Richer markdown editing (in-app editor)
* Better state inspection tools
* Shinylive-compatible architecture

⸻

Name

Pythagoras reflects:

* Structure and relationships
* Geometry and distance (central to data science concepts)
* A system for understanding complex spaces through composition

⸻

Summary

Pythagoras is not just a Shiny app.

It is an attempt to build:

A modular, inspectable, reorder-able "visual language" for data analysis.

⸻
