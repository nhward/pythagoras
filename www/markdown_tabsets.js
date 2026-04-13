console.log("[tabset] script running")

document.addEventListener("DOMContentLoaded", function() {
      document.querySelectorAll("h1.tabset, h2.tabset, h3.tabset").forEach(function(tabsetHeader) {
        const container = document.createElement("div");
        container.classList.add("nav", "nav-tabs");
        
        let currentPanel = null;
        let panelsContainer = document.createElement("div");
        panelsContainer.classList.add("tab-content");

        // Grab sibling headings and content
        let sibling = tabsetHeader.nextElementSibling;
        while (sibling && !["H1","H2","H3"].includes(sibling.tagName)) {
          if (sibling.tagName === "H2") {
            const tabId = "tab-" + sibling.textContent.trim().replace(/\s+/g, "-");
            const tabLink = document.createElement("a");
            tabLink.classList.add("nav-link");
            tabLink.dataset.bsToggle = "tab";
            tabLink.href = "#" + tabId;
            tabLink.innerText = sibling.textContent;
            container.appendChild(tabLink);

            currentPanel = document.createElement("div");
            currentPanel.classList.add("tab-pane", "fade");
            currentPanel.id = tabId;
            panelsContainer.appendChild(currentPanel);

          } else if (currentPanel) {
            currentPanel.appendChild(sibling.cloneNode(true));
          }
          sibling = sibling.nextElementSibling;
        }

        tabsetHeader.replaceWith(container, panelsContainer);
        if (container.firstChild) container.firstChild.classList.add("active");
        if (panelsContainer.firstChild) panelsContainer.firstChild.classList.add("active", "show");
      });
    });