"use strict";

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    console.info("DOM has loaded");


    Shiny?.addCustomMessageHandler?.("init_card", function(opts) {
        const card = document.getElementById(opts.id);
        if (!card) {
            console.warn(`Initialising card - card not found: "${opts.id}"`);
            return;
        }
        console.debug(`Initialising card "${opts.id}"`);
        // Sortable roles (when applicable)
        initRolesCard(card);
        emitRoleMapFromCard(card);
        initParallelCoordinatesHover(card);
        // Expanding
        const expandButton = card.querySelector(".expand-btn");
        expandButton?.addEventListener("click", () => {
            expandCard(card);
        });
        // Contracting
        const contractButton = card.querySelector(".contract-btn");
        contractButton?.parentElement?.classList.add("hidden");  // hide its wrapper initially
        contractButton?.addEventListener("click", () => {
            contractCard(card);
        });
        // Flipping
        const flipButton = card.querySelector(".flip-btn");
        flipButton?.addEventListener("click", () => {
            flipToggle(card);
        });
        publishCardFace(card);
    });


    function publishCardFace(card) {
        const cardbody = card?.querySelector(".card-body");
        if (!cardbody) return;
        const isFront = !cardbody.classList.contains("flipped");
        window.Shiny?.setInputValue?.(
            `${card.id}_is_front`,
            isFront,
            { priority: "event" }
        );
    }

    function setCardFlipped(card, flipped) {
        const cardbody = card?.querySelector(".card-body");
        if (!cardbody) return;
        cardbody.scrollTop = 0;
        cardbody.scrollLeft = 0;
        cardbody.classList.toggle("flipped", Boolean(flipped));
        publishCardFace(card);
    }

    function initParallelCoordinatesHover(card) {
        const host = card.querySelector("[data-parallel-hover='true']");
        if (!host || host.dataset.hoverObserverInitialized === "true") return;
        host.dataset.hoverObserverInitialized = "true";

        const tooltip = host.querySelector(".parallel-hover-tooltip");
        const comparisonLabel = host.querySelector(".parallel-comparison-label");
        const clearComparisonButton = host.querySelector(
            ".parallel-clear-comparison"
        );
        const svgNamespace = "http://www.w3.org/2000/svg";
        const comparisonOverlay = document.createElementNS(svgNamespace, "svg");
        comparisonOverlay.classList.add("parallel-comparison-overlay");
        comparisonOverlay.setAttribute("aria-hidden", "true");
        const comparisonShadow = document.createElementNS(svgNamespace, "polyline");
        comparisonShadow.classList.add("parallel-comparison-shadow");
        const comparisonLine = document.createElementNS(svgNamespace, "polyline");
        comparisonLine.classList.add("parallel-comparison-line");
        comparisonOverlay.append(comparisonShadow, comparisonLine);
        comparisonOverlay.setAttribute("hidden", "");
        host.appendChild(comparisonOverlay);

        let attachedPlot = null;
        let animationFrame = null;
        let latestEvent = null;
        let nearest = null;
        let selectedIdentity = null;

        const hideTooltip = () => {
            if (!tooltip) return;
            tooltip.hidden = true;
            tooltip.textContent = "";
        };

        const includesConstraint = (value, constraint) => {
            if (!Array.isArray(constraint) || constraint.length === 0) return true;
            const ranges = Array.isArray(constraint[0]) ? constraint : [constraint];
            return ranges.some((range) => (
                Array.isArray(range)
                && range.length >= 2
                && value >= Math.min(range[0], range[1])
                && value <= Math.max(range[0], range[1])
            ));
        };

        const chartState = () => {
            const plot = attachedPlot;
            const trace = plot?.data?.find?.((item) => item.type === "parcoords");
            const dimensions = trace?.dimensions || [];
            const identities = trace?.customdata || [];
            if (dimensions.length < 2 || identities.length === 0) return null;

            const byLabel = new Map(
                dimensions.map((dimension) => [String(dimension.label), dimension])
            );
            const axes = Array.from(
                plot.querySelectorAll(".parcoords-control-view .y-axis")
            ).map((axis) => {
                const title = axis.querySelector(".axis-title");
                const brush = axis.querySelector(".axis-brush .background");
                const label = title?.getAttribute("data-unformatted")
                    || title?.textContent;
                const rectangle = brush?.getBoundingClientRect();
                const dimension = byLabel.get(String(label));
                if (!rectangle || !dimension) return null;
                return {
                    dimension,
                    x: rectangle.left + rectangle.width / 2,
                    top: rectangle.top,
                    bottom: rectangle.bottom,
                };
            }).filter(Boolean).sort((left, right) => left.x - right.x);
            if (axes.length < 2) return null;
            const ordinate = (axis, value) => {
                const values = axis.dimension.values || [];
                const supplied = axis.dimension.range;
                let low;
                let high;
                if (Array.isArray(supplied) && supplied.length >= 2) {
                    [low, high] = supplied;
                } else {
                    const finite = Array.from(values).filter(Number.isFinite);
                    low = Math.min(...finite);
                    high = Math.max(...finite);
                }
                if (!Number.isFinite(value) || !Number.isFinite(low)
                        || !Number.isFinite(high)) return null;
                if (low === high) return (axis.top + axis.bottom) / 2;
                const fraction = (value - low) / (high - low);
                return axis.bottom - fraction * (axis.bottom - axis.top);
            };
            const rowVisible = (row) => dimensions.every((dimension) => (
                includesConstraint(
                    Number(dimension.values?.[row]),
                    dimension.constraintrange,
                )
            ));
            return { plot, trace, dimensions, identities, axes, ordinate, rowVisible };
        };

        const nearestLine = (event, state) => {
            if (!event || !state || event.clientY < state.axes[0].top
                    || event.clientY > state.axes[0].bottom) return null;
            let left = null;
            let right = null;
            for (let index = 0; index < state.axes.length - 1; index += 1) {
                if (event.clientX >= state.axes[index].x
                        && event.clientX <= state.axes[index + 1].x) {
                    left = state.axes[index];
                    right = state.axes[index + 1];
                    break;
                }
            }
            if (!left || !right) return null;
            const fraction = (event.clientX - left.x) / (right.x - left.x);
            const rowCount = Math.min(
                state.identities.length,
                left.dimension.values?.length || 0,
                right.dimension.values?.length || 0,
            );
            let closestRow = -1;
            let closestY = null;
            let closestDistance = Number.POSITIVE_INFINITY;
            for (let row = 0; row < rowCount; row += 1) {
                if (!state.rowVisible(row)) continue;
                const leftY = state.ordinate(
                    left, Number(left.dimension.values[row])
                );
                const rightY = state.ordinate(
                    right, Number(right.dimension.values[row])
                );
                if (leftY === null || rightY === null) continue;
                const lineY = leftY + fraction * (rightY - leftY);
                const distance = Math.abs(event.clientY - lineY);
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestRow = row;
                    closestY = lineY;
                }
            }
            const threshold = card.classList.contains("fullscreen-active") ? 8 : 6;
            if (closestRow < 0 || closestDistance > threshold) return null;
            return { row: closestRow, y: closestY, state };
        };

        const drawComparison = (state) => {
            const row = selectedIdentity === null || !state
                ? -1
                : Array.from(state.identities).findIndex(
                    (identity) => String(identity) === selectedIdentity
                );
            if (row < 0 || !state.rowVisible(row)) {
                comparisonOverlay.setAttribute("hidden", "");
                return;
            }
            const hostRectangle = host.getBoundingClientRect();
            const points = state.axes.map((axis) => {
                const y = state.ordinate(
                    axis, Number(axis.dimension.values?.[row])
                );
                if (y === null) return null;
                return `${axis.x - hostRectangle.left},${y - hostRectangle.top}`;
            }).filter(Boolean);
            if (points.length !== state.axes.length) {
                comparisonOverlay.setAttribute("hidden", "");
                return;
            }
            comparisonOverlay.setAttribute("viewBox", (
                `0 0 ${hostRectangle.width} ${hostRectangle.height}`
            ));
            comparisonShadow.setAttribute("points", points.join(" "));
            comparisonLine.setAttribute("points", points.join(" "));
            comparisonOverlay.removeAttribute("hidden");
        };

        const identifyNearestLine = () => {
            animationFrame = null;
            const state = chartState();
            drawComparison(state);
            nearest = nearestLine(latestEvent, state);
            if (!nearest || !tooltip) {
                attachedPlot?.classList.remove("parallel-line-near");
                hideTooltip();
                return;
            }
            attachedPlot.classList.add("parallel-line-near");
            const closestRow = nearest.row;
            const closestY = nearest.y;
            const hostRectangle = host.getBoundingClientRect();
            tooltip.textContent = String(state.identities[closestRow]);
            tooltip.hidden = false;
            const preferredLeft = latestEvent.clientX - hostRectangle.left + 12;
            const preferredTop = closestY - hostRectangle.top - 14;
            const boundedLeft = Math.max(
                4,
                Math.min(preferredLeft, hostRectangle.width - tooltip.offsetWidth - 4),
            );
            const boundedTop = Math.max(
                4,
                Math.min(preferredTop, hostRectangle.height - tooltip.offsetHeight - 4),
            );
            tooltip.style.left = `${boundedLeft}px`;
            tooltip.style.top = `${boundedTop}px`;
        };

        const selectComparison = (event) => {
            const candidate = nearestLine(event, chartState());
            if (!candidate) return;
            nearest = candidate;
            const identity = String(candidate.state.identities[candidate.row]);
            selectedIdentity = selectedIdentity === identity ? null : identity;
            if (comparisonLabel) {
                comparisonLabel.textContent = selectedIdentity
                    ? `Comparing: ${selectedIdentity}`
                    : "";
                comparisonLabel.hidden = selectedIdentity === null;
            }
            if (clearComparisonButton) {
                clearComparisonButton.hidden = selectedIdentity === null;
            }
            drawComparison(candidate.state);
        };

        const clearComparison = (event) => {
            event?.stopPropagation();
            selectedIdentity = null;
            comparisonOverlay.setAttribute("hidden", "");
            if (comparisonLabel) {
                comparisonLabel.textContent = "";
                comparisonLabel.hidden = true;
            }
            if (clearComparisonButton) clearComparisonButton.hidden = true;
        };

        const attach = () => {
            const plot = host.querySelector(".js-plotly-plot");
            if (!plot || plot === attachedPlot) return;
            if (attachedPlot) {
                attachedPlot.removeEventListener("mousemove", onMouseMove);
                attachedPlot.removeEventListener("mouseleave", hideTooltip);
                attachedPlot.removeEventListener("click", selectComparison);
            }
            attachedPlot = plot;
            plot.addEventListener("mousemove", onMouseMove);
            plot.addEventListener("mouseleave", hideTooltip);
            plot.addEventListener("click", selectComparison);
            requestAnimationFrame(() => drawComparison(chartState()));
        };

        function onMouseMove(event) {
            if (event.buttons) {
                hideTooltip();
                return;
            }
            latestEvent = event;
            if (animationFrame === null) {
                animationFrame = requestAnimationFrame(identifyNearestLine);
            }
        }

        clearComparisonButton?.addEventListener("click", clearComparison);

        const observer = new MutationObserver(attach);
        observer.observe(host, { childList: true, subtree: true });
        attach();
    }

    function flipToggle(card) {
        const cardbody = card?.querySelector(".card-body");
        if (!cardbody) return;
        setCardFlipped(card, !cardbody.classList.contains("flipped"));
    }

    /* animate element e.g. shakeX or bounce */
    Shiny?.addCustomMessageHandler?.("animate", function(opts) {
        console.log("animation running");
        opts = opts || {};
        const el = document.getElementById(opts.id);
        if (!el) {
            console.warn("animate element not found: ", opts.id);
            return;
        }
        const lockedElement = document.getElementById(opts.lock);
        const animClass = `animate__${opts.animation}`;
        const delay = opts.delay ?? 0;
        const duration = opts.duration ?? 1000;
        el.classList.remove("animate__animated", animClass);
        void el.offsetWidth;  //reflow
        el.style.setProperty("--animate-delay", `${delay}ms`);
        el.style.setProperty("--animate-duration", `${duration}ms`);
        if (lockedElement) {
            lockedElement.setAttribute("inert", "");
            lockedElement.setAttribute("aria-busy", "true");
            lockedElement.classList.add("animation-locked");
        }
        el.classList.add("animate__animated", animClass);
        // optional: remove the animation class after animation ends (so you can re-trigger easily)
        function cleanup() {
            // cleanup any inline overrides we applied
            el.style.removeProperty("--animate-delay");
            el.style.removeProperty("--animate-duration");
            el.classList.remove("animate__animated", animClass);
            if (lockedElement) {
                lockedElement.removeAttribute("inert", "");
                lockedElement.removeAttribute("aria-busy", "true");
                lockedElement.classList.remove("animation-locked");
            }
        } 
        el.addEventListener("animationend", cleanup, { once: true });
        el._animationCleanupTimer = setTimeout(cleanup, delay + duration + 100);
    });

    
    // Hide or show a card (or any element)
    Shiny?.addCustomMessageHandler?.("toggle_visibility", function(opts) {
        console.log(`toggle_visibility running for ${opts.id}`);
        const el = document.getElementById(opts.id);
        if (!el) {
            console.warn(`toggle_visibility has not found id "${opts.id}"`);
            return;
        }
        if (opts.visible) el.classList.remove("hidden");
        else el.classList.add("hidden");
    });

    /* Sortable role-assignment elements */
    function emitRoleMapFromCard(card) { /* assign <ns>-role_map with current assignments */
        const lists = card.querySelectorAll(".sortable-role");
        if (lists.length === 0) return;
        const payload = {};
        lists.forEach((el) => {
            const role = el.dataset.role;
            payload[role] = Array.from(el.children).map(x => x.dataset.varname);
        });
        const ns = card.id.replace(/-Card$/, "");
        window.Shiny?.setInputValue?.(`${ns}-role_map`, payload, { priority: "event" });
    }

    function emitRoleMap(evt) {
        const card = evt.item.closest(".card");
        emitRoleMapFromCard(card);
    }

    function initRolesCard(card) {
        const lists = card.querySelectorAll(".sortable-role");
        if (lists.length === 0) return;
        lists.forEach((el) => {
            if (el.dataset.sortableInitialized === "true") return;
            if (Sortable.create(el, {
                group: "variable-roles",
                animation: 150,
                ghostClass: "ghost",
                chosenClass: "chosen",
                onEnd: emitRoleMap
            })) {
                el.dataset.sortableInitialized = "true";
            }
        });
    }

    // Use the json in msg.role_map to populate the various divs that relate to the role asignment dialogue.
    window.populateRolesHandler = function(msg) {
        console.debug("PopulateRoles running");
        msg = msg || {};
        const card = document.getElementById(msg.card);
        if (!card) {
            console.error("Card element not found:", msg.card);
            return;
        }
        initRolesCard(card);
        const roleMap = msg.role_map || {};
        // Clear all existing chips from all role buckets in this card
        card.querySelectorAll(".sortable-role").forEach((bucket) => {
            bucket.replaceChildren();
        });
        // Rebuild each role bucket from the payload
        Object.entries(roleMap).forEach(([role, columns]) => {
            const bucket = card.querySelector(`.sortable-role[data-role="${role}"]`);
            // console.debug(`Role ${role} in card '${msg.card}'`);
            if (!bucket) {
                console.warn(`No bucket found for role '${role}' in card '${msg.card}'`);
                return;
            }
            // console.debug(`Variables ${columns} in card '${msg.card}'`);
            (columns || []).forEach((col) => {
                const chip = document.createElement("div");
                chip.className = "var-chip";
                chip.dataset.varname = col;
                chip.textContent = col;
                bucket.appendChild(chip);
                // console.debug(`Added ${col} to role ${role} in card '${msg.card}'`);
            });
        });
        emitRoleMapFromCard(card);
    };

    Shiny?.addCustomMessageHandler?.("PopulateRoles", window.populateRolesHandler)

    // ---- helpers so clicks and keyboard use the same behavior ----
    const expandCard = (card) => {
        if (!card || card.classList.contains("fullscreen-active")) return;
        card.scrollTop = 0;  // bring contents to the top before expanding
        card.scrollLeft= 0;  // bring contents to the left before expanding
        const cnt = card.querySelector(".roles-layout")
        if (cnt) {
            cnt.scrollLeft = 0;  // bring contents to the left before expanding
        }
        const contractBtn = card.querySelector(".contract-btn");
        const expandWrapper = card.querySelector(".expand-btn")?.parentElement;
        const closeWrapper  = card.querySelector(".close-btn")?.parentElement;

        // Hide the expand button's wrapper
        expandWrapper?.classList.add("hidden");
        // Swap maximize → minimize (+ hide close while fullscreen)
        contractBtn?.parentElement?.classList.remove("hidden");
        closeWrapper?.classList.add("hidden");

        card.classList.add("fullscreen-active");
        document.body.classList.add("fullscreen-mode");
        card.setAttribute?.("aria-expanded", "true");
        card.setAttribute("aria-modal", "true");
        window.Shiny?.setInputValue?.(`${card.id}_full_screen`, true, { priority: "event" });
    };

    const contractCard = (card) => {
        if (!card || !card.classList.contains("fullscreen-active")) return;
        card.scrollTop = 0;  // bring contents to the top before contracting
        card.scrollLeft= 0;  // bring contents to the left before contracting
        const cnt = card.querySelector(".roles-layout")
        if (cnt) {
            cnt.scrollLeft = 0;  // bring contents to the left before expanding
        }
        // Find header controls relative to this card
        const contractBtn = card.querySelector(".contract-btn");
        const expandWrapper = card.querySelector(".expand-btn")?.parentElement;
        const closeWrapper  = card.querySelector(".close-btn")?.parentElement;

        // Hide minimize, show maximize + close
        contractBtn?.parentElement?.classList.add("hidden");
        expandWrapper?.classList.remove("hidden");
        closeWrapper?.classList?.remove("hidden");

        card.classList.remove("fullscreen-active");
        document.body.classList.remove("fullscreen-mode");
        contractBtn?.setAttribute?.("aria-expanded", "false");
        window.Shiny?.setInputValue?.(`${card.id}_full_screen`, false, { priority: "event" });
        // Now draw attention to where it landed
        requestAnimationFrame(() => highlightCard(card, { jello: true }));
    };

    function highlightCard(card, { jello = true } = {}) {
        if (!card) return;
        // Make sure users can *see* where it went
        // (do after you remove fullscreen class / restore DOM)
        card.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
        // Trigger animation
        const cls = [];
        cls.push("animate-highlight");
        if (jello) cls.push("animate-jello");
        card.classList.add(...cls);
        // Clean up classes when done (with a timeout safety)
        const cleanup = () => {
            card.classList.remove("animate-highlight", "animate-jello");
            card.removeEventListener("animationend", cleanup);
        };
        card.addEventListener("animationend", cleanup);
        setTimeout(cleanup, 1500);
     };


    // ---- ESC key: unflip if flipped; else contract if fullscreen ----
    document.addEventListener("keydown", (evt) => {
        if (evt.key !== "Escape") return;

        // Don’t interfere while typing
        const t = evt.target;
        if (t && (t.closest?.("input, textarea, select") || t.isContentEditable)) return;

        // Prefer to resolve a flip first (whether fullscreen or not)
        // Pick the most relevant card: fullscreen one if present; otherwise nearest card to focus
        const fsCard = document.querySelector(".card.fullscreen-active");
        if (fsCard) {
            const flipped = fsCard.querySelector(".card-body.flipped");
            if (flipped) {
                setCardFlipped(fsCard, false);
            } else {
                contractCard(fsCard);
            }
        } else {
            document.querySelectorAll(".card .card-body.flipped").forEach((cardbody) => {
                const card = cardbody.closest(".card");
                if (card) setCardFlipped(card, false);
            });
        }
    });

    window.fullscreen_app = function(msg) {
        var element = document.documentElement,
        enterFS = element.requestFullscreen || element.msRequestFullscreen || element.mozRequestFullScreen || element.webkitRequestFullscreen,
        exitFS = document.exitFullscreen || document.msExitFullscreen || document.mozCancelFullScreen || document.webkitExitFullscreen;
        if (!document.fullscreenElement && !document.msFullscreenElement && !document.mozFullScreenElement && !document.webkitFullscreenElement) {
            enterFS.call(element);
        } else {
            exitFS.call(document);
        }
    };
    Shiny?.addCustomMessageHandler?.("fullscreen_app", window.fullscreen_app);

    window.quit_app = function(msg) {
        window.close();
    };
    Shiny?.addCustomMessageHandler?.("quit_app", window.quit_app);

    document.querySelectorAll(".bslib-sidebar-layout.sidebar-collapsed.sidebar-right>.collapse-toggle").forEach((btn) => btn.classList.add("hover-btn"));

    Shiny.addCustomMessageHandler("set_input", (msg) => {
        Shiny.setInputValue(msg.id, msg.value, { priority: "event" });
    });

    Shiny.addCustomMessageHandler("UpdateCardOrder", (msg) => {
        publishCardOrder(msg.id, msg.input_id);
    });


    function publishCardOrder(id, input_id) {
        // Sortable behaviour for card drag and drop
        const container = document.getElementById(id);
        if (! container) {
            console.warn(`Sortable container not found: "${id}"`)
            return
        }
        const ids = Array.from(container.children)
            .filter((element) => element.classList.contains("card"))
            .map((element) => element.id)
            .filter(Boolean);
        window.Shiny?.setInputValue?.(input_id, ids, { priority: "event" })
        setSectionEmptyState(container.dataset.sectionId, ids.length === 0);
    };

    function sectionElement(selector, section) {
        return Array.from(document.querySelectorAll(selector)).find(
            (element) => element.dataset.sectionId === section
        );
    }

    function ensureSectionDeleteControl(section) {
        if (!section) return null;

        const existing = sectionElement(".delete-empty-section", section);
        if (existing) return existing;

        const tab = Array.from(
            document.querySelectorAll("#Navbar .nav-link[data-value]")
        ).find((element) => element.dataset.value === section);
        const accordionItem = Array.from(
            document.querySelectorAll("#Accordion .accordion-item[data-value]")
        ).find((element) => element.dataset.value === section);
        const host = tab?.closest(".nav-item")
            || accordionItem?.querySelector(".accordion-header");
        if (!host) return null;
        const displayName = tab?.textContent.trim()
            || accordionItem?.querySelector(".accordion-title")?.textContent.trim()
            || section;

        const control = document.createElement("button");
        control.type = "button";
        control.className = "delete-empty-section";
        control.dataset.sectionId = section;
        control.title = `Delete empty section ${displayName}`;
        control.setAttribute(
            "aria-label",
            `Delete empty section ${displayName}`
        );
        control.textContent = "\u00d7";
        host.classList.add(
            tab ? "section-nav-item-with-delete" : "section-accordion-header-with-delete"
        );
        host.appendChild(control);
        return control;
    }

    function setSectionEmptyState(section, empty) {
        if (!section) return;
        const emptyState = sectionElement(".section-empty-state", section);
        if (emptyState) emptyState.hidden = !empty;
        const control = ensureSectionDeleteControl(section);
        if (control) control.hidden = !empty;
    }

    document.addEventListener("click", (event) => {
        const deleteControl = event.target.closest(".delete-empty-section");
        if (deleteControl) {
            event.preventDefault();
            event.stopPropagation();
            window.Shiny?.setInputValue?.(
                "DeleteSection",
                {
                    section: deleteControl.dataset.sectionId,
                    nonce: Date.now()
                },
                { priority: "event" }
            );
            return;
        }

        const addControl = event.target.closest(".section-add-card");
        if (addControl) {
            window.Shiny?.setInputValue?.(
                "AddCardToSection",
                { section: addControl.dataset.sectionId, nonce: Date.now() },
                { priority: "event" }
            );
        }
    });

    document.addEventListener("shown.bs.tab", (event) => {
        if (!event.target.closest("#AddItemType")) return;
        const action = event.target.dataset.value;
        const button = event.target.closest(".modal")?.querySelector(
            "#CardPicker_ok"
        );
        if (!button?.dataset.actionLabels) return;
        try {
            const labels = JSON.parse(button.dataset.actionLabels);
            button.textContent = labels[action] || "Continue";
        } catch (error) {
            console.warn("Could not update modal action label", error);
        }
    });

    Shiny.addCustomMessageHandler("RenameSection", (msg) => {
        const sectionId = msg?.section_id;
        const name = msg?.name;
        if (!sectionId || !name) return;

        const tab = Array.from(
            document.querySelectorAll("#Navbar .nav-link[data-value]")
        ).find((element) => element.dataset.value === sectionId);
        if (tab) tab.textContent = name;

        const accordionItem = Array.from(
            document.querySelectorAll("#Accordion .accordion-item[data-value]")
        ).find((element) => element.dataset.value === sectionId);
        const accordionTitle = accordionItem?.querySelector(".accordion-title");
        if (accordionTitle) accordionTitle.textContent = name;

        const deleteControl = sectionElement(
            ".delete-empty-section",
            sectionId
        );
        if (deleteControl) {
            deleteControl.title = `Delete empty section ${name}`;
            deleteControl.setAttribute(
                "aria-label",
                `Delete empty section ${name}`
            );
        }
    });

    Shiny.addCustomMessageHandler("MakeSortable", (msg) => {
        const container = document.getElementById(msg.id);
        if (! container) {
            console.warn(`Sortable container not found: "${msg.id}"`)
            return
        }
        console.info(`Sortable container found: "${msg.id}"`)
        if (container.dataset.sortableInitialized !== "true") {
            Sortable.create(container, {
                animation: 150,
                handle: ".drag-handle",
                ghostClass: "drag-ghost",
                forceFallback: true,
                fallbackOnBody: true,
                // make drag clone safe immediately
                onClone: (evt) => {
                    // evt.clone is the element appended to <body>
                    markCloneNonBindable(evt.clone);
                },
                onStart: () => {
                    container.classList.add("dragging");
                },
                onEnd: () => {
                    // Remove any leftover ghost flags/clone
                    if (dragClone && dragClone.parentNode) {
                        // defensive: ensure the clone can never be bound
                        dragClone.setAttribute("data-shiny-ignore", "true");
                        dragClone.remove();   // get it out of the DOM
                    }
                    dragClone = null;
                    container.classList.remove("dragging");
                    clearSelection();
                    publishCardOrder(msg.id, msg.input_id);
                }
            });
            container.dataset.sortableInitialized = "true";
        }
        publishCardOrder(msg.id, msg.input_id);
    });


    const clearSelection = () => {
        try {
        const sel = window.getSelection && window.getSelection();
        if (sel && sel.removeAllRanges) sel.removeAllRanges();
        else if (sel && sel.empty) sel.empty();
        else if (document.selection) document.selection.empty();
        } catch {}
    };

    // We’ll keep a reference to the ghost clone so we can nuke IDs / mark as non-bindable
    let dragClone = null;
    const markCloneNonBindable = (clone) => {
        if (!clone) return;
        dragClone = clone;
        // Flag the whole subtree as non-bindable
        clone.setAttribute("data-shiny-ignore", "true");  // Shiny ignores elements with this attr
        clone.classList.add("sortable-ghost-nobind");
        // Remove duplicate IDs in the clone to avoid any accidental bindings/lookups
        clone.querySelectorAll("[id]").forEach((el) => el.removeAttribute("id"));
    };

    const bookmarkDatabaseName = "pythagoras-bookmarks";
    const bookmarkStoreName = "bookmarks";
    const stagedBookmarkKey = "pythagoras-bookmark-once";
    const settingsQueryParameter = "_pythagoras_settings";
    const systemSettingNames = [
        "section_style",
        "show_start",
        "reuse_cards",
        "max_card_height",
        "max_dupl_cards",
    ];

    const systemSettings = (configuration) => Object.fromEntries(
        systemSettingNames.map((name) => [
            name,
            configuration?.settings?.[name],
        ]),
    );

    const renderedSystemSettings = () => {
        const content = document.querySelector(
            'meta[name="pythagoras-settings"]',
        )?.content;
        return content ? systemSettings({ settings: JSON.parse(content) }) : {};
    };

    const clearSettingsQuery = () => {
        const url = new URL(window.location.href);
        if (!url.searchParams.has(settingsQueryParameter)) return;
        url.searchParams.delete(settingsQueryParameter);
        window.history.replaceState(null, "", url);
    };

    const openBookmarkDatabase = () => new Promise((resolve, reject) => {
        const request = indexedDB.open(bookmarkDatabaseName, 1);
        request.onupgradeneeded = () => {
            if (!request.result.objectStoreNames.contains(bookmarkStoreName)) {
                request.result.createObjectStore(
                    bookmarkStoreName,
                    { keyPath: "filename" },
                );
            }
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });

    const bookmarkRecords = async () => {
        const database = await openBookmarkDatabase();
        try {
            return await new Promise((resolve, reject) => {
                const request = database.transaction(
                    bookmarkStoreName,
                    "readonly",
                ).objectStore(bookmarkStoreName).getAll();
                request.onsuccess = () => resolve(
                    request.result.sort(
                        (left, right) => right.createdAt - left.createdAt,
                    ),
                );
                request.onerror = () => reject(request.error);
            });
        } finally {
            database.close();
        }
    };

    const addBookmarkRecord = async (record) => {
        const database = await openBookmarkDatabase();
        try {
            await new Promise((resolve, reject) => {
                const request = database.transaction(
                    bookmarkStoreName,
                    "readwrite",
                ).objectStore(bookmarkStoreName).add(record);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
        } finally {
            database.close();
        }
    };

    const bookmarkRecord = async (filename) => {
        const database = await openBookmarkDatabase();
        try {
            return await new Promise((resolve, reject) => {
                const request = database.transaction(
                    bookmarkStoreName,
                    "readonly",
                ).objectStore(bookmarkStoreName).get(filename);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        } finally {
            database.close();
        }
    };

    const bookmarkFilenameParts = (filename) => {
        const suffix = ".pythagoras.json";
        if (!filename.endsWith(suffix)) return null;
        const stem = filename.slice(0, -suffix.length);
        const separator = stem.lastIndexOf("--");
        if (separator <= 0 || separator === stem.length - 2) return null;
        return {
            dataName: stem.slice(0, separator),
            timestamp: stem.slice(separator + 2),
        };
    };

    const bookmarkDisplayTime = (timestamp, createdAt) => {
        const match = timestamp.match(
            /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})([+-])(\d{2})(\d{2})$/,
        );
        const value = match
            ? new Date(
                `${match[1]}-${match[2]}-${match[3]}`
                + `T${match[4]}:${match[5]}:${match[6]}`
                + `${match[7]}${match[8]}:${match[9]}`,
            )
            : new Date(createdAt * 1000);
        return value.toLocaleString();
    };

    const publishBookmarkList = async (inputId) => {
        if (!inputId) return;
        const records = await bookmarkRecords();
        Shiny.setInputValue(
            inputId,
            records.flatMap(({ filename, createdAt }) => {
                const parts = bookmarkFilenameParts(filename);
                if (!parts) return [];
                return [{
                    filename,
                    createdAt,
                    ...parts,
                    displayTime: bookmarkDisplayTime(
                        parts.timestamp,
                        createdAt,
                    ),
                }];
            }),
            { priority: "event" },
        );
    };

    const reportBookmarkOperation = (inputId, ok, message) => {
        if (!inputId) return;
        Shiny.setInputValue(
            inputId,
            { ok, message, nonce: Date.now() },
            { priority: "event" },
        );
    };

    const downloadBookmark = (filename, configuration) => {
        const blob = new Blob(
            [`${JSON.stringify(configuration, null, 2)}\n`],
            { type: "application/json" },
        );
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
    };

    const stageBookmarkAndReload = (configuration) => {
        sessionStorage.setItem(
            stagedBookmarkKey,
            JSON.stringify(configuration),
        );
        const url = new URL(window.location.href);
        url.searchParams.set(
            settingsQueryParameter,
            JSON.stringify(systemSettings(configuration)),
        );
        // Let the Shiny flush that delivered this message finish before closing
        // its WebSocket. Reloading synchronously can leave the server attempting
        // to complete work against a session that the browser has just closed.
        window.setTimeout(() => window.location.replace(url), 250);
    };

    let bookmarkStartupPublished = false;
    const publishBookmarkStartup = async () => {
        if (bookmarkStartupPublished) return;
        bookmarkStartupPublished = true;
        try {
            const staged = sessionStorage.getItem(stagedBookmarkKey);
            if (staged) {
                sessionStorage.removeItem(stagedBookmarkKey);
                clearSettingsQuery();
                Shiny.setInputValue(
                    "BookmarkStartup",
                    {
                        ready: true,
                        source: "staged",
                        configuration: JSON.parse(staged),
                    },
                    { priority: "event" },
                );
                return;
            }
            const records = await bookmarkRecords();
            const latest = records[0];
            if (
                latest
                && JSON.stringify(systemSettings(latest.configuration))
                    !== JSON.stringify(renderedSystemSettings())
            ) {
                stageBookmarkAndReload(latest.configuration);
                return;
            }
            Shiny.setInputValue(
                "BookmarkStartup",
                {
                    ready: true,
                    source: latest ? "indexeddb" : "none",
                    configuration: latest?.configuration || null,
                },
                { priority: "event" },
            );
        } catch (error) {
            console.warn("Could not read browser bookmarks", error);
            Shiny.setInputValue(
                "BookmarkStartup",
                { ready: true, source: "none", configuration: null },
                { priority: "event" },
            );
        }
    };

    // `shiny:connected` fires before Shiny sends its `init` message. Publishing
    // an input from that event can therefore make the server receive `update`
    // while the session is still in its Start state. The first idle event is
    // emitted only after the initial server flush, when updates are valid.
    window.jQuery(document).one("shiny:idle", publishBookmarkStartup);

    Shiny.addCustomMessageHandler("bookmark_list", async (message) => {
        try {
            await publishBookmarkList(message.inputId);
        } catch (error) {
            reportBookmarkOperation(
                message.operationInputId,
                false,
                `Could not list browser bookmarks: ${error.message}`,
            );
        }
    });

    Shiny.addCustomMessageHandler("bookmark_save", async (message) => {
        try {
            await addBookmarkRecord({
                filename: message.filename,
                createdAt: message.createdAt,
                configuration: message.configuration,
            });
            downloadBookmark(message.filename, message.configuration);
            await publishBookmarkList(message.listInputId);
            reportBookmarkOperation(
                message.operationInputId,
                true,
                `Bookmark saved as ${message.filename}.`,
            );
        } catch (error) {
            reportBookmarkOperation(
                message.operationInputId,
                false,
                error?.name === "ConstraintError"
                    ? `Bookmark ${message.filename} already exists.`
                    : `Bookmark was not saved: ${error.message}`,
            );
        }
    });

    Shiny.addCustomMessageHandler("bookmark_load", async (message) => {
        try {
            const record = await bookmarkRecord(message.filename);
            if (!record) throw new Error("The selected bookmark was not found");
            stageBookmarkAndReload(record.configuration);
        } catch (error) {
            reportBookmarkOperation(
                message.operationInputId,
                false,
                `Bookmark could not be loaded: ${error.message}`,
            );
        }
    });

    Shiny.addCustomMessageHandler("bookmark_import", async (message) => {
        try {
            await addBookmarkRecord({
                filename: message.filename,
                createdAt: message.createdAt,
                configuration: message.configuration,
            });
            stageBookmarkAndReload(message.configuration);
        } catch (error) {
            reportBookmarkOperation(
                message.operationInputId,
                false,
                error?.name === "ConstraintError"
                    ? `Bookmark ${message.filename} already exists.`
                    : `Bookmark could not be imported: ${error.message}`,
            );
        }
    });

    Shiny.addCustomMessageHandler(
        "bookmark_reload_configuration",
        (message) => stageBookmarkAndReload(message.configuration),
    );

});
