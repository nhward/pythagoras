"use strict";

console.info("[pythagoras] Script running")

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    console.info("[pythagoras] DOM has loaded");


    Shiny?.addCustomMessageHandler?.("init_card", function(opts) {
        const card = document.getElementById(opts.id);
        if (!card) {
            console.warn(`[pythagoras] Initialising card - card not found: "${opts.id}"`);
            return;
        }
        // Sortable roles (when applicable)
        initRolesCard(card);
        emitRoleMapFromCard(card);
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
            flipToggle(card)
        });
        // Closing
        const closeButton = card.querySelector(".close-btn");
        closeButton?.addEventListener("click", (evt) => {
            closeCard(card)
        });
    });


    function flipToggle(card) {
        const cardbody = card.querySelector(".card-body");
        cardbody.scrollTop = 0;  // bring contents to the top before flipping
        cardbody.scrollLeft= 0;  // bring contents to the left before flipping
        cardbody.classList.toggle("flipped");
    }

    function closeCard(card) {
        const ns = card.id.replace(/-Card$/, "");
        window.Shiny?.setInputValue?.(`${ns}-RemoveCard`, {
            id: card.id,
            ts: Date.now()
        }, { priority: "event" });
    };

    /* animate element e.g. shakeX or bounce */
    Shiny?.addCustomMessageHandler?.("animate", function(opts) {
        console.log("[pythagoras] animate running");
        opts = opts || {};
        const el = document.getElementById(opts.id);
        if (!el) {
            console.warn("[pythagoras] animate element not found: ", opts.id);
            return;
        }
        const animClass = `animate__${opts.animation}`;
        el.classList.remove("animate__animated", animClass);
        void el.offsetWidth;  //reflow
        if (opts.delay != null)    el.style.setProperty("--animate-delay", `${opts.delay}ms`);
        if (opts.duration != null) el.style.setProperty("--animate-duration", `${opts.duration}ms`);
        el.classList.add("animate__animated", animClass);
        // optional: remove the animation class after animation ends (so you can re-trigger easily)
        el.addEventListener("animationend", () => {
            // cleanup any inline overrides we applied
            if (opts.delay != null)    el.style.removeProperty("--animate-delay");
            if (opts.duration != null) el.style.removeProperty("--animate-duration");
            el.classList.remove("animate__animated", animClass);
        }, { once: true });
    });

    // Hide or show a card (or any element)
    Shiny?.addCustomMessageHandler?.("toggle_visibility", function(opts) {
        console.log(`[pythagoras] toggle_visibility running for ${opts.id}`);
        const el = document.getElementById(opts.id);
        if (!el) {
            console.warn(`[pythagoras] toggle_visibility has not found id "${opts.id}"`);
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
        console.debug("[pythagoras] PopulateRoles running");
        msg = msg || {};
        const card = document.getElementById(msg.card);
        if (!card) {
            console.error("[pythagoras] Card element not found:", msg.card);
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
            // console.debug(`[pythagoras] Role ${role} in card '${msg.card}'`);
            if (!bucket) {
                console.warn(`[pythagoras] No bucket found for role '${role}' in card '${msg.card}'`);
                return;
            }
            // console.debug(`[pythagoras] Variables ${columns} in card '${msg.card}'`);
            (columns || []).forEach((col) => {
                const chip = document.createElement("div");
                chip.className = "var-chip";
                chip.dataset.varname = col;
                chip.textContent = col;
                bucket.appendChild(chip);
                // console.debug(`[pythagoras] Added ${col} to role ${role} in card '${msg.card}'`);
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
            const flipped = fsCard.querySelector(".flipped")
            if (flipped) {
                flipped.classList.remove("flipped");
            } else {
                contractCard(fsCard);
            }
        } else {
            document.querySelectorAll(".flipped").forEach((el) => el.classList.remove("flipped"));
        }
    });

    function publishCardOrder() {
        // Sortable behaviour for card drag and drop
        const container = document.getElementById("cards-container");
        console.info("[pythagoras] Sortable-Script running (for card drag and drop)")
        const ids = Array.from(container.children).map((x) => x.id).filter(Boolean);
        window.Shiny?.setInputValue?.("CardOrder", ids, { priority: "event" })
    };
    
    publishCardOrder() //ensure this is initially available

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
    
    const container = document.getElementById("cards-container");

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
            publishCardOrder();
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

    Shiny.addCustomMessageHandler("UpdateCardOrder", (msg) => {
        publishCardOrder();
    });

    Shiny.addCustomMessageHandler("set_input", (msg) => {
        Shiny.setInputValue(msg.id, msg.value, { priority: "event" });
    });

});
