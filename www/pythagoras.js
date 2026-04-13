"use strict";

console.info("[pythagoras] Script running")

// Run once DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    console.info("[pythagoras] DOM has loaded");

    /* toggle element */
    Shiny?.addCustomMessageHandler?.("toggle_flip", function(msg) {
        msg = msg || {};
        const el = document.getElementById(msg.id);
        if (!el) {
            console.warn("[pythagoras] Flip element not found: ", msg.id)
            return;
        }
        el.scrollTop = 0;  // bring contents to the top before flipping
        el.classList.toggle("flipped");
    });

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
        const payload = {};
        card.querySelectorAll(".sortable-role").forEach((el) => {
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

    function initSortableRoles() {
        document.querySelectorAll(".card").forEach((card) => {
            const lists = card.querySelectorAll(".sortable-role");
            if (lists.length === 0) return;

            console.info(`Found class "sortable-role" in ${card.id}`);

            lists.forEach((el) => {
                if (el.dataset.sortableInitialized === "true") return;

                Sortable.create(el, {
                    group: "variable-roles",
                    animation: 150,
                    ghostClass: "ghost",
                    chosenClass: "chosen",
                    onEnd: emitRoleMap
                });

                el.dataset.sortableInitialized = "true";
            });
        });
    }

    function emitAllRoleMaps() {
        document.querySelectorAll(".card").forEach((card) => {
            const lists = card.querySelectorAll(".sortable-role");
            if (lists.length > 0) {
                emitRoleMapFromCard(card);
            }
        });
    }

    initSortableRoles();
    if (window.Shiny?.initializedPromise) {
        window.Shiny.initializedPromise.then(() => {
            emitAllRoleMaps();
        });
    } else {
        // fallback
        setTimeout(emitAllRoleMaps, 0);
    }


    // Shiny?.addCustomMessageHandler?.("SortableRoles", function(msg) {
    //     console.info(`[pythagoras] 1 SortableRoles running`);
    //     msg = msg || {};
    //     const el = document.getElementById(msg.card);
    //     if (!el) {
    //         console.error("[pythagoras] Sortable element not found: ", msg.card)
    //         return;
    //     }
    //     console.info(`[pythagoras] 2 SortableRoles running`);
    //     const input = msg.input
    //     if (!input) {
    //         console.error("[pythagoras] Input not supplied: ")
    //         return;
    //     }
    //     function emitRoleMap() {
    //         const payload = {};
    //         el.querySelectorAll(".sortable-role").forEach((list) => {
    //             const role = list.dataset.role;
    //             payload[role] = Array.from(list.children).map(se => se.dataset.varname);
    //         });
    //         window.Shiny?.setInputValue?.(input,  payload, { priority: "event" });
    //     }
    //     console.info(`[pythagoras] 3 SortableRoles running`);
    //     el.querySelectorAll(".sortable-role").forEach((el) => {
    //         Sortable.create(el, {
    //             group: "variable-roles",
    //             animation: 150,
    //             ghostClass: "ghost",
    //             chosenClass: "chosen",
    //             onEnd: emitRoleMap
    //         });
    //     });
    //     console.info(`[pythagoras] 4 SortableRoles running`);
    //     emitRoleMap();
    // });

    // ---- helpers so clicks and keyboard use the same behavior ----
    const expandCard = (card, btn /* the .expand-btn that was clicked (optional) */) => {
        if (!card || card.classList.contains("fullscreen-active")) return;

        // Hide the expand button's wrapper
        btn?.parentElement?.classList.add("hidden");

        // Swap maximize → minimize (+ hide close while fullscreen)
        const ch = btn?.parentElement?.parentElement || card;
        ch?.querySelector(".contract-btn")?.parentElement?.classList.remove("hidden");
        const close = ch?.querySelector(".close-btn");
        close?.parentElement?.classList.add("hidden");

        card.classList.add("fullscreen-active");
        document.body.classList.add("fullscreen-mode");
        btn?.setAttribute?.("aria-expanded", "true");
        card.setAttribute("aria-modal", "true");
        window.Shiny?.setInputValue?.(`${card.id}_full_screen`, true, { priority: "event" });
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

    const contractCard = (card) => {
        if (!card || !card.classList.contains("fullscreen-active")) return;

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

    // ---- clicks use the helpers ----
    const expandButtons = document.querySelectorAll(".expand-btn");
    expandButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
        const card = btn.closest(".card");
        if (!card) { console.warn("[FullScreen] No card found"); return; }
        expandCard(card, btn);
        });
    });

    const contractButtons = document.querySelectorAll(".contract-btn");
    contractButtons.forEach((btn) => {
        // hide its wrapper initially (your original behavior)
        btn?.parentElement?.classList.add("hidden");
        btn.addEventListener("click", () => {
        const card = btn.closest(".card");
        if (!card) { console.warn("[FullScreen] No card found"); return; }
        contractCard(card);
        });
    });

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

    // Sortable behaviour for card drag and drop
    const container = document.getElementById("cards-container");
    console.info("[pythagoras] Sortable-Script running (for card drag and drop)")

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
            const ids = Array.from(container.children).map((x) => x.id).filter(Boolean);
            window.Shiny?.setInputValue?.("CardOrder", { ids, ts: Date.now() }, { priority: "event" });
        }
    });



    // // Sortable behaviour for Role Assignment
    // function emitRoleMap() {
    //     console.info(`Role Assignment running`)
    //     const payload = {};
    //     document.querySelectorAll(".sortable-role").forEach((list) => {
    //         const role = list.dataset.role;
    //         payload[role] = Array.from(list.children).map(el => el.dataset.varname);
    //     });
    //     window.Shiny?.setInputValue?.("RoleMapping-role_map", payload, { priority: "event" });
    // }

    // document.querySelectorAll(".sortable-role").forEach((el) => {
    //     console.info(`Found sortable-role ${el.id}`)
    //     Sortable.create(el, {
    //         group: "variable-roles",
    //         animation: 150,
    //         ghostClass: "ghost",
    //         chosenClass: "chosen",
    //         onEnd: emitRoleMap
    //     });
    // });

    // emitRoleMap();
});



// // RoleMap Widget
// (function() {
//     const UNASSIGNED_SENTINEL = "__unassigned__";

//     function initRoleMapWidget(root) {
//         const zones = root.querySelectorAll(".role-zone");
//         if (!zones.length) return;

//         const widgetId = root.id;  // e.g. "-RoleMapWidget"
//         const nsPrefix = widgetId.replace(/-RoleMapWidget$/, ""); // "config"

//         function collectColumnRoleMap() {
//         const mapping = {};

//         zones.forEach((zone) => {
//             const role = zone.dataset.role; // e.g. "target", "predictor", "__unassigned__"
//             // For the RoleMap, we either:
//             // - skip unassigned entirely, or
//             // - map to Role.NONE.value ("none")
//             const roleValue = (role === UNASSIGNED_SENTINEL) ? "none" : role;

//             const items = zone.querySelectorAll("[data-col]");
//             items.forEach((el) => {
//             const col = el.dataset.col;
//             // We enforce one role per column: singleton list
//             mapping[col] = [roleValue];
//             });
//         });

//         if (window.Shiny) {
//             const inputId = nsPrefix + "-RoleMap"; // matches ns("RoleMap") on Python side
//             window.Shiny.setInputValue(inputId, mapping, { priority: "event" });
//         }
//         }

//         zones.forEach((zone) => {
//         Sortable.create(zone, {
//             group: "varRoles",  // all connected
//             animation: 150,
//             onAdd: collectColumnRoleMap,
//             onUpdate: collectColumnRoleMap,
//             onRemove: collectColumnRoleMap,
//         });
//         });

//         // Push initial state
//         collectColumnRoleMap();
//     }

//     // When Shiny is connected, initialise for all cards present
//     document.addEventListener("shiny:connected", () => {
//         document
//         .querySelectorAll('[id$="-RoleMapWidget"]')
//         .forEach((root) => initRoleMapWidget(root));
//     });
//     })();









