import Shepherd from "./shepherd-15.3.0.mjs";

console.log("[guide] module running");

function runTour(steps, onFinish = () => {}) {
    console.log("[guide] create_run_tour running");

    if (window.myShepherdTour && window.myShepherdTour.isActive()) return;

    const tour = new Shepherd.Tour({
        useModalOverlay: true,
        defaultStepOptions: {
            scrollTo: true,
            canClickTarget: true,
            arrow: true,   // make sure arrows are drawn
        }
    });

    steps.forEach(step => {
        const el = document.querySelector(step.selector);
        // Skip if element doesn’t exist
        if (!el) {
            //console.warn(`[guide] ${step.id}: Skipping step - element not available using selector "${step.selector}"`);
            return;
        }
        function isVisibleInLayout(el) {
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
        }
        //console.info(`[guide] ${step.id}: Creating step`);
        tour.addStep({
            id: step.id,
            arrow: { padding: 20 },
            text: step.text,
            title: step.title,
            attachTo: { element: step.selector, on: step.position },
            buttons: [
                { text: "Back", action: tour.back },
                { text: "Next", action: tour.next },
                { text: "Exit", action: tour.cancel }
            ],
            beforeShowPromise: () => {
                return new Promise(resolve => {
                    const el = document.querySelector(step.selector);
                    if (!el) {
                        //console.error(`[guide] Step ${step.id}: element not found in "beforeShowPromise"`);
                        return resolve();
                    }
                    //console.info(`[guide] ${step.id}: Located element`);
                    const card = el.closest(".card")
                    const sidebar = card?.querySelector(".sidebar")

                    // === 0. FULL SCREEN HANDLING ===
                    if (card) {
                        if (card?.classList.contains("fullscreen-active")) {
                            const isExpand = el.querySelector(".expand-btn")
                            // if (isExpand) console.log("[guide] Is expand button")
                            const contract = card.querySelector(".contract-btn")
                            if (isExpand && contract) contract.click()
                            const isClose = el.querySelector(".close-btn")
                            if (isClose && contract) contract.click()
                        } else {
                            const isContract = el.querySelector(".contract-btn")
                            // if (isContract) console.log("[guide] Is contract button")
                            const expand = card.querySelector(".expand-btn")
                            if (isContract && expand) expand.click()
                        }
                    }
                    // // Ensure element is visible (basic safeguard)
                    // if (el.classList.contains("hidden")) {
                    //     el.classList.remove("hidden")
                    // }
                    el.scrollIntoView({ behavior: "smooth", block: "center" });

                    // === 1. TAB HANDLING ===
                    const pane = el.closest(".tab-pane");
                    if (pane && pane.id) {
                        const tabToggle = document.querySelector(`[data-bs-toggle="tab"][href="#${pane.id}"]`);
                        if (tabToggle) {
                            //console.info(`[guide] ${step.id}: Activating tab for pane #${pane.id}`);
                            const tab = new bootstrap.Tab(tabToggle);
                            tab.show();
                        } else {
                            //console.warn(`[guide] ${step.id}: No tab-toggle found for #${pane.id}`);
                        }
                    }

                    // === 2. FLIP HANDLING ===
                    const back = el.closest(".back");
                    if (back) {
                        const container = back.closest(".flip-container");
                        if (container && !container.classList.contains("flipped")) {
                            const flip = card?.querySelector(".flip-btn")
                            if (!flip) {
                                console.warn("[guide] Flip btn not found");
                            } else {
                                flip.click()
                                console.info("[guide] Flip btn used for Front");
                            }
                        }
                    } else {
                        const front = el.closest(".front");
                        if (front) {
                            //console.info("[guide] ${step.id}: In on the 'front'");
                            const container = front.closest(".flip-container");
                            if (container && container.classList.contains("flipped")) {
                                const flip = card?.querySelector(".flip-btn")
                                if (!flip) {
                                    console.warn("[guide] Flip btn not found");
                                } else {
                                    flip.click()
                                    console.info("[guide] Flip btn used for Back");
                                }
                            }
                            if (sidebar && isVisibleInLayout(sidebar)) {
                                const toggleBtn = card.querySelector(".collapse-toggle");
                                if (toggleBtn) {
                                    toggleBtn.click();
                                }
                            }
                        }
                    }

                    // === 3. SIDEBAR HANDLING  ===
                    const inSidebar = el.closest(".sidebar")
                    if (card && inSidebar && sidebar && !isVisibleInLayout(sidebar)) {
                        const toggleBtn = card.querySelector(".collapse-toggle");
                        if (toggleBtn) {
                            // console.log(`[guide] Toggle button found`);
                            toggleBtn.click();
                        }
                    }
                    resolve();
                });
            }

        });
    });

    const first = document.querySelector(steps[0]?.selector)
    if (!first) {
        console.error("[guide] No usable first step");
        return;
    }
    const card = first.closest(".card");


    function reset(card) { 
        window.myShepherdTour = null;
        onFinish();
        if (!card) return;
        card.classList.remove("tour-running");
        const container = card.querySelector(".flip-container")
        console.info("[guide] Reset called")
        if (container && container.classList.contains("flipped")) {
            const flip = card?.querySelector(".flip-btn")
            if (flip) {
                flip.click()
            }
        }
        // const sb = card.querySelector(".sidebar")
        // if (sb && isVisibleInLayout(sb)) {
        //     const toggle = card.querySelector(".collapse-toggle");
        //     if (toggle) {
        //         toggle.click();
        //     }
        // }
    };

    tour.on("complete", () => reset(card));
    tour.on("cancel", ()   => reset(card));
    tour.on("start", ()    => { 
        window.myShepherdTour = tour; 
        // console.log(`[guide] class tour-running added to card`);
        card?.classList.add("tour-running");
    });

    tour.start();
}

Shiny.addCustomMessageHandler("create_run_tour", function(json_steps) {
    runTour(JSON.parse(json_steps));
});

// Resolve navigation targets when the tour starts, including newly added sections.
document.addEventListener("click", async (event) => {
    if (!event.target.closest("#ApplicationGuide, #GuideButton")) return;
    if (window.myShepherdTour?.isActive()) return;
    const accordion = document.querySelector("#Accordion");
    if (accordion) {
        await Promise.all(Array.from(accordion.querySelectorAll(".accordion-collapse"), (panel) =>
            new Promise(resolve => {
                function close() {
                    if (!panel.classList.contains("show")) return resolve();
                    panel.addEventListener("hidden.bs.collapse", resolve, { once: true });
                    bootstrap.Collapse.getOrCreateInstance(panel, { toggle: false }).hide();
                }
                if (panel.classList.contains("collapsing")) {
                    panel.addEventListener("shown.bs.collapse", close, { once: true });
                    panel.addEventListener("hidden.bs.collapse", resolve, { once: true });
                } else {
                    close();
                }
            })
        ));
    }

    function finishNavigation() {
        if (accordion) {
            // Start is only rendered when show_start is enabled.
            const start = accordion.querySelector('[data-value="start"] .accordion-collapse');
            if (start) bootstrap.Collapse.getOrCreateInstance(start, { toggle: false }).show();
        } else {
            const firstTab = document.querySelector("#Navbar .nav-link[data-value]");
            if (firstTab) bootstrap.Tab.getOrCreateInstance(firstTab).show();
        }
    }

    const steps = [];
    function add(selector, title, text) {
        if (!document.querySelector(selector)) return;
        steps.push({ id: `application-${steps.length}`, selector, title, text, position: "bottom" });
    }

    add(accordion ? "#Accordion" : "#Navbar", "Sections",
        accordion
            ? "Work through the sections in order, expanding each panel to view and work with its cards."
            : "Work through the sections in order, using the tabs to view and work with each section's cards.");
    add("#ApplicationGuide", "Application guide",
        "Explore the application controls and sections. Each card also has its own guide. <p>A convenient to navigate the guide is with the ⬅️ and ➡️ arrow keys. The  ␛ key ends the tour.</p>");
    add("#ManageCardSection", "Manage cards and sections",
        "Add cards and sections, or change application settings here.");
    add("#SaveConfiguration", "Bookmarks",
        "Save your configuration or restore a previously saved configuration.");
    add("#AppInfo", "About Pythagoras", "View information about the application.");
    add("#FullScreen", "Full screen", "Toggle full screen to give your work more space.");
    add("#Quit", "Quit session", "Use this control when you are ready to end your session.");
    runTour(steps, finishNavigation);
});
