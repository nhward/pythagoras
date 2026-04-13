console.log("[clipboard] Script running")

document.addEventListener("shown.bs.modal", function (e) {
    const modal = e.target;
    const button = modal.querySelector(".clipboard-btn");
    const codeEl = modal.querySelector(".clipboard-text");
    if (!button || !codeEl) return;

    button.addEventListener("click", () => {
        console.log("[clipboard] Click event")
        navigator.clipboard.writeText(codeEl.innerText.trim())
            .then(() => console.log("[clipboard] Copied from modal"))
            .catch(err => console.error("[clipboard] Copy failed", err));
    });
});