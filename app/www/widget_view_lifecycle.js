// Keep the provider's one widget manager and patch only model teardown.
// Jupyter's close() deletes model.views before its pending view removals run.
// Their callbacks still need that registry (and each view's model reference).
require(["@jupyter-widgets/base"], ({WidgetModel}) => {
    const prototype = WidgetModel.prototype;
    if (prototype.pythagorasTeardownGuard) return;
    const close = prototype.close;
    const closing = new WeakMap();
    prototype.close = function (...args) {
        if (closing.has(this)) return closing.get(this);
        const operation = (async () => {
            const views = this.views;
            await Promise.all(Object.values(views || {}).map(async (pending) => {
                const view = await pending;
                // The affected bridge can remove this reference too early.
                if (!view.model) view.model = this;
            }));
            try {
                const pending = close.apply(this, args);
                // Restore synchronously, before asynchronous remove callbacks.
                if (!this.views && views) this.views = views;
                await pending;
            } finally {
                if (this._closed) delete this.views;
            }
        })();
        closing.set(this, operation);
        return operation;
    };
    prototype.pythagorasTeardownGuard = true;
});
