"""No-op TensorBoard writer used as a safe default.

This avoids creating implicit ``./runs`` folders when callers intend to
inject their own SummaryWriter with an explicit log_dir.
"""


class NoOpSummaryWriter:
    """Drop-in no-op replacement for SummaryWriter."""

    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None

