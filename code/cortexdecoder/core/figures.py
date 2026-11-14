import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DPI = 600
FIG_DPI_RANGE = (600, 1200)


def save_figure(fig, path, dpi=FIG_DPI, **kwargs):
    """Save and close `fig`. Every figure the pipeline writes goes through here so DPI stays in FIG_DPI_RANGE."""
    lo, hi = FIG_DPI_RANGE
    if not lo <= dpi <= hi:
        raise ValueError(f"figure dpi must be in [{lo}, {hi}], got {dpi}")
    fig.savefig(path, dpi=dpi, **kwargs)
    plt.close(fig)
    return path
