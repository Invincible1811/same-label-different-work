"""Shared figure style for all RQ scripts (static figures for a printed report).

Colours follow a validated categorical palette (adjacent CVD ΔE >= 9.1, checked
with the dataviz validator): identity colours in fixed slot order, a single-hue
blue ramp for magnitude, and a blue<->red diverging ramp with a grey midpoint.
Grey is reserved for 'baseline / other' categories."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"  # baseline / "other" category

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a5a4", "#e34948", "#a32524"]
)

# display names for figure labels only (data files keep the dataset's agent_type codes)
AGENT_LABEL = {
    "Claude_Code": "Claude Code",
    "SWE_agent": "SWE-agent",
    "Gemini_Code_Assist": "Gemini Code Assist",
    "Amazon_Q": "Amazon Q",
    "PR_Agent": "PR-Agent",
}


def name(agent):
    """Readable agent name for a figure label; keeps a trailing marker such as the dagger."""
    base = agent.rstrip("†")
    return AGENT_LABEL.get(base, base) + agent[len(base) :]


def apply():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "xtick.color": MUTED,
            "ytick.color": INK_2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 200,
            "pdf.fonttype": 42,
        }
    )


def clean_axes(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.set_axisbelow(True)


def text_color_on(fill):
    r, g, b = to_rgb(fill)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else "#ffffff"


def save(fig, path_without_ext):
    fig.savefig(
        f"{path_without_ext}.pdf", metadata={"CreationDate": None}
    )  # no timestamp: reproducible bytes
    fig.savefig(f"{path_without_ext}.png")
    plt.close(fig)
