"""Generate the downscaled figures embedded in the README.

The README displays `docs/*.png`, which are LANCZOS-downscaled copies of
the full-resolution figures under `analysis/figures/maxcut/`. Running the
comparison only writes the full-resolution files, so this script keeps the
README copies in sync.

Run: python3 experiments/make_docs_figures.py
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent.parent
SRC = ROOT / "analysis" / "figures" / "maxcut"
DST = ROOT / "docs"
SIZES = {
    "win_rate_heatmap": (896, 560),
    "runtime_comparison": (896, 560),
    "relative_performance": (1100, 500),
}


def main():
    DST.mkdir(exist_ok=True)
    for name, size in SIZES.items():
        src = SRC / f"{name}.png"
        if not src.exists():
            print(f"skip {name}: {src} not found")
            continue
        Image.open(src).convert("RGB").resize(size, Image.LANCZOS).save(DST / f"{name}.png")
        print(f"wrote docs/{name}.png {size}")


if __name__ == "__main__":
    main()
