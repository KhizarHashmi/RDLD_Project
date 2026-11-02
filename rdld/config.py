"""Paths shared by every script on every branch.

Data and results never live in git. They live in one folder outside the repo
(default: <repo>/../RDLD_data) so that both the khizar and alishba worktrees
read and write the same files. Override with the RDLD_DATA environment variable.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("RDLD_DATA", REPO.parent / "RDLD_data")).resolve()

DOCLAYNET = DATA_ROOT / "doclaynet"            # clean 300-page corpus (P0)
IMAGES = DOCLAYNET / "images"
ANNOTATIONS = DOCLAYNET / "annotations.json"   # COCO, 11 DocLayNet classes
MANIFEST = DOCLAYNET / "manifest.csv"          # per-page metadata
TEST_PAGE = DATA_ROOT / "test_page.png"        # single page used by feasibility gates

DEGRADED = DATA_ROOT / "degraded"              # perturbation cache (P2, Alishba)
PREDICTIONS = DATA_ROOT / "predictions"        # detector output cache (P1+, Khizar)
RESULTS = DATA_ROOT / "results"                # metrics tables, figures

THIRD_PARTY = REPO / "third_party"


def out_dir(*parts):
    """Return RESULTS/<parts...>, creating it."""
    p = RESULTS.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
