"""Project paths. Override the raw-data location with the environment variable MISSTATE_DATA."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("MISSTATE_DATA", ROOT / "data" / "raw"))
# Convenience: if the repository sits inside the folder that holds the raw data, use that folder.
if "MISSTATE_DATA" not in os.environ and not any(DATA_RAW.glob("data_FraudDetection_JAR2020.csv*")) \
        and any(ROOT.parent.glob("data_FraudDetection_JAR2020.csv*")):
    DATA_RAW = ROOT.parent
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"

for p in (DATA_INTERIM, DATA_PROCESSED, TABLES, FIGURES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
BAO_TEST_YEARS = list(range(2003, 2009))       # Bao et al. (2020) test window
LATE_TEST_YEARS = list(range(2009, 2015))      # label-truncated years in the Bao data
GAP = 2                                        # years between last training year and test year
