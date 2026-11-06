"""Remove exact title duplicates from raw_records.csv (titles compared lower-case, letters and digits only)."""
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent
r = pd.read_csv(HERE / "raw_records.csv").fillna("")
r["k"] = r.title.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
g = r.groupby("k").agg(
    title=("title", "first"),
    year=("year", "max"),
    venue=("venue", lambda s: " | ".join(sorted({v for v in s if v}))[:70]),
    doi=("doi", lambda s: next((d for d in s if d and "arxiv" not in d), "")),
    src=("source", lambda s: "".join(sorted({x[0] for x in s}))),
    q=("query", lambda s: "".join(sorted({x[1] for x in s}))),
    abstract=("abstract", lambda s: max(s, key=len)),
).reset_index(drop=True)
g.to_csv(HERE / "dedup_records.csv", index=False)
print(len(r), "records,", len(g), "unique")
