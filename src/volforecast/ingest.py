"""Pull a point-in-time S&P 500 daily panel from CRSP (via WRDS).

Ported from the team's data-acquisition notebook so it can run as a pipeline step.

Usage (needs a WRDS account; the password is prompted for, or read from ~/.pgpass):

    pip install -e ".[ingest]"
    python -m volforecast.ingest --username <wrds_user> --out data/raw/sp500_panel.parquet

CRSP data is licensed: never commit the output file (data/ is git-ignored).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from volforecast import config

MEMBER_TABLE = "crsp.dsp500list_v2"
DAILY_TABLE = "crsp.dsf_v2"


def flag_membership(panel: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: was the stock an S&P 500 member on that date?

    ``members`` has one row per membership spell with columns permno, start, ending.
    Both endpoints are inclusive, and a stock can have several spells.
    """
    pairs = panel[["permno", "date"]].reset_index()
    spells = pairs.merge(members[["permno", "start", "ending"]], on="permno", how="inner")
    inside = (spells["date"] >= spells["start"]) & (spells["date"] <= spells["ending"])
    member_idx = spells.loc[inside, "index"].unique()
    return pd.Series(panel.index.isin(member_idx), index=panel.index, name="isin_sp500")


def fetch_members(db, start: str = config.START, end: str = config.END) -> pd.DataFrame:
    """All membership spells that overlap [start, end]."""
    members = db.raw_sql(
        f"SELECT permno, mbrstartdt AS start, mbrenddt AS ending FROM {MEMBER_TABLE}",
        date_cols=["start", "ending"],
    )
    # Current members have no end date yet
    members["ending"] = members["ending"].fillna(pd.Timestamp(end))
    overlaps = (members["ending"] >= pd.Timestamp(start)) & (members["start"] <= pd.Timestamp(end))
    return members.loc[overlaps].reset_index(drop=True)


def fetch_daily(db, permnos: list[int], start: str, end: str) -> pd.DataFrame:
    """Daily return / price / volume for the given PERMNOs, queried one year at a time."""
    permno_list = ",".join(str(int(p)) for p in permnos)
    chunks = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        chunks.append(
            db.raw_sql(
                f"""
                SELECT permno, dlycaldt AS date, dlyret AS return,
                       dlyprc AS price, dlyvol AS volume
                FROM {DAILY_TABLE}
                WHERE permno IN ({permno_list})
                  AND dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31'
                """,
                date_cols=["date"],
            )
        )
    return pd.concat(chunks, ignore_index=True)


def build_panel(daily: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Clean types, add the point-in-time membership flag and log returns."""
    panel = daily.copy()
    for col in ["return", "price", "volume"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").astype(float)
    panel["permno"] = panel["permno"].astype(int)
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    panel["isin_sp500"] = flag_membership(panel, members)
    panel["log_return"] = np.log1p(panel["return"])
    return panel


def fetch_sp500_panel(
    username: str, start: str = config.START, end: str = config.END
) -> pd.DataFrame:
    import wrds  # optional dependency, only needed for ingestion

    db = wrds.Connection(wrds_username=username)
    try:
        members = fetch_members(db, start, end)
        daily = fetch_daily(db, members["permno"].unique().tolist(), start, end)
    finally:
        db.close()
    return build_panel(daily, members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", default=os.getenv("WRDS_USERNAME"))
    parser.add_argument("--out", default=config.DATA_PATH)
    parser.add_argument("--start", default=config.START)
    parser.add_argument("--end", default=config.END)
    args = parser.parse_args()
    if not args.username:
        parser.error("Pass --username or set WRDS_USERNAME")

    panel = fetch_sp500_panel(args.username, args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, index=False)
    print(
        f"Saved {len(panel):,} rows, {panel['permno'].nunique()} stocks "
        f"({panel['isin_sp500'].mean():.1%} member-days) -> {out}"
    )


if __name__ == "__main__":
    main()
