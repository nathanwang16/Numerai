"""Weekly monitoring report.

For each configured model, pull round-by-round performance and write a
markdown summary to ``runs/reports/YYYY-WW.md``.

Uses the ``numerapi`` client (equivalent to the Numerai MCP tools
``get_model_performance``, ``get_current_round``, ``get_leaderboard``).

Usage:
    python scripts/weekly_report.py --models <model_id_1> <model_id_2>

Requires NUMERAI_PUBLIC_ID + NUMERAI_SECRET_KEY for auth'd queries.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd


def _napi():
    from numerapi import NumerAPI

    from numerai_stack.compute.credentials import get_credentials

    creds = get_credentials()
    return NumerAPI(public_id=creds.public_id, secret_key=creds.secret_key)


def fetch_performance(napi, model_id: str, last_n: int = 20) -> pd.DataFrame:
    query = """
      query($modelId: String!) {
        v3UserProfile(modelId: $modelId) {
          roundModelPerformances(last: %d) {
            roundNumber
            corr20V2
            mmc
            fncV3
            payoutPending
            payoutSettled
          }
        }
      }
    """ % last_n
    data = napi.raw_query(
        query, variables={"modelId": model_id}, authorization=True
    )
    rows = data["data"]["v3UserProfile"]["roundModelPerformances"]
    return pd.DataFrame(rows).sort_values("roundNumber", ascending=False)


def alert_negative_mmc(df: pd.DataFrame, k: int = 3) -> str | None:
    recent = df.head(k)
    if "mmc" in recent.columns and recent["mmc"].dropna().le(0).all() and len(recent.dropna(subset=["mmc"])) >= k:
        return f"WARNING: last {k} rounds had non-positive MMC"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="Numerai model UUIDs")
    ap.add_argument("--last-n", type=int, default=20)
    ap.add_argument("--out-dir", default="runs/reports")
    args = ap.parse_args()

    napi = _napi()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iso_week = datetime.utcnow().strftime("%Y-W%V")
    path = out_dir / f"{iso_week}.md"
    lines: list[str] = [f"# Weekly report -- {iso_week}", ""]
    for mid in args.models:
        try:
            df = fetch_performance(napi, mid, last_n=args.last_n)
        except Exception as e:
            lines.append(f"## model {mid}")
            lines.append(f"Error: {e}")
            continue
        lines.append(f"## model {mid}")
        if df.empty:
            lines.append("no performance data")
            continue
        mean = df[["corr20V2", "mmc", "fncV3"]].mean(numeric_only=True)
        lines.append(f"mean(CORR)={mean.get('corr20V2', float('nan')):.4f}  "
                     f"mean(MMC)={mean.get('mmc', float('nan')):.4f}  "
                     f"mean(FNC)={mean.get('fncV3', float('nan')):.4f}  "
                     f"rounds={len(df)}")
        alert = alert_negative_mmc(df, k=3)
        if alert:
            lines.append("")
            lines.append(f"> {alert}")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
