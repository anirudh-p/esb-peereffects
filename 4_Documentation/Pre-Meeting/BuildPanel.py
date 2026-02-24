from __future__ import annotations

from pathlib import Path
import argparse
from typing import Optional, List
import re

import pandas as pd


QUARTER_RE = re.compile(r"^(?P<year>\d{4})\s*[Qq](?P<q>[1-4])$")


def parse_quarter(qstr: Optional[str]) -> Optional[pd.Period]:
    if not isinstance(qstr, str) or not qstr.strip():
        return None
    s = qstr.strip()
    m = QUARTER_RE.match(s)
    if not m:
        return None
    year = int(m.group("year"))
    q = int(m.group("q"))
    try:
        return pd.Period(f"{year}Q{q}", freq="Q")
    except Exception:
        return None


def period_to_str(p: Optional[pd.Period]) -> Optional[str]:
    if p is None:
        return None
    return f"{p.year}Q{p.quarter}"


def max_stage_for_quarter(q: pd.Period,
                          interest: Optional[pd.Period],
                          applied: Optional[pd.Period],
                          awarded: Optional[pd.Period],
                          order: Optional[pd.Period],
                          delivered: Optional[pd.Period],
                          operating: Optional[pd.Period]) -> str:
    # ordered progression by seniority
    stages = [
        (operating, "operating"),
        (delivered, "delivered"),
        (order, "ordered"),
        (awarded, "awarded"),
        (applied, "applied"),
        (interest, "interested"),
    ]
    for p, name in stages:
        if isinstance(p, pd.Period) and p <= q:
            return name
    return "none"


def build_panel(lea_stage_path: Path,
                peer_vars_path: Optional[Path] = None,
                lea_col: str = "1c. LEA ID",
                out_path: Path = Path("lea_quarter_panel.csv")) -> Path:
    stage_df = pd.read_csv(lea_stage_path)
    # Normalize column names (strip)
    stage_df.columns = [c.strip() if isinstance(c, str) else c for c in stage_df.columns]

    # Parse event quarters into Periods for comparison
    def pcol(col: str) -> List[Optional[pd.Period]]:
        if col in stage_df.columns:
            return [parse_quarter(x) for x in stage_df[col]]
        return [None] * len(stage_df)

    interest_p = pcol("interest_q")
    applied_p = pcol("applied_q")
    awarded_p = pcol("awarded_q")
    order_p = pcol("order_q")
    delivered_p = pcol("delivered_q")
    operating_p = pcol("operating_q")

    # Compute global quarter span
    all_ps = []
    for seq in (interest_p, applied_p, awarded_p, order_p, delivered_p, operating_p):
        all_ps.extend([p for p in seq if isinstance(p, pd.Period)])
    if not all_ps:
        raise ValueError("No valid stage quarters found in input; cannot build panel.")
    qmin = min(all_ps)
    qmax = max(all_ps)
    qrange = pd.period_range(start=qmin, end=qmax, freq="Q")

    # Base (LEA x Quarter) grid
    lea_ids = stage_df[lea_col].tolist()
    base = (
        pd.MultiIndex.from_product([lea_ids, qrange], names=[lea_col, "quarter"]).to_frame(index=False)
    )

    # Attach event quarters back to each LEA
    events = stage_df[[lea_col]].copy()
    events["interest_p"] = interest_p
    events["applied_p"] = applied_p
    events["awarded_p"] = awarded_p
    events["order_p"] = order_p
    events["delivered_p"] = delivered_p
    events["operating_p"] = operating_p

    panel = base.merge(events, on=lea_col, how="left")

    # Stage flags and durations since
    def ge_flag(q: pd.Period, p: Optional[pd.Period]) -> int:
        return int(isinstance(p, pd.Period) and q >= p)

    def since(q: pd.Period, p: Optional[pd.Period]) -> Optional[int]:
        if isinstance(p, pd.Period) and q >= p:
            return int((q.year - p.year) * 4 + (q.quarter - p.quarter))
        return None

    flags = {
        "interested": [],
        "applied": [],
        "awarded": [],
        "ordered": [],
        "delivered": [],
        "operating": [],
        "quarters_since_interest": [],
        "quarters_since_applied": [],
        "quarters_since_awarded": [],
        "quarters_since_operating": [],
        "stage_max": [],
    }

    for _, r in panel.iterrows():
        q = r["quarter"]
        ip, ap, aw, op, dp, gp = r["interest_p"], r["applied_p"], r["awarded_p"], r["order_p"], r["delivered_p"], r["operating_p"]
        flags["interested"].append(ge_flag(q, ip))
        flags["applied"].append(ge_flag(q, ap))
        flags["awarded"].append(ge_flag(q, aw))
        flags["ordered"].append(ge_flag(q, op))
        flags["delivered"].append(ge_flag(q, dp))
        flags["operating"].append(ge_flag(q, gp))
        flags["quarters_since_interest"].append(since(q, ip))
        flags["quarters_since_applied"].append(since(q, ap))
        flags["quarters_since_awarded"].append(since(q, aw))
        flags["quarters_since_operating"].append(since(q, gp))
        flags["stage_max"].append(max_stage_for_quarter(q, ip, ap, aw, op, dp, gp))

    for k, v in flags.items():
        panel[k] = v

    # Store quarter as string
    panel["quarter"] = panel["quarter"].apply(period_to_str)

    # Attach static attributes / peer metrics if provided
    if peer_vars_path and Path(peer_vars_path).exists():
        peer_df = pd.read_csv(peer_vars_path)
        peer_df.columns = [c.strip() if isinstance(c, str) else c for c in peer_df.columns]
        # Avoid duplicate columns on merge
        dup_cols = [c for c in peer_df.columns if c in panel.columns and c != lea_col]
        peer_use = peer_df.drop(columns=dup_cols)
        panel = panel.merge(peer_use, on=lea_col, how="left")

    # Reorder columns: keys, time, flags, durations, then static
    front = [lea_col, "quarter", "stage_max",
             "interested", "applied", "awarded", "ordered", "delivered", "operating",
             "quarters_since_interest", "quarters_since_applied", "quarters_since_awarded", "quarters_since_operating"]
    other_cols = [c for c in panel.columns if c not in set(front + [
        "interest_p", "applied_p", "awarded_p", "order_p", "delivered_p", "operating_p"
    ])]
    panel = panel[front + other_cols]

    panel.to_csv(out_path, index=False)
    print(f"Wrote panel: {len(panel):,} rows, {panel[lea_col].nunique():,} LEAs, {panel['quarter'].nunique():,} quarters -> {out_path.resolve()}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build LEA-by-quarter panel from imputed stage timings and static attributes.")
    ap.add_argument("--stages", type=Path, default=Path("lea_stage_imputation.csv"))
    ap.add_argument("--peer-vars", type=Path, default=Path("lea_peer_vars.csv"))
    ap.add_argument("--lea-col", default="1c. LEA ID")
    ap.add_argument("--out", type=Path, default=Path("lea_quarter_panel.csv"))
    args = ap.parse_args()

    build_panel(args.stages, args.peer_vars, args.lea_col, args.out)


if __name__ == "__main__":
    main()
