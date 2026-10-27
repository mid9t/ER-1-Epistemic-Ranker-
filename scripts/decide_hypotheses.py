#!/usr/bin/env python
"""Apply the pre-registered decision rules and write verdicts.md.

The verdict column follows PREREGISTRATION.md section 2.3. Holm-adjusted
p-values for H1, H3, and the positive result are reported beside it. H2 is a
bound check and is left out of the Holm correction. Quote the combined
interval, not the utterance-only interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bert_daedl.posthoc.report import write_json, write_md
from bert_daedl.posthoc.stats import holm, one_sided_p

H1_ROW = "H1: CE energy − EDL vacuity"
H2_ROW = "H2: qsigmoid×mul − vacuity"
POS_VAC = "Positive: primary pick − vacuity"
POS_DEN = "Positive: primary pick − density"
H2_SPEC = "daedl:qsigmoid:mul/vacuity"
TIE_BAND = 0.002
H2_BOUND = 0.002


def verdict_h1(ci95_low, ci95_high, ci90_low, ci90_high) -> str:
    if ci95_low > 0:
        return "supported"
    if ci90_low >= -TIE_BAND and ci90_high <= TIE_BAND:
        return "tie"
    if ci95_high < 0:
        return "reversed"
    return "inconclusive"


def verdict_h2(ci95_low, ci95_high, n_worse, n_seeds) -> str:
    if ci95_high < H2_BOUND and n_worse == n_seeds == 5:
        return "supported"
    if ci95_low > 0:
        return "refuted"
    return "inconclusive"


def verdict_h3(ci95_low, ci95_high) -> str:
    if ci95_low > 0:
        return "supported"
    if ci95_high < 0:
        return "refuted"
    return "inconclusive"


def verdict_positive(low_vs_vacuity, low_vs_density) -> str:
    if low_vs_vacuity > 0 and low_vs_density > 0:
        return "supported"
    return "kill"


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row(rows, comparison: str) -> dict:
    for row in rows:
        if row.get("comparison") == comparison:
            return row
    raise SystemExit(f"missing comparison {comparison!r}")


def _f(row, key) -> float:
    return float(row[key])


def _interval(row) -> dict:
    return {
        "delta": _f(row, "delta"),
        "ci95": [_f(row, "combined_ci95_low"), _f(row, "combined_ci95_high")],
        "ci90": [_f(row, "combined_ci90_low"), _f(row, "combined_ci90_high")],
        "se_combined": _f(row, "se_combined"),
    }


def _ece_worse(grid_rows) -> tuple[int, int]:
    hits = []
    for row in grid_rows:
        spec = row.get("config") or row.get("spec")
        if spec != H2_SPEC:
            continue
        if row.get("fit_on") not in (None, "", "train"):
            continue
        hits.append(row)
    by_seed = {}
    for row in hits:
        by_seed[row.get("seed", "")] = row
    n_worse = 0
    for row in by_seed.values():
        if float(row["test_ece_scaled"]) > float(row["test_ece_plain"]):
            n_worse += 1
    return n_worse, len(by_seed)


def _lower_bound(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("gain_is_lower_bound"))


def decide(root: Path) -> dict:
    root = Path(root)
    post = root / "posthoc"
    full = _read_csv(post / "all" / "bootstrap_full.csv")
    port = _read_csv(post / "all" / "bootstrap_port.csv")
    h1 = _row(full, H1_ROW)
    h2_full = _row(full, H2_ROW)
    h2_port = _row(port, H2_ROW)
    pos_v = _row(full, POS_VAC)
    pos_d = _row(full, POS_DEN)
    worse_full, n_full = _ece_worse(_read_csv(post / "all" / "grid_full" / "grid_runs.csv"))
    worse_port, n_port = _ece_worse(_read_csv(post / "all" / "grid_port" / "grid_runs.csv"))

    claims = {
        "H1": {
            "claim": "H1: score matching replicates",
            "comparison": "CE energy − EDL vacuity",
            "verdict": verdict_h1(
                _f(h1, "combined_ci95_low"),
                _f(h1, "combined_ci95_high"),
                _f(h1, "combined_ci90_low"),
                _f(h1, "combined_ci90_high"),
            ),
            **_interval(h1),
            "source": "posthoc/all/bootstrap_full.csv",
        },
        "H2_full": {
            "claim": "H2: multiplicative coupling gives no benefit (D_full)",
            "comparison": "qsigmoid×mul − vacuity",
            "verdict": verdict_h2(
                _f(h2_full, "combined_ci95_low"),
                _f(h2_full, "combined_ci95_high"),
                worse_full,
                n_full,
            ),
            **_interval(h2_full),
            "ece_worse_seeds": worse_full,
            "n_seeds": n_full,
            "source": "posthoc/all/bootstrap_full.csv",
        },
        "H2_port": {
            "claim": "H2: multiplicative coupling gives no benefit (D_port)",
            "comparison": "qsigmoid×mul − vacuity",
            "verdict": verdict_h2(
                _f(h2_port, "combined_ci95_low"),
                _f(h2_port, "combined_ci95_high"),
                worse_port,
                n_port,
            ),
            **_interval(h2_port),
            "ece_worse_seeds": worse_port,
            "n_seeds": n_port,
            "source": "posthoc/all/bootstrap_port.csv",
        },
        "positive": {
            "claim": "Positive coupling result",
            "comparison": "primary pick − vacuity and primary pick − density",
            "verdict": verdict_positive(
                _f(pos_v, "combined_ci95_low"),
                _f(pos_d, "combined_ci95_low"),
            ),
            "vs_vacuity": _interval(pos_v),
            "vs_density": _interval(pos_d),
            "source": "posthoc/all/bootstrap_full.csv",
        },
    }

    h3_path = post / "h3_did.csv"
    if h3_path.is_file():
        h3 = _read_csv(h3_path)[0]
        claims["H3"] = {
            "claim": "H3: the density helps more on near-OOS",
            "comparison": "DiD = (A − vacuity on ID-OOS) − (A − vacuity on OOD-OOS)",
            "verdict": verdict_h3(_f(h3, "combined_ci95_low"), _f(h3, "combined_ci95_high")),
            "a": _f(h3, "a"),
            "b": _f(h3, "b"),
            **_interval(h3),
            "source": "posthoc/h3_did.csv",
        }
    else:
        claims["H3"] = {
            "claim": "H3: the density helps more on near-OOS",
            "verdict": "not_applicable",
            "note": "This study has one undivided OOS set, so there is no difference in differences.",
        }

    p_values = {
        "H1": one_sided_p(claims["H1"]["delta"], claims["H1"]["se_combined"]),
        "positive": max(
            one_sided_p(claims["positive"]["vs_vacuity"]["delta"], claims["positive"]["vs_vacuity"]["se_combined"]),
            one_sided_p(claims["positive"]["vs_density"]["delta"], claims["positive"]["vs_density"]["se_combined"]),
        ),
    }
    if claims["H3"]["verdict"] != "not_applicable":
        p_values["H3"] = one_sided_p(claims["H3"]["delta"], claims["H3"]["se_combined"])

    notes = []
    if _lower_bound(post / "all" / "grid_full" / "selected.json"):
        notes.append("The primary D_full pick is still on the λ-grid edge after the one allowed extension. Its gain is a lower bound.")
    if _lower_bound(post / "all" / "grid_full" / "selected_secondary.json"):
        notes.append("The secondary D_full pick is still on the λ-grid edge after the one allowed extension. Its gain is a lower bound.")
    if _lower_bound(post / "all" / "grid_port" / "selected.json"):
        notes.append("The primary D_port pick is still on the λ-grid edge after the one allowed extension. Its gain is a lower bound.")

    return {
        "claims": claims,
        "holm": holm(p_values),
        "notes": notes,
        "rule": "Verdicts follow PREREGISTRATION.md section 2.3. Holm covers H1, H3, and the positive result only.",
    }


def _fmt_ci(pair) -> str:
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def render_md(result: dict) -> str:
    lines = [
        "# Verdicts",
        "",
        "Decided from the combined interval, SE = √(SE²_boot + s²_seed / n). "
        "The verdict column is the rule in the pre-registration. "
        "Holm-adjusted one-sided p-values are reported for the three directional claims and do not replace that rule. "
        "H2 is a bound check and is not in the Holm correction.",
        "",
        "| Claim | Verdict | Estimate | Combined 95% CI |",
        "|---|---|---:|---|",
    ]
    order = ("H1", "H2_full", "H2_port", "H3", "positive")
    for key in order:
        claim = result["claims"][key]
        if claim["verdict"] == "not_applicable":
            lines.append(f"| {claim['claim']} | not applicable | — | — |")
            continue
        if key == "positive":
            est = claim["vs_vacuity"]["delta"]
            ci = _fmt_ci(claim["vs_vacuity"]["ci95"]) + "; density " + _fmt_ci(claim["vs_density"]["ci95"])
        elif key == "H3":
            est = claim["delta"]
            ci = _fmt_ci(claim["ci95"]) + f"; a {claim['a']:+.4f}, b {claim['b']:+.4f}"
        else:
            est = claim["delta"]
            ci = _fmt_ci(claim["ci95"])
        lines.append(f"| {claim['claim']} | {claim['verdict']} | {est:+.4f} | {ci} |")
    lines += ["", "## Holm", "", "| Claim | one-sided p | Holm p | reject at 0.05 |", "|---|---:|---:|---|"]
    for name, item in result["holm"].items():
        lines.append(
            f"| {name} | {item['p']:.4g} | {item['p_holm']:.4g} | {'yes' if item['reject'] else 'no'} |"
        )
    if result["notes"]:
        lines += ["", "## Notes"]
        for note in result["notes"]:
            lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True, help="Directory for verdicts.md and verdicts.json.")
    args = p.parse_args()
    result = decide(Path(args.root))
    out = Path(args.out)
    write_json(out / "verdicts.json", result)
    text = render_md(result)
    write_md(out / "verdicts.md", text)
    print(text)


if __name__ == "__main__":
    main()
