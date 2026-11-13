"""Aggregate the final evaluation into results/final_eval/ (aggregates only; row-level outputs stay in runs/).

  python final_eval/summarize.py            # reads runs/final_eval/{trace_val,mmlu_pro} and runs/full (references)

Writes results/final_eval/trace_val_summary.json, mmlu_pro_summary.json and tables.md, and copies each
TRACE-val report.md. Statistics:
  * TRACE val: per-seed official accuracy; mean and sample SD over the 3 seeds; seeds-pooled accuracy (each
    row's correctness averaged over its seeds) with a 95% bootstrap CI over the 2,000 rows (10,000 resamples,
    fixed RNG seed 0); paired differences between models on the same rows, same bootstrap.
  * MMLU-Pro: the official accuracy (random letter for an unextractable answer, official seed and order) and
    strict accuracy (unextractable = wrong); paired per-question differences use strict correctness, with a
    bootstrap CI and an exact two-sided McNemar test.
Stdlib only.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = REPO / "results" / "final_eval"
SEEDS = (42, 43, 44)
B = 10_000
LAST_RE = __import__("re").compile(r"answer is \(?([A-J])\)?")

TRACE_MODELS = [  # (served dir, display name, HF id)
    ("qwen3.8-27b", "Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("qwen3.5-9b", "Qwen3.5-9B", "Qwen/Qwen3.5-9B"),
    ("gemma-4-12b-it", "Gemma-4-12B-it", "google/gemma-4-12B-it"),
    ("llava-onevision-2-8b-instruct", "LLaVA-OneVision-2-8B", "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"),
]
# earlier no_tool runs of this repo, same protocol and code path (runs/full/<served>/no_tool)
TRACE_REFERENCES = [
    ("qwen3.5-9b", "Qwen3.5-9B (2026-10-06 run)"),
    ("qwen3.5-4b", "Qwen3.5-4B (2026-10-06 run)"),
    ("qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct (2026-10-05 run)"),
    ("qwen2.5-vl-3b-instruct", "Qwen2.5-VL-3B-Instruct (2026-10-05 run)"),
]
# diagnostic runs (runs/final_eval/<dir>), compared with the main run of the same model on the same rows
# Main Gemma run: checkpoint image budget (280 soft tokens), images only bounded to TRACE's pixel range (family
# patch factor 1). Both diagnostics also rounded every image to 48-px multiples on the client first.
TRACE_DIAGNOSTICS = [
    ("trace_val_diag/gemma_sot280_f48/gemma-4-12b-it", "gemma-4-12b-it",
     "Gemma-4-12B-it, budget 280, client-side 48-px rounding (diagnostic)"),
    ("trace_val_diag/gemma_sot1120_f48/gemma-4-12b-it", "gemma-4-12b-it",
     "Gemma-4-12B-it, budget 1120, client-side 48-px rounding (diagnostic)"),
]
MMLU_MODELS = [
    ("vero-qwen25-7b", "Vero-Qwen25-7B", "zlab-princeton/Vero-Qwen25-7B"),
    ("trace-qwen2.5-vl-7b", "TRACE-Qwen2.5-VL-7B", "maveryn/trace-qwen2.5-vl-7b"),
    ("qwen2.5-vl-7b-instruct", "Qwen2.5-VL-7B-Instruct (base of both)", "Qwen/Qwen2.5-VL-7B-Instruct"),
]


def boot_ci(values: list[float], b: int = B, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return means[int(0.025 * b)], means[int(0.975 * b) - 1]


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


# ------------------------------------------------------------------------------------------- MMLU-Pro helpers
import re as _re

_R1 = _re.compile(r"answer is \(?([A-J])\)?")
_R2 = _re.compile(r".*[aA]nswer:\s*([A-J])")
_R3 = _re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", _re.DOTALL)
_BOX = _re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUM = _re.compile(r"-?\d[\d,]*\.?\d*")


def official_extract(text: str):
    """The official MMLU-Pro chain (extract_answer -> extract_again -> extract_final), re-stated."""
    for rx, group in ((_R1, 1), (_R2, 1), (_R3, 0)):
        m = rx.search(text)
        if m:
            return m.group(group)
    return None


def _norm(s: str):
    s = s.strip().strip("$").replace(",", "").replace("\\%", "%").replace(" ", "")
    s = _re.sub(r"\\(text|mathrm)\{([^}]*)\}", r"\2", s).rstrip(".")
    try:
        return ("num", round(float(s.rstrip("%")), 4))
    except ValueError:
        return ("str", s.lower())


def value_mapped_option(text: str, options: list[str]):
    """Sensitivity only: for an answer with no extractable letter, map its last boxed value (else the text after
    the last 'answer is') to an option when exactly one option equals it (string, or the single number in it)."""
    boxed = _BOX.findall(text)
    cand = boxed[-1] if boxed else (_re.findall(r"answer is[:\s]*([^\n]+)", text) or [None])[-1]
    if cand is None:
        return None
    c = _norm(cand)
    hits = [i for i, o in enumerate(options) if _norm(o) == c]
    if not hits and c[0] == "num":
        hits = [i for i, o in enumerate(options)
                if len(nm := _NUM.findall(o.replace(",", ""))) == 1 and abs(float(nm[0]) - c[1]) < 1e-6]
    return hits[0] if len(hits) == 1 else None


def mmlu_test_options() -> dict[int, list[str]]:
    import os
    import pyarrow.parquet as pq

    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    path = home / "hub" / "datasets--TIGER-Lab--MMLU-Pro" / "snapshots" / "b189ec765aa7ed75c8acfea42df31fdae71f97be" / "data" / "test-00000-of-00001.parquet"
    return {r["question_id"]: [o for o in r["options"] if o != "N/A"] for r in pq.read_table(path).to_pylist()}


def paired_stats(a: dict, b: dict) -> dict:
    keys = sorted(a)
    diffs = [a[k] - b[k] for k in keys]
    b01 = sum(1 for d in diffs if d > 0)
    b10 = sum(1 for d in diffs if d < 0)
    lo, hi = boot_ci(diffs)
    return {"diff": statistics.fmean(diffs), "ci95": [lo, hi], "a_only": b01, "b_only": b10, "mcnemar_p": mcnemar_exact(b01, b10)}


# ------------------------------------------------------------------------------------------- TRACE validation
def trace_model(run_dir: Path) -> dict | None:
    seeds = {}
    for seed in SEEDS:
        d = run_dir / "no_tool" / f"seed{seed}"
        if not (d / "scores.jsonl").exists():
            return None
        seeds[seed] = {"summary": json.loads((d / "summary.json").read_text()), "rows": load_jsonl(d / "scores.jsonl")}
        pending = seeds[seed]["summary"].get("judge_pending") or 0
        if pending or seeds[seed]["summary"].get("excluded_endpoint_errors"):
            raise SystemExit(f"{d}: {pending} rows still await the judge or rows ended in endpoint errors; "
                             "run final_eval/trace_val_judge.sh first")
    per_row = defaultdict(list)
    meta = {}
    for seed, s in seeds.items():
        for r in s["rows"]:
            per_row[r["instance_id"]].append(r["official_correct"])
            meta[r["instance_id"]] = (r["domain"], r["answer_type"])
    if any(len(v) != len(SEEDS) for v in per_row.values()) or len(per_row) != 2000:
        raise SystemExit(f"{run_dir}: expected 2000 rows x {len(SEEDS)} seeds, got {len(per_row)} rows")
    pooled = {k: sum(v) / len(v) for k, v in per_row.items()}
    accs = [s["summary"]["official_acc"] for s in seeds.values()]
    lo, hi = boot_ci(list(pooled.values()))
    by_domain, by_type = defaultdict(list), defaultdict(list)
    for k, v in pooled.items():
        by_domain[meta[k][0]].append(v)
        by_type[meta[k][1]].append(v)
    stops = Counter()
    for s in seeds.values():
        stops.update(r["stop_reason"] for r in s["rows"])
    n_ep = sum(len(s["rows"]) for s in seeds.values())
    return {
        "per_seed": {
            str(seed): {k: s["summary"].get(k) for k in ("n", "official_acc", "deterministic_acc", "strict_acc",
                                                          "strict_format_rate", "judge_resolved", "unresolved",
                                                          "judge_pending", "excluded_endpoint_errors",
                                                          "mean_completion_tokens", "stop_reasons")}
            for seed, s in seeds.items()
        },
        "mean_acc": statistics.fmean(accs),
        "sd_acc": statistics.stdev(accs),
        "pooled_acc": statistics.fmean(pooled.values()),
        "pooled_ci95": [lo, hi],
        "truncated_rate": stops.get("length", 0) / n_ep,
        "stop_reasons": dict(stops),
        "judge_resolved_rate": sum(s["summary"]["judge_resolved"] for s in seeds.values()) / n_ep,
        "unresolved_rate": sum(s["summary"]["unresolved"] for s in seeds.values()) / n_ep,
        "mean_completion_tokens": statistics.fmean(r["completion_tokens"] for s in seeds.values() for r in s["rows"]),
        "by_domain": {d: {"n": len(v), "acc": statistics.fmean(v)} for d, v in sorted(by_domain.items())},
        "by_answer_type": {t: {"n": len(v), "acc": statistics.fmean(v)} for t, v in sorted(by_type.items())},
        "_pooled": pooled,
        "_per_seed_rows": {seed: {r["instance_id"]: r["official_correct"] for r in s["rows"]} for seed, s in seeds.items()},
    }


def paired(a: dict, b: dict) -> dict:
    keys = sorted(a["_pooled"])
    diffs = [a["_pooled"][k] - b["_pooled"][k] for k in keys]
    lo, hi = boot_ci(diffs)
    return {"diff": statistics.fmean(diffs), "ci95": [lo, hi]}


def trace_section() -> dict:
    models, refs = {}, {}
    for served, name, hf in TRACE_MODELS:
        m = trace_model(RUNS / "final_eval" / "trace_val" / served)
        if m:
            m.update(name=name, hf_id=hf)
            cfg = json.loads((RUNS / "final_eval" / "trace_val" / served / "no_tool" / "seed42" / "run_config.json").read_text())
            m["run_config"] = {k: cfg.get(k) for k in ("model_id", "model_revision", "family", "patch_factor", "chat_template_kwargs",
                                                       "max_turns", "max_tokens_per_turn", "min_pixels", "max_pixels",
                                                       "decoding_params", "system_prompt_sha256", "dataset_sha256",
                                                       "git_commit", "code_sha256") if k in cfg}
            models[served] = m
    for served, name in TRACE_REFERENCES:
        m = trace_model(RUNS / "full" / served)
        if m:
            m.update(name=name)
            refs[served] = m
    pairs = {}
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs[f"{names[i]} - {names[j]}"] = paired(models[names[i]], models[names[j]])
    diags = {}
    for rel, main, name in TRACE_DIAGNOSTICS:
        m = trace_model(RUNS / "final_eval" / rel)
        if m and main in models:
            m.update(name=name, vs_main=paired(m, models[main]))
            diags[rel] = m
    repro = None
    if "qwen3.5-9b" in models and "qwen3.5-9b" in refs:
        new, old = models["qwen3.5-9b"], refs["qwen3.5-9b"]
        per_seed = {}
        for seed in SEEDS:
            a, b = new["_per_seed_rows"][seed], old["_per_seed_rows"][seed]
            agree = sum(a[k] == b[k] for k in a) / len(a)
            per_seed[str(seed)] = {"new": statistics.fmean(a.values()), "old": statistics.fmean(b.values()), "row_agreement": agree}
        repro = {"per_seed": per_seed, "pooled_new_minus_old": paired(new, old)}
    for m in list(models.values()) + list(refs.values()) + list(diags.values()):
        m.pop("_pooled")
        m.pop("_per_seed_rows")
    return {"models": models, "references": refs, "diagnostics": diags, "pairwise_pooled": pairs,
            "qwen35_9b_reproducibility": repro}


# ------------------------------------------------------------------------------------------- MMLU-Pro
def mmlu_section() -> dict:
    models, correct = {}, {}
    for name, display, hf in MMLU_MODELS:
        d = RUNS / "final_eval" / "mmlu_pro" / name
        if not (d / "summary.json").exists():
            continue
        s = json.loads((d / "summary.json").read_text())
        s.update(name=display, hf_id=hf)
        models[name] = s
        # per-question strict correctness from the official results files (pred == answer, no random fill)
        corr = {}
        for p in (d / "official_results").glob("*.json"):
            for e in json.loads(p.read_text()):
                corr[e["question_id"]] = float(e["pred"] is not None and e["pred"] == e["answer"])
        correct[name] = corr
        # sensitivity: score the LAST "answer is (X)" instead of the official first match (strict; no fallback)
        last = {}
        for p in sorted(d.glob("shard*.jsonl")):
            for r in load_jsonl(p):
                hits = LAST_RE.findall(r["model_outputs"])
                last[r["question_id"]] = float(bool(hits) and hits[-1] == r["answer"])
        s["sensitivity_last_match_strict_accu"] = statistics.fmean(last.values())
    pairs = {}
    want = [("vero-qwen25-7b", "qwen2.5-vl-7b-instruct"), ("trace-qwen2.5-vl-7b", "qwen2.5-vl-7b-instruct"),
            ("vero-qwen25-7b", "trace-qwen2.5-vl-7b")]
    for a, b in want:
        if a in correct and b in correct:
            keys = sorted(correct[a])
            diffs = [correct[a][k] - correct[b][k] for k in keys]
            b01 = sum(1 for k in keys if correct[a][k] > correct[b][k])
            b10 = sum(1 for k in keys if correct[a][k] < correct[b][k])
            lo, hi = boot_ci(diffs)
            pairs[f"{a} - {b}"] = {"diff_strict": statistics.fmean(diffs), "ci95": [lo, hi], "a_only": b01, "b_only": b10,
                                   "mcnemar_p": mcnemar_exact(b01, b10)}
    if pairs:
        for k, adj in zip(pairs, holm([v["mcnemar_p"] for v in pairs.values()])):
            pairs[k]["mcnemar_p_holm"] = adj
    # secondary protocol: official evaluate_from_api.py prompts as one chat message (mmlu_pro_api_eval.py)
    chat, chat_corr = {}, {}
    for name, display, hf in MMLU_MODELS:
        d = RUNS / "final_eval" / "mmlu_pro_api" / name
        if (d / "summary.json").exists():
            c = json.loads((d / "summary.json").read_text())
            c.update(name=display, hf_id=hf)
            chat[name] = c
            chat_corr[name] = {int(k): float(v) for k, v in json.loads((d / "strict_correct.json").read_text()).items()}
    chat_pairs = {}
    for a, b in want:
        if a in chat_corr and b in chat_corr:
            keys = sorted(chat_corr[a])
            diffs = [chat_corr[a][k] - chat_corr[b][k] for k in keys]
            b01 = sum(1 for k in keys if chat_corr[a][k] > chat_corr[b][k])
            b10 = sum(1 for k in keys if chat_corr[a][k] < chat_corr[b][k])
            lo, hi = boot_ci(diffs)
            chat_pairs[f"{a} - {b}"] = {"diff_strict": statistics.fmean(diffs), "ci95": [lo, hi], "a_only": b01, "b_only": b10,
                                        "mcnemar_p": mcnemar_exact(b01, b10)}
    if chat_pairs:
        for k, adj in zip(chat_pairs, holm([v["mcnemar_p"] for v in chat_pairs.values()])):
            chat_pairs[k]["mcnemar_p_holm"] = adj
    # format effect per model: chat strict minus raw-text strict on the same questions; for the base, split by
    # whether its raw-text output stopped or ran to 2,048 tokens
    format_effect, base_split, mapped, mapped_pairs = {}, {}, {}, {}
    if chat_corr:
        for name in chat_corr:
            if name in correct:
                format_effect[name] = paired_stats(chat_corr[name], correct[name])
        base = "qwen2.5-vl-7b-instruct"
        if base in chat_corr and base in correct:
            fin = {}
            for p in sorted((RUNS / "final_eval" / "mmlu_pro" / base).glob("shard*.jsonl")):
                for r in load_jsonl(p):
                    fin[r["question_id"]] = r["finish_reason"]
            for kind in ("length", "stop"):
                ks = [k for k in correct[base] if fin[k] == kind]
                base_split[kind] = {"n": len(ks), "raw_strict": statistics.fmean(correct[base][k] for k in ks),
                                    "chat_strict": statistics.fmean(chat_corr[base][k] for k in ks)}
        # sensitivity: credit an unextractable chat answer whose boxed/stated value matches exactly one option
        opts = mmlu_test_options()
        for name in chat_corr:
            recs = {}
            for p in sorted((RUNS / "final_eval" / "mmlu_pro_api" / name).glob("shard*.jsonl")):
                for r in load_jsonl(p):
                    recs[r["question_id"]] = r
            v, added, added_math = dict(chat_corr[name]), 0, 0
            for q, r in recs.items():
                text = r["model_outputs"].replace("**", "")
                if official_extract(text) is not None:
                    continue
                i = value_mapped_option(text, opts[q])
                if i is not None and i == r["answer_index"]:
                    v[q] = 1.0
                    added += 1
                    added_math += r["category"] == "math"
            unext = sum(1 for r in recs.values() if official_extract(r["model_outputs"].replace("**", "")) is None)
            unext_math = sum(1 for r in recs.values() if r["category"] == "math" and official_extract(r["model_outputs"].replace("**", "")) is None)
            mapped[name] = {"strict_accu": statistics.fmean(v.values()), "credited": added, "credited_math": added_math,
                            "unextractable": unext, "unextractable_math": unext_math, "_v": v}
        for a, b in want:
            if a in mapped and b in mapped:
                mapped_pairs[f"{a} - {b}"] = paired_stats(mapped[a]["_v"], mapped[b]["_v"])
        for m in mapped.values():
            m.pop("_v")
    return {"models": models, "pairwise_strict": pairs,
            "chat_protocol": {"models": chat, "pairwise_strict": chat_pairs, "format_effect_chat_minus_raw": format_effect,
                              "base_by_raw_finish": base_split, "value_mapped_sensitivity": mapped,
                              "value_mapped_pairwise": mapped_pairs}}


# ------------------------------------------------------------------------------------------- tables
def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def tables(tv: dict, mp: dict) -> str:
    L = ["# Final evaluation tables (generated by final_eval/summarize.py)", ""]
    if mp["models"]:
        L += ["## MMLU-Pro (official TIGER-AI-Lab protocol: 5-shot CoT, greedy, 12,032 questions)", "",
              "| model | official acc | strict acc | no extraction | truncated (2,048 tok) | mean output tokens | strict, last match (sensitivity) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for m in mp["models"].values():
            t = m["total"]
            L.append(f"| {m['name']} | {pct(t['accu'])} | {pct(t['strict_accu'])} | {t['no_extraction']} | {t['truncated']} | {t['mean_completion_tokens']:.0f} | {pct(m['sensitivity_last_match_strict_accu'])} |")
        cats = sorted(next(iter(mp["models"].values()))["by_category"])
        L += ["", "Official accuracy (%) by category", "", "| category | n | " + " | ".join(m["name"] for m in mp["models"].values()) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in mp["models"]) + " |"]
        for c in cats:
            n = next(iter(mp["models"].values()))["by_category"][c]["n"]
            L.append(f"| {c} | {n} | " + " | ".join(pct(m["by_category"][c]["accu"]) for m in mp["models"].values()) + " |")
        if mp["pairwise_strict"]:
            L += ["", "Paired differences, strict correctness (pp; 95% bootstrap CI; exact McNemar, Holm over the 3 pairs)", "",
                  "| pair | diff | 95% CI | a right only | b right only | p (Holm) |", "| --- | ---: | --- | ---: | ---: | ---: |"]
            for k, v in mp["pairwise_strict"].items():
                L.append(f"| {k} | {100 * v['diff_strict']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] | {v['a_only']} | {v['b_only']} | {v['mcnemar_p_holm']:.2g} |")
    cp = mp.get("chat_protocol") or {}
    if cp.get("models"):
        L += ["", "## MMLU-Pro, secondary protocol: official `evaluate_from_api.py` prompts as one chat message (greedy, 4,000 tokens)", "",
              "| model | strict acc | no extraction | truncated (4,000 tok) | mean output tokens |", "| --- | ---: | ---: | ---: | ---: |"]
        for m in cp["models"].values():
            t = m["total"]
            L.append(f"| {m['name']} | {pct(t['strict_accu'])} | {t['no_extraction']} | {t['truncated']} | {t['mean_completion_tokens']:.0f} |")
        cats = sorted(next(iter(cp["models"].values()))["by_category"])
        L += ["", "Strict accuracy (%) by category, chat protocol", "", "| category | n | " + " | ".join(m["name"] for m in cp["models"].values()) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in cp["models"]) + " |"]
        for c in cats:
            n = next(iter(cp["models"].values()))["by_category"][c]["n"]
            L.append(f"| {c} | {n} | " + " | ".join(pct(m["by_category"][c]["strict_accu"]) for m in cp["models"].values()) + " |")
        if cp["pairwise_strict"]:
            L += ["", "Paired differences, chat protocol, strict correctness (pp; 95% bootstrap CI; exact McNemar, Holm over the 3 pairs)", "",
                  "| pair | diff | 95% CI | a right only | b right only | p (Holm) |", "| --- | ---: | --- | ---: | ---: | ---: |"]
            for k, v in cp["pairwise_strict"].items():
                L.append(f"| {k} | {100 * v['diff_strict']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] | {v['a_only']} | {v['b_only']} | {v['mcnemar_p_holm']:.2g} |")
    if cp.get("format_effect_chat_minus_raw"):
        L += ["", "Format effect per model: chat strict − raw-text strict on the same 12,032 questions (pp; 95% bootstrap CI; exact McNemar, unadjusted)", "",
              "| model | chat − raw | 95% CI | up | down | p |", "| --- | ---: | --- | ---: | ---: | ---: |"]
        for k, v in cp["format_effect_chat_minus_raw"].items():
            L.append(f"| {k} | {100 * v['diff']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] | {v['a_only']} | {v['b_only']} | {v['mcnemar_p']:.2g} |")
        if cp.get("base_by_raw_finish"):
            L += ["", "Base model, split by how its raw-text output ended (strict %)", "", "| raw-text output | n | raw | chat | chat − raw |", "| --- | ---: | ---: | ---: | ---: |"]
            for kind, v in cp["base_by_raw_finish"].items():
                L.append(f"| {'ran to 2,048 tokens' if kind == 'length' else 'stopped'} | {v['n']} | {pct(v['raw_strict'])} | {pct(v['chat_strict'])} | {100 * (v['chat_strict'] - v['raw_strict']):+.2f} |")
    if cp.get("value_mapped_sensitivity"):
        L += ["", "Chat sensitivity (not a protocol): an unextractable answer is credited when its boxed (or stated) value matches exactly one option and that option is correct", "",
              "| model | unextractable (math) | credited (math) | strict acc, value-mapped |", "| --- | ---: | ---: | ---: |"]
        for k, v in cp["value_mapped_sensitivity"].items():
            L.append(f"| {k} | {v['unextractable']} ({v['unextractable_math']}) | {v['credited']} ({v['credited_math']}) | {pct(v['strict_accu'])} |")
        L += ["", "| pair (value-mapped) | diff | 95% CI | p (unadjusted) |", "| --- | ---: | --- | ---: |"]
        for k, v in cp["value_mapped_pairwise"].items():
            L.append(f"| {k} | {100 * v['diff']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] | {v['mcnemar_p']:.2g} |")
    if tv["models"]:
        L += ["", "## TRACE validation (TRACE protocol, `no_tool`; 2,000 rows x seeds 42, 43, 44)", "",
              "| model | seed 42 | seed 43 | seed 44 | mean ± SD | seeds-pooled [95% CI] | truncated | judge-resolved | unresolved | mean tokens |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
        for group in (tv["models"], tv["references"], tv["diagnostics"]):
            for m in group.values():
                ps = m["per_seed"]
                L.append(f"| {m['name']} | " + " | ".join(pct(ps[str(s)]["official_acc"]) for s in SEEDS)
                         + f" | {pct(m['mean_acc'])} ± {100 * m['sd_acc']:.2f} | {pct(m['pooled_acc'])} [{pct(m['pooled_ci95'][0])}, {pct(m['pooled_ci95'][1])}]"
                         + f" | {pct(m['truncated_rate'])}% | {pct(m['judge_resolved_rate'])}% | {pct(m['unresolved_rate'])}% | {m['mean_completion_tokens']:.0f} |")
        doms = sorted(next(iter(tv["models"].values()))["by_domain"])
        allm = list(tv["models"].values()) + list(tv["references"].values())
        L += ["", "Seeds-pooled accuracy (%) by domain", "", "| domain | n | " + " | ".join(m["name"] for m in allm) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in allm) + " |"]
        for d in doms:
            n = next(iter(tv["models"].values()))["by_domain"][d]["n"]
            L.append(f"| {d} | {n} | " + " | ".join(pct(m["by_domain"][d]["acc"]) for m in allm) + " |")
        types = sorted(next(iter(tv["models"].values()))["by_answer_type"])
        L += ["", "Seeds-pooled accuracy (%) by answer type", "", "| answer type | n | " + " | ".join(m["name"] for m in allm) + " |",
              "| --- | ---: | " + " | ".join("---:" for _ in allm) + " |"]
        for t in types:
            n = next(iter(tv["models"].values()))["by_answer_type"][t]["n"]
            L.append(f"| {t} | {n} | " + " | ".join(pct(m["by_answer_type"][t]["acc"]) for m in allm) + " |")
        if tv["pairwise_pooled"]:
            L += ["", "Paired differences between the new models, seeds-pooled (pp; 95% bootstrap CI over rows)", "",
                  "| pair | diff | 95% CI |", "| --- | ---: | --- |"]
            for k, v in tv["pairwise_pooled"].items():
                L.append(f"| {k} | {100 * v['diff']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] |")
        if tv["diagnostics"]:
            L += ["", "Diagnostics vs the main run of the same model (seeds-pooled, same rows; pp, 95% bootstrap CI)", "",
                  "| diagnostic | acc | diag − main | 95% CI |", "| --- | ---: | ---: | --- |"]
            for d in tv["diagnostics"].values():
                v = d["vs_main"]
                L.append(f"| {d['name']} | {pct(d['pooled_acc'])} | {100 * v['diff']:+.2f} | [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}] |")
        r = tv["qwen35_9b_reproducibility"]
        if r:
            L += ["", "Qwen3.5-9B rerun vs the 2026-10-06 run (same checkpoint, protocol and code path)", "",
                  "| seed | new | old | per-row agreement |", "| --- | ---: | ---: | ---: |"]
            for s, v in r["per_seed"].items():
                L.append(f"| {s} | {pct(v['new'])} | {pct(v['old'])} | {pct(v['row_agreement'])}% |")
            p = r["pooled_new_minus_old"]
            L.append(f"| pooled new − old | {100 * p['diff']:+.2f} pp | [{100 * p['ci95'][0]:+.2f}, {100 * p['ci95'][1]:+.2f}] | |")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tv, mp = trace_section(), mmlu_section()
    (OUT / "trace_val_summary.json").write_text(json.dumps(tv, indent=1))
    (OUT / "mmlu_pro_summary.json").write_text(json.dumps(mp, indent=1))
    (OUT / "tables.md").write_text(tables(tv, mp))
    for served in tv["models"]:
        src = RUNS / "final_eval" / "trace_val" / served / "report.md"
        if src.exists():
            (OUT / "trace_val" / served).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, OUT / "trace_val" / served / "report.md")
    print((OUT / "tables.md").read_text())


if __name__ == "__main__":
    main()
