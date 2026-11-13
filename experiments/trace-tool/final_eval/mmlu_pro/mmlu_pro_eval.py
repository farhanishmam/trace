"""MMLU-Pro under the official TIGER-AI-Lab protocol (evaluate_from_local.py @ f418b116), data-parallel.

The prompt builder, answer extractor and scorer are imported unchanged from the vendored official file
(``official/evaluate_from_local.py``, sha256-checked). What this wrapper changes, none of which changes a
prompt, a generated token or a score:

* data parallel: ``generate`` runs one shard (``--shard i/n``) per GPU; ``score`` merges the shards. The
  official script runs one tensor-parallel engine over all GPUs, subject by subject. Greedy decoding makes
  each answer independent of the batch, up to floating-point non-determinism.
* data loading: the official ``load_dataset("TIGER-Lab/MMLU-Pro")`` is replaced by reading the same parquet
  files from a pinned Hub snapshot (offline compute nodes); rows go through the official ``preprocess``.
* the vLLM engine skips multimodal profiling (``limit_mm_per_prompt`` 0 images) for the VL checkpoints and
  takes ``--gpu-util`` (official default 0.8; 7B VL weights need more on 24 GB cards). Same 4,096-token
  context, ``SamplingParams(temperature=0, max_tokens=2048, stop=["Question:"])`` and raw-text prompts
  (no chat template), exactly as the official script.

Scoring = the official ``save_res``: subjects in sorted order, ``save_res`` called twice per subject (as
``eval_cot`` does), the module's ``random`` seeded 12345 at import, so the random letter given to an answer
the regexes cannot extract is the official one. Also reported: strict accuracy (no extraction = wrong).

  python mmlu_pro_eval.py generate --model PATH_OR_ID --revision REV --shard 0/8 --out DIR
  python mmlu_pro_eval.py score --out DIR [--model-name NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "official"
OFFICIAL_SHA256 = {
    "evaluate_from_local.py": "d0a4ae51b80efa8f99713a21253ad5542913ef2ac9915e95d40c8a62174e9401",
    "cot_prompt_lib/initial_prompt.txt": "d7337806b4d84eef7a86bcb20bf3d64b65e72736c1603e16d08cadebac436f0a",
}
OFFICIAL_COMMIT = "f418b116db00b065c2aea046518d8fcf74d39872"
DATASET_REPO = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
N_TEST = 12032


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_official(stub_heavy: bool = False):
    """Import the vendored official module (its cwd-relative prompt file needs cwd = official/)."""
    for rel, want in OFFICIAL_SHA256.items():
        got = _sha256(OFFICIAL / rel)
        if got != want:
            raise SystemExit(f"official/{rel} sha256 {got} != pinned {want}")
    # The module imports `datasets` (only for load_dataset, unused here) and, at top level, torch, vllm,
    # transformers and tqdm. Scoring stubs those (stub_heavy) so it runs on a small CPU box.
    def _stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod

    if stub_heavy:
        _stub("torch", no_grad=lambda: (lambda f: f))
        _stub("vllm", LLM=None, SamplingParams=None)
        _stub("vllm.lora")
        _stub("vllm.lora.request", LoRARequest=None)
        _stub("transformers")
        _stub("tqdm", tqdm=None)
    try:
        import datasets  # noqa: F401
    except ImportError:
        _stub("datasets", load_dataset=None)
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_from_local as official  # noqa: E402

    return official


def dataset_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    d = home / "hub" / "datasets--TIGER-Lab--MMLU-Pro" / "snapshots" / DATASET_REVISION / "data"
    if not d.is_dir():
        raise SystemExit(f"MMLU-Pro snapshot {DATASET_REVISION} not found under {home}/hub (snapshot_download it first)")
    return d


def load_rows(official):
    import pyarrow.parquet as pq

    d = dataset_dir()
    test = official.preprocess(pq.read_table(d / "test-00000-of-00001.parquet").to_pylist())
    val = official.preprocess(pq.read_table(d / "validation-00000-of-00001.parquet").to_pylist())
    if len(test) != N_TEST:
        raise SystemExit(f"expected {N_TEST} test rows, got {len(test)}")
    return test, val


def build_prompts(official, test, val, tokenizer, ntrain: int = 5):
    """The official eval_cot prompt loop, verbatim in effect: k shots, dropping one while the prompt is
    >= max_model_length - max_new_tokens tokens."""
    prompts, shots = [], []
    for curr in test:
        k = ntrain
        while True:
            prompt = official.generate_cot_prompt(official.select_by_category(val, curr["category"]), curr, k)
            length = len(tokenizer(prompt)["input_ids"])
            k -= 1
            if length < official.max_model_length - official.max_new_tokens:
                break
        prompts.append(prompt)
        shots.append(k + 1)
    return prompts, shots


def cmd_generate(args) -> None:
    os.chdir(OFFICIAL)  # generate_cot_prompt opens cot_prompt_lib/initial_prompt.txt relative to cwd
    official = load_official()
    import transformers
    from vllm import LLM, SamplingParams

    i, n = (int(x) for x in args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_path = out / f"shard{i:02d}of{n:02d}.jsonl"
    done = set()
    if shard_path.exists():
        done = {json.loads(line)["question_id"] for line in shard_path.open() if line.strip()}
    test, val = load_rows(official)
    mine = [j for j in range(len(test)) if j % n == i]
    todo = [j for j in mine if test[j]["question_id"] not in done]
    print(f"shard {i}/{n}: {len(mine)} questions, {len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    tok = transformers.AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    t0 = time.time()
    prompts, shots = build_prompts(official, [test[j] for j in todo], val, tok, args.ntrain)
    print(f"built {len(prompts)} prompts in {time.time() - t0:.0f}s; shots used {dict(Counter(shots))}", flush=True)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        gpu_memory_utilization=args.gpu_util,
        tensor_parallel_size=1,
        max_model_len=official.max_model_length,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=0,
    )
    sp = SamplingParams(temperature=0, max_tokens=official.max_new_tokens, stop=["Question:"])
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"generated {len(outputs)} in {time.time() - t0:.0f}s", flush=True)
    with shard_path.open("a") as f:
        for j, k, o in zip(todo, shots, outputs):
            row = test[j]
            text = o.outputs[0].text
            f.write(json.dumps({
                "question_id": row["question_id"], "category": row["category"], "answer": row["answer"],
                "answer_index": row["answer_index"], "n_options": len(row["options"]), "shots": k,
                "prompt_tokens": len(o.prompt_token_ids), "completion_tokens": len(o.outputs[0].token_ids),
                "finish_reason": o.outputs[0].finish_reason, "stop_reason": o.outputs[0].stop_reason,
                "model_outputs": text,
            }) + "\n")
    print("wrote", shard_path, flush=True)


def cmd_score(args) -> None:
    os.chdir(OFFICIAL)
    official = load_official(stub_heavy=True)  # seeds the module's `random` with 12345, as at official import
    test, _ = load_rows(official)
    out = Path(args.out)
    recs = {}
    dup = 0
    for p in sorted(out.glob("shard*.jsonl")):
        for line in p.open():
            if line.strip():
                r = json.loads(line)
                dup += r["question_id"] in recs
                recs[r["question_id"]] = r
    missing = [row["question_id"] for row in test if row["question_id"] not in recs]
    if missing or dup:
        raise SystemExit(f"{out}: {len(missing)} questions missing, {dup} duplicated -- not scoring")
    # Official flow: per subject (sorted), res = rows in test order with pred = extract_answer(output);
    # save_res is called twice and the second result is kept. Extraction prints on a miss; silence it.
    by_subject = defaultdict(list)
    for row in test:
        by_subject[row["category"]].append(row)
    sta, strict, nofmt, cut = {}, Counter(), Counter(), Counter()
    tok = Counter()
    devnull = open(os.devnull, "w")
    for subject in sorted(by_subject):
        res = []
        for row in by_subject[subject]:
            r = recs[row["question_id"]]
            stdout, sys.stdout = sys.stdout, devnull
            try:
                pred = official.extract_answer(r["model_outputs"])
            finally:
                sys.stdout = stdout
            each = dict(row)
            each["pred"] = pred
            each["model_outputs"] = r["model_outputs"]
            res.append(each)
            strict[subject] += pred is not None and pred == row["answer"]
            nofmt[subject] += pred is None
            cut[subject] += r["finish_reason"] == "length"
            tok[subject] += r["completion_tokens"]
        path = out / "official_results" / f"{subject}.json"
        path.parent.mkdir(exist_ok=True)
        official.save_res(res, str(path))
        acc, corr, wrong = official.save_res(res, str(path))
        sta[subject] = {"corr": corr, "wrong": wrong, "accu": acc, "n": len(res), "strict_corr": strict[subject],
                        "no_extraction": nofmt[subject], "truncated": cut[subject], "completion_tokens": tok[subject]}
    tc = sum(v["corr"] for v in sta.values())
    tw = sum(v["wrong"] for v in sta.values())
    total = {
        "n": len(test), "corr": tc, "wrong": tw, "accu": tc / (tc + tw + 0.000001),  # the official formula
        "strict_accu": sum(strict.values()) / len(test),
        "no_extraction": sum(nofmt.values()), "truncated": sum(cut.values()),
        "mean_completion_tokens": sum(tok.values()) / len(test),
        "first_regex_rate": sum(1 for r in recs.values() if re.search(r"answer is \(?([A-J])\)?", r["model_outputs"])) / len(test),
        "shots": dict(Counter(r["shots"] for r in recs.values())),
    }
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    summary = {"model": args.model_name or meta.get("model"), "meta": meta, "official_commit": OFFICIAL_COMMIT,
               "dataset": f"{DATASET_REPO}@{DATASET_REVISION}", "total": total, "by_category": sta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{summary['model']}: official {100 * total['accu']:.2f}  strict {100 * total['strict_accu']:.2f}  "
          f"no-extraction {total['no_extraction']}  truncated {total['truncated']}  "
          f"mean tokens {total['mean_completion_tokens']:.0f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--revision")
    g.add_argument("--shard", default="0/1")
    g.add_argument("--out", required=True)
    g.add_argument("--ntrain", type=int, default=5)
    g.add_argument("--gpu-util", type=float, default=0.8)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--model-name")
    args = p.parse_args(argv)
    args.out = str(Path(args.out).resolve())
    {"generate": cmd_generate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
