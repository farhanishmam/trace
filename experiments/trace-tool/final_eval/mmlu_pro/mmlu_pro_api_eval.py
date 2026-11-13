"""MMLU-Pro, secondary protocol: the official API-style evaluator (evaluate_from_api.py @ f418b116) served locally.

The primary result is mmlu_pro_eval.py: the official local protocol with raw-text 5-shot completion. In raw
completion the RL-tuned checkpoints keep writing after their answer until the token limit. This variant uses
MMLU-Pro's other official entry point, which sends the same 5-shot chain of thought as ONE chat user message,
so the models stop at their end-of-turn token.

* prompt: built by the vendored ``single_request`` itself, with its ``call_api`` swapped for a function that
  captures (instruction, inputs), so the user message is the official one byte for byte.
* generation: vLLM ``LLM.chat`` with the checkpoint's chat template (its default system prompt, if any);
  ``SamplingParams(temperature=0, top_p=1, max_tokens=4000)`` as in the script's OpenAI branch; one greedy
  engine per GPU on a question shard.
* scoring: ``response.replace('**', '')`` then the official ``extract_answer``. Reported as strict accuracy
  (unextractable = wrong); the API script's random-letter fallback is not replicated (its draw order depends on
  its resume loop). The no-extraction count is reported next to it.

  python mmlu_pro_api_eval.py generate --model ID --revision REV --shard 0/8 --out DIR
  python mmlu_pro_api_eval.py score --out DIR [--model-name NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "official"
API_SHA256 = "6e00c0d57ed307b9e707b712ea796fbd2286c6d21889cd63bb58090fa72c5a99"
MAX_TOKENS = 4000
MAX_MODEL_LEN = 12288  # room for the 4,000-token budget after the longest 5-shot user message

sys.path.insert(0, str(HERE))
import mmlu_pro_eval as local  # noqa: E402  (dataset snapshot + pins)


def load_api_official():
    got = hashlib.sha256((OFFICIAL / "evaluate_from_api.py").read_bytes()).hexdigest()
    if got != API_SHA256:
        raise SystemExit(f"official/evaluate_from_api.py sha256 {got} != pinned {API_SHA256}")

    def _ensure(name, **attrs):
        """Import the real module if installed (vLLM needs the real openai); otherwise stub it."""
        try:
            __import__(name)
            return
        except ImportError:
            pass
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], child, mod)

    # API client libraries the module imports at top level; none of them is called here
    for name, attrs in [("openai", {"OpenAI": None}), ("anthropic", {}), ("google", {}), ("google.generativeai", {}),
                        ("ai21", {"AI21Client": None}), ("ai21.models", {}),
                        ("ai21.models.chat", {k: None for k in ("ChatMessage", "ResponseFormat", "DocumentSchema",
                                                                "FunctionToolDefinition", "ToolDefinition", "ToolParameters")}),
                        ("datasets", {"load_dataset": None}), ("tqdm", {"tqdm": None}), ("requests", {})]:
        _ensure(name, **attrs)
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_from_api as api  # noqa: E402

    return api


def official_messages(api, test, val):
    """(question_id, user message) per test row, captured from the official single_request."""
    cot = defaultdict(list)
    for e in val:
        cot[e["category"]].append(e)
    captured = []

    def capture(client, instruction, inputs):
        captured.append(instruction + inputs)
        return "the answer is (A)"

    api.call_api = capture
    out = []
    for row in test:
        api.single_request(None, row, cot, [])
        out.append((row["question_id"], captured.pop()))
    return out


def cmd_generate(args) -> None:
    os.chdir(local.OFFICIAL)
    off_local = local.load_official()
    api = load_api_official()
    from vllm import LLM, SamplingParams

    i, n = (int(x) for x in args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_path = out / f"shard{i:02d}of{n:02d}.jsonl"
    done = set()
    if shard_path.exists():
        done = {json.loads(line)["question_id"] for line in shard_path.open() if line.strip()}
    test, val = local.load_rows(off_local)  # same parquet + official preprocess (N/A options dropped)
    rows = [test[j] for j in range(len(test)) if j % n == i and test[j]["question_id"] not in done]
    print(f"shard {i}/{n}: {len(rows)} to run ({len(done)} done)", flush=True)
    if not rows:
        return
    msgs = official_messages(api, rows, val)
    llm = LLM(model=args.model, revision=args.revision, gpu_memory_utilization=args.gpu_util, tensor_parallel_size=1,
              max_model_len=MAX_MODEL_LEN, trust_remote_code=True, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
    sp = SamplingParams(temperature=0, top_p=1, max_tokens=MAX_TOKENS)
    t0 = time.time()
    outputs = llm.chat([[{"role": "user", "content": m}] for _, m in msgs], sp)
    print(f"generated {len(outputs)} in {time.time() - t0:.0f}s", flush=True)
    with shard_path.open("a") as f:
        for row, (qid, _), o in zip(rows, msgs, outputs):
            assert qid == row["question_id"]
            f.write(json.dumps({
                "question_id": qid, "category": row["category"], "answer": row["answer"],
                "answer_index": row["answer_index"], "prompt_tokens": len(o.prompt_token_ids),
                "completion_tokens": len(o.outputs[0].token_ids), "finish_reason": o.outputs[0].finish_reason,
                "model_outputs": o.outputs[0].text,
            }) + "\n")
    print("wrote", shard_path, flush=True)


def cmd_score(args) -> None:
    os.chdir(local.OFFICIAL)
    off_local = local.load_official(stub_heavy=True)
    api = load_api_official()
    test, _ = local.load_rows(off_local)
    recs = {}
    for p in sorted(Path(args.out).glob("shard*.jsonl")):
        for line in p.open():
            if line.strip():
                r = json.loads(line)
                recs[r["question_id"]] = r
    missing = [r["question_id"] for r in test if r["question_id"] not in recs]
    if missing:
        raise SystemExit(f"{args.out}: {len(missing)} questions missing -- not scoring")
    devnull = open(os.devnull, "w")
    by_cat = defaultdict(lambda: Counter())
    strict = {}
    for row in test:
        r = recs[row["question_id"]]
        stdout, sys.stdout = sys.stdout, devnull
        try:
            pred = api.extract_answer(r["model_outputs"].replace("**", ""))
        finally:
            sys.stdout = stdout
        c = by_cat[row["category"]]
        c["n"] += 1
        c["strict_corr"] += pred is not None and pred == row["answer"]
        strict[row["question_id"]] = int(pred is not None and pred == row["answer"])
        c["no_extraction"] += pred is None
        c["truncated"] += r["finish_reason"] == "length"
        c["completion_tokens"] += r["completion_tokens"]
    tot = Counter()
    for c in by_cat.values():
        tot.update(c)
    summary = {
        "model": args.model_name, "protocol": "evaluate_from_api.py (chat, one user message, 5-shot CoT) via vLLM, greedy, 4000 tokens",
        "total": {"n": tot["n"], "strict_accu": tot["strict_corr"] / tot["n"], "no_extraction": tot["no_extraction"],
                  "truncated": tot["truncated"], "mean_completion_tokens": tot["completion_tokens"] / tot["n"],
                  "max_prompt_tokens": max(r["prompt_tokens"] for r in recs.values())},
        "by_category": {k: {**v, "strict_accu": v["strict_corr"] / v["n"]} for k, v in sorted(by_cat.items())},
    }
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=1))
    (Path(args.out) / "strict_correct.json").write_text(json.dumps(strict))
    t = summary["total"]
    print(f"{args.model_name}: strict {100 * t['strict_accu']:.2f}  no-extraction {t['no_extraction']}  truncated {t['truncated']}  "
          f"mean tokens {t['mean_completion_tokens']:.0f}  max prompt {t['max_prompt_tokens']}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--revision")
    g.add_argument("--shard", default="0/1")
    g.add_argument("--out", required=True)
    g.add_argument("--gpu-util", type=float, default=0.92)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--model-name")
    args = p.parse_args(argv)
    args.out = str(Path(args.out).resolve())
    {"generate": cmd_generate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
