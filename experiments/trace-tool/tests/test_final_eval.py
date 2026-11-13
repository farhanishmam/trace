"""Final evaluation (2026-11-12): the no_tool-only model families and the MMLU-Pro wrapper (CPU only)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from trace_tool.conditions import build_episode_config, load_conditions
from trace_tool.models import FAMILIES, chat_template_kwargs, detect_family

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "model_id, family, kwargs",
    [
        ("Qwen/Qwen3.8-27B", "qwen3.8", {"enable_thinking": False}),
        ("qwen3.8-27b", "qwen3.8", {"enable_thinking": False}),
        ("google/gemma-4-12B-it", "gemma4", {"enable_thinking": False}),
        ("gemma-4-12b-it", "gemma4", {"enable_thinking": False}),
        ("lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct", "llava-onevision-2", None),
        ("llava-onevision-2-8b-instruct", "llava-onevision-2", None),
        ("Qwen/Qwen3.5-9B", "qwen3.5", {"enable_thinking": False}),  # unchanged
        ("maveryn/trace-qwen2.5-vl-7b", "qwen2.5-vl", None),  # unchanged
    ],
)
def test_detect_new_families(model_id, family, kwargs):
    fam = detect_family(model_id)
    assert fam.name == family
    assert chat_template_kwargs(fam) == kwargs


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "qwen3-8b", "/ckpt/qwen3_8b", "Qwen/Qwen3-80B", "llava-onevision-qwen2-7b-ov", "google/gemma-3-4b-it"])
def test_lookalikes_are_not_new_families(model_id):
    with pytest.raises(ValueError):
        detect_family(model_id)


@pytest.mark.parametrize("family", ["qwen3.8", "gemma4", "llava-onevision-2"])
def test_new_families_run_no_tool_only(family):
    conditions = load_conditions(REPO / "configs" / "conditions.yaml")
    fam = FAMILIES[family]
    cfg = build_episode_config(conditions["no_tool"], seed=42, family=fam)
    assert cfg.tools == {} and cfg.max_turns == 1 and cfg.max_tokens_per_turn == 2048
    assert (cfg.min_pixels, cfg.max_pixels) == (262144, 4194304)
    with pytest.raises(RuntimeError, match="no studied tool-call format"):
        build_episode_config(conditions["zoom"], seed=42, family=fam)


def _mmlu():
    spec = importlib.util.spec_from_file_location("mmlu_pro_eval", REPO / "final_eval" / "mmlu_pro" / "mmlu_pro_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mmlu_official_files_pinned_and_extraction(monkeypatch):
    mod = _mmlu()
    monkeypatch.chdir(mod.OFFICIAL)
    saved = {k: sys.modules.get(k) for k in ("torch", "vllm", "vllm.lora", "vllm.lora.request", "transformers", "tqdm", "datasets", "evaluate_from_local")}
    try:
        off = mod.load_official(stub_heavy=True)  # checks both sha256 pins
        assert off.max_model_length == 4096 and off.max_new_tokens == 2048
        assert off.extract_answer("so the answer is (C).") == "C"
        assert off.extract_answer("Answer: D because") == "D"
        assert off.extract_answer("I pick B") == "B"
        assert off.extract_answer("no letter here") is None
        # the official prompt: header with the subject, k shots, then the open question
        val = [{"category": "law", "question": f"q{i}", "options": ["x", "y"], "cot_content": "A: Let's think step by step. ok"} for i in range(6)]
        curr = {"category": "law", "question": "Q?", "options": ["a", "b", "c"]}
        p = off.generate_cot_prompt(val, curr, 2)
        assert p.startswith("The following are multiple choice questions (with answers) about law.")
        assert p.count("Question:\n") == 3 and p.endswith("Options:\nA. a\nB. b\nC. c\nAnswer: Let's think step by step.")
        assert "Answer: Let's think step by step. ok" in p
        # official scoring: an unextractable answer gets a random option drawn from the module's seeded `random`
        res = [{"pred": None, "options": ["a"] * 10, "answer_index": 3, "answer": "D"}, {"pred": "B", "options": ["a"] * 10, "answer_index": 1, "answer": "B"}]
        state = random.getstate()
        acc, corr, wrong = off.save_res(res, str(Path(mod.OFFICIAL).parent.parent / "official_test_tmp.json"))
        assert corr + wrong == 2 and corr >= 1
        random.setstate(state)
    finally:
        (Path(mod.OFFICIAL).parent.parent / "official_test_tmp.json").unlink(missing_ok=True)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        if str(mod.OFFICIAL) in sys.path:
            sys.path.remove(str(mod.OFFICIAL))
