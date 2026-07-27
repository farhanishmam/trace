"""Gradio entry point for the public Trace task explorer."""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from trace_demo import (
    DEFAULT_DOMAIN,
    DEFAULT_SCENE_ID,
    DEFAULT_SEED,
    DEFAULT_TASK_ID,
    MAX_SEED,
    build_catalog,
    generate_demo,
    load_presets,
    sample_random_selection,
)

CATALOG = build_catalog()
PRESETS = load_presets()

_CSS = """
.trace-shell {max-width: 1440px; margin: 0 auto;}
.trace-kicker {letter-spacing: .12em; text-transform: uppercase; font-size: .78rem;
  color: var(--body-text-color-subdued);}
.trace-title h1 {margin-bottom: .25rem;}
.trace-title p {max-width: 900px; font-size: 1.02rem;}
.trace-stat {border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: .8rem 1rem; background: var(--background-fill-secondary);}
.trace-stat strong {font-size: 1.35rem; display: block;}
.trace-run {min-height: 48px;}
.trace-note {font-size: .9rem; color: var(--body-text-color-subdued);}
"""


def _scene_update(domain: str):
    scenes = CATALOG.scenes(domain)
    scene_id = scenes[0]
    return (
        gr.Dropdown(choices=list(scenes), value=scene_id),
        gr.Dropdown(
            choices=list(CATALOG.tasks(domain, scene_id)),
            value=CATALOG.tasks(domain, scene_id)[0],
        ),
    )


def _task_update(domain: str, scene_id: str):
    tasks = CATALOG.tasks(domain, scene_id)
    return gr.Dropdown(choices=list(tasks), value=tasks[0])


def _preset_update(preset_index: str):
    try:
        preset = PRESETS[int(preset_index)]
    except (IndexError, TypeError, ValueError) as exc:
        raise gr.Error("Choose one of the curated Trace presets.") from exc
    return (
        gr.Dropdown(choices=list(CATALOG.domains), value=preset.domain),
        gr.Dropdown(
            choices=list(CATALOG.scenes(preset.domain)),
            value=preset.scene_id,
        ),
        gr.Dropdown(
            choices=list(CATALOG.tasks(preset.domain, preset.scene_id)),
            value=preset.task_id,
        ),
        preset.seed,
    )


def _run_generation(task_id: str, seed: int):
    try:
        result = generate_demo(task_id, seed, catalog=CATALOG)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise gr.Error(f"Trace could not generate that selection: {str(exc)[:300]}") from exc
    return (
        result.original_image,
        result.annotation_overlay,
        result.prompt,
        result.ground_truth,
        result.reward_contract,
        result.trace_summary,
        result.public_trace,
        result.reproduction,
        result.links_markdown,
    )


def _random_question():
    selection = sample_random_selection(CATALOG)
    return (
        gr.Dropdown(
            choices=list(CATALOG.domains),
            value=selection.domain,
        ),
        gr.Dropdown(
            choices=list(CATALOG.scenes(selection.domain)),
            value=selection.scene_id,
        ),
        gr.Dropdown(
            choices=list(CATALOG.tasks(selection.domain, selection.scene_id)),
            value=selection.task_id,
        ),
        selection.seed,
        *_run_generation(selection.task_id, selection.seed),
    )


with gr.Blocks(
    title="Trace · Grounded visual reasoning",
    analytics_enabled=False,
    fill_width=True,
) as demo:
    with gr.Column(elem_classes="trace-shell"):
        gr.HTML('<div class="trace-kicker">Grounded visual reasoning · deterministic by design</div>')
        gr.Markdown(
            """
# Explore Trace

Generate any of Trace's **1,000 tasks** across **277 scenes** and **11 visual
domains**. Every image, prompt, typed answer, annotation, reward contract, and
public execution trace comes from the same deterministic state.
""",
            elem_classes="trace-title",
        )

        with gr.Row(equal_height=True):
            gr.HTML("<div class='trace-stat'><strong>1,000</strong>tasks</div>")
            gr.HTML("<div class='trace-stat'><strong>277</strong>scenes</div>")
            gr.HTML("<div class='trace-stat'><strong>11</strong>domains</div>")
            gr.HTML("<div class='trace-stat'><strong>1 seed</strong>exact replay</div>")

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Row():
                    domain = gr.Dropdown(
                        choices=list(CATALOG.domains),
                        value=DEFAULT_DOMAIN,
                        label="1 · Domain",
                        interactive=True,
                    )
                    scene_id = gr.Dropdown(
                        choices=list(CATALOG.scenes(DEFAULT_DOMAIN)),
                        value=DEFAULT_SCENE_ID,
                        label="2 · Scene",
                        interactive=True,
                    )
                task_id = gr.Dropdown(
                    choices=list(CATALOG.tasks(DEFAULT_DOMAIN, DEFAULT_SCENE_ID)),
                    value=DEFAULT_TASK_ID,
                    label="3 · Task (searchable)",
                    filterable=True,
                    interactive=True,
                )
            with gr.Column(scale=3):
                seed = gr.Number(
                    value=DEFAULT_SEED,
                    minimum=0,
                    maximum=MAX_SEED,
                    precision=0,
                    label="Seed",
                    interactive=True,
                )
                with gr.Row():
                    randomize = gr.Button("Random question", variant="secondary")
                    generate = gr.Button(
                        "Generate task",
                        variant="primary",
                        elem_classes="trace-run",
                    )
                gr.Markdown(
                    "Inputs are limited to a registered task and integer seed. "
                    "Generation uses `max_attempts=100`.",
                    elem_classes="trace-note",
                )

        preset = gr.Dropdown(
            choices=[
                (item.label, str(index))
                for index, item in enumerate(PRESETS)
            ],
            value=None,
            label="Curated gallery · 22 deterministic presets, two per domain",
            filterable=True,
            interactive=True,
        )

        with gr.Tabs():
            with gr.Tab("Problem"):
                with gr.Row():
                    original_image = gr.Image(
                        label="Generated image",
                        type="pil",
                        format="png",
                        interactive=False,
                    )
                    annotation_overlay = gr.Image(
                        label="Public annotation overlay",
                        type="pil",
                        format="png",
                        interactive=False,
                    )
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    interactive=False,
                    buttons=["copy"],
                )
            with gr.Tab("Ground truth"):
                with gr.Row():
                    ground_truth = gr.JSON(label="Typed answer and annotation")
                    reward_contract = gr.JSON(label="Reward contract")
            with gr.Tab("Execution trace"):
                with gr.Row():
                    trace_summary = gr.JSON(label="Trace summary")
                    public_trace = gr.JSON(
                        label="Full public trace",
                        open=False,
                    )
            with gr.Tab("Reproduce"):
                reproduction = gr.Code(
                    label="Exact-revision reproduction",
                    language="shell",
                    interactive=False,
                    lines=13,
                )

        links = gr.Markdown(
            "Choose a task and seed, then select **Generate task**.",
        )
        gr.Markdown(
            """
Trace uses metadata contracts—not pixels—as verifier ground truth. The overlay
is an inspection aid; the typed payload and reward contract are authoritative.

[GitHub](https://github.com/maveryn/trace) ·
[Documentation](https://maveryn.github.io/trace/) ·
[Dataset](https://huggingface.co/datasets/maveryn/trace) ·
[Paper](https://arxiv.org/abs/2607.19790)
""",
            elem_classes="trace-note",
        )

    domain.input(
        _scene_update,
        inputs=domain,
        outputs=[scene_id, task_id],
        api_name=False,
        concurrency_limit=1,
    )
    scene_id.input(
        _task_update,
        inputs=[domain, scene_id],
        outputs=task_id,
        api_name=False,
        concurrency_limit=1,
    )
    randomize.click(
        _random_question,
        outputs=[
            domain,
            scene_id,
            task_id,
            seed,
            original_image,
            annotation_overlay,
            prompt,
            ground_truth,
            reward_contract,
            trace_summary,
            public_trace,
            reproduction,
            links,
        ],
        api_name=False,
        concurrency_limit=1,
    )
    preset.change(
        _preset_update,
        inputs=preset,
        outputs=[domain, scene_id, task_id, seed],
        api_name=False,
        concurrency_limit=1,
    )
    generate.click(
        _run_generation,
        inputs=[task_id, seed],
        outputs=[
            original_image,
            annotation_overlay,
            prompt,
            ground_truth,
            reward_contract,
            trace_summary,
            public_trace,
            reproduction,
            links,
        ],
        api_name=False,
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1, max_size=32)


if __name__ == "__main__":
    demo.launch(css=_CSS, footer_links=[])
