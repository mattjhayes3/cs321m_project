"""
Modal Image Definition
========================
Separated from main.py so that edits to the orchestration logic
do NOT invalidate the cached image.

Model weights are served from the persistent Modal Volume (hf-cache-volume),
NOT baked into the image. This means:
  - Source file edits only rebuild the lightweight file-copy layers (~seconds)
  - No multi-GB downloads during image builds
  - The pre_download entrypoint in main.py populates the volume independently

The image only rebuilds when:
  - pip dependencies change
  - This file itself changes
  - Local source files change (fast, no downloads)
"""

import modal


# ────────────────────────────────────────────────────────────────
# IMAGE DEFINITION
# ────────────────────────────────────────────────────────────────
# Local files are added with copy=True so they become part of the
# image layer. Model weights live on the volume, not in the image.

loop_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub>=1.5.0",
        "lm-eval[hf,api]",
        "tokenizers",
        "pandas",
        "numpy",
        "scipy",
        "einops",
        "openai",
        "anthropic",
        "datasets",
        "matplotlib",
        "sentence-transformers",
        "scikit-learn",
    )
    .add_local_file("interfaces.py", "/root/interfaces.py", copy=True)
    .add_local_file("irt_model.py", "/root/irt_model.py", copy=True)
    .add_local_file("question_generator.py", "/root/question_generator.py", copy=True)
    .add_local_file("prompter.py", "/root/prompter.py", copy=True)
    .add_local_file("verifier.py", "/root/verifier.py", copy=True)
    .add_local_file("target_selector.py", "/root/target_selector.py", copy=True)
    .add_local_file("call_llm.py", "/root/call_llm.py", copy=True)
    .add_local_file("utils.py", "/root/utils.py", copy=True)
    .add_local_file("evaluate_generated.py", "/root/evaluate_generated.py", copy=True)
    .add_local_file("benchmark.py", "/root/benchmark.py", copy=True)
    .add_local_file("main.py", "/root/main.py", copy=True)
    .add_local_file("modal_image.py", "/root/modal_image.py", copy=True)
    .add_local_dir("../Competition/torch_measure/src/torch_measure", "/root/torch_measure")
)
