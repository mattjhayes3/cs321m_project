"""
Benchmark Evaluation on Modal
==============================
Modal version of benchmark_evaluation.ipynb.
Runs lm-evaluation-harness on Modal GPUs with persistent volume storage.

Usage:
  # 1. Run evaluations (parallel across models)
  modal run benchmark_eval_modal.py --task-name arc_easy

  # 2. Download results to local directory
  modal run benchmark_eval_modal.py::download_results --task-name arc_easy

  # 3. Upload existing results (e.g. from Colab/Drive)
  modal run benchmark_eval_modal.py::upload_results \
      --task-name arc_easy --local-dir ./arc_easy_eval/raw_results

  # 4. Run IRT analysis on existing results
  modal run benchmark_eval_modal.py::run_irt_analysis --task-name arc_easy

Prerequisites:
  - `modal` CLI installed and authenticated (`pip install modal && modal setup`)
  - HuggingFace token stored as a Modal secret:
      modal secret create huggingface HF_TOKEN=hf_YOUR_TOKEN_HERE
"""

import json
import os
import time
from pathlib import Path

import modal

# ══════════════════════════════════════════════════════════════════
#  MODAL INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

app = modal.App("benchmark-eval")

# Persistent volume for storing raw results across runs
results_vol = modal.Volume.from_name("benchmark-eval-results", create_if_missing=True)
RESULTS_MOUNT = "/results"

# 1. Container image for running evaluations (clean, no torch_measure conflict)
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub>=1.5.0",
        "lm-eval[hf,api]",
        "accelerate",
        "bitsandbytes",
        "torch",
        "torchvision",
        "decord",
        "av",
        "transformers",
        "tokenizers",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "einops",
        "pytest",
    )
    .run_commands(
        "python3 -c \"with open('/usr/local/lib/python3.11/site-packages/lm_eval/models/hf_vlms.py', 'r') as f: code = f.read(); patched = code.replace('self.tokenizer = self.processor.tokenizer', 'self.tokenizer = self.processor.tokenizer if hasattr(self.processor, \\'tokenizer\\') else self.processor'); open('/usr/local/lib/python3.11/site-packages/lm_eval/models/hf_vlms.py', 'w').write(patched)\""
    )
)

# 2. Container image for running IRT analysis (safely handles older huggingface_hub)
irt_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "pandas",
        "scipy",
        "matplotlib",
        "torch_measure",
    )
)

# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════

def model(hf_id, short_name, params_B, needs_4bit=False,
          model_type="hf", gpu="A10G", trust_remote_code=False):
    """Define a model for evaluation.

    Args:
        gpu: Modal GPU type. Use "A10G" for ≤13B models,
             "A100" for larger models, "T4" for small models.
    """
    return dict(
        hf_id=hf_id, short_name=short_name, params_B=params_B,
        needs_4bit=needs_4bit, model_type=model_type, gpu=gpu,
        trust_remote_code=trust_remote_code,
    )


MODELS = [
    # Pythia suite (all deduped, base models)
    model("EleutherAI/pythia-70m-deduped",   "pythia-70m",   0.07, gpu="T4"),
    model("EleutherAI/pythia-160m-deduped",  "pythia-160m",  0.16, gpu="T4"),
    model("EleutherAI/pythia-410m-deduped",  "pythia-410m",  0.41, gpu="T4"),
    model("EleutherAI/pythia-1b-deduped",    "pythia-1b",    1.0,  gpu="T4"),
    model("EleutherAI/pythia-1.4b-deduped",  "pythia-1.4b",  1.4,  gpu="T4"),
    model("EleutherAI/pythia-2.8b-deduped",  "pythia-2.8b",  2.8,  gpu="T4"),
    model("EleutherAI/pythia-6.9b-deduped",  "pythia-6.9b",  6.9),
    model("EleutherAI/pythia-12b-deduped",   "pythia-12b",   12.0, gpu="A100-80GB"),

    # Instruction-tuned models (saturation band)
    model("meta-llama/Llama-2-7b-hf",       "llama2-7b",      7.0),
    model("meta-llama/Llama-2-13b-hf",      "llama2-13b",     13.0, gpu="A100-80GB"),
    model("meta-llama/Meta-Llama-3-8B-Instruct", "llama3-8b-inst",    8.0),
    model("meta-llama/Llama-3.2-1B-Instruct",    "llama3.2-1b-inst",  1.0,  gpu="T4"),
    model("meta-llama/Llama-3.2-3B-Instruct",    "llama3.2-3b-inst",  3.0,  gpu="T4"),
    model("meta-llama/Llama-3.1-8B-Instruct",    "llama3.1-8b-inst",  8.0),
    # model("meta-llama/Llama-3.2-11B-Vision-Instruct", "llama3.2-11b-vl", 11.0, gpu="A100-80GB", model_type="hf-multimodal"),

    model("microsoft/Phi-3-mini-4k-instruct",    "phi3-mini",         3.8),
    # Excluded: Phi-3-Small has an upstream RoPE config mismatch (ValueError: Field short_factor is required) in transformers v5.x dynamic loader.
    # model("microsoft/Phi-3-small-8k-instruct",   "phi3-small",        7.0, trust_remote_code=True),
    # Excluded: Phi-3-Medium has an upstream RoPE config key mismatch in transformers v5.x and deprecated remote DynamicCache classmethod dependencies.
    # model("microsoft/Phi-3-medium-128k-instruct", "phi3-medium",      14.0, gpu="A100-80GB"),
    model("mistralai/Mistral-7B-Instruct-v0.3",  "mistral-7b-inst",   7.0),
    model("mistralai/Mistral-Nemo-Instruct-2407", "mistral-nemo",     12.0, gpu="A100-80GB"),

    # GPT-2 (classic baselines)
    model("openai-community/gpt2",          "gpt2-small",     0.12, gpu="T4"),
    model("openai-community/gpt2-large",    "gpt2-large",     0.77, gpu="T4"),

    # Gemma
    model("google/gemma-3-1b-it",           "gemma3-1b-it",   1.0, gpu="T4"),
    model("google/gemma-2b",                "gemma1-2b",      2.0,  gpu="T4"),
    model("google/gemma-2-2b",              "gemma2-2b",      2.0,  gpu="T4"),
    model("google/gemma-2-9b-it",           "gemma2-9b-it",   9.0),
    model("google/gemma-7b",                "gemma1-7b",      7.0),
    model("google/gemma-3-12b-it",          "gemma3-12b-it",  12.0, gpu="A100-80GB"),



    # Qwen
    model("Qwen/Qwen2.5-3B-Instruct",      "qwen2.5-3b-inst",  3.0,  gpu="T4"),

    model("Qwen/Qwen2.5-7B-Instruct",      "qwen2.5-7b-inst",  7.0),

    model("Qwen/Qwen2.5-14B-Instruct",     "qwen2.5-14b-inst", 14.0, gpu="A100-80GB"),
    model("Qwen/Qwen2.5-Coder-14B-Instruct", "qwen2.5-coder-14b", 14.0, gpu="A100-80GB"),
    model("Qwen/Qwen3-14B",                "qwen3-14b",        14.0, gpu="A100-80GB", trust_remote_code=True),

    model("Qwen/Qwen2.5-32B-Instruct",     "qwen2.5-32b-inst", 32.0, gpu="A100-80GB"),
    # Excluded: Qwen3-32B is a base model (not instruct/chat), scoring random guessing accuracy (~25.08% on ARC-Easy).
    # model("Qwen/Qwen3-32B",                "qwen3-32b",        32.0, gpu="A100-80GB"),
    model("Qwen/Qwen3.5-27B",              "qwen3.5-27b",      27.0, gpu="A100-80GB", trust_remote_code=True),
    model("Qwen/Qwen3.5-35B-A3B",          "qwen3.5-35b",      35.0, gpu="A100-80GB", trust_remote_code=True),
]


# ══════════════════════════════════════════════════════════════════
#  SINGLE-MODEL EVALUATION (runs on GPU)
# ══════════════════════════════════════════════════════════════════

@app.function(
    image=eval_image,
    # GPU is overridden per-model via .spawn() — default A10G
    gpu="A10G",
    timeout=7200,  # 2 hours per model max
    volumes={RESULTS_MOUNT: results_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=16384,  # 16 GB RAM
)
def evaluate_single_model(m: dict, task_name: str) -> dict:
    """Evaluate a single model on the given task using lm-evaluation-harness."""
    import subprocess
    
    # Programmatically authenticate HuggingFace globally
    import os
    from huggingface_hub import login
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        
    hf_id = m["hf_id"]
    short_name = m["short_name"]
    needs_4bit = m["needs_4bit"]
    model_type = m["model_type"]

    # Output directory on the persistent volume
    save_dir = f"{RESULTS_MOUNT}/{task_name}_eval/raw_results/{short_name}"
    output_dir = f"/tmp/eval_output/{short_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Evaluating: {short_name} ({m['params_B']}B) — {hf_id}")
    print(f"  Task: {task_name}")
    print(f"{'='*70}")

    def _run_eval(mtype):
        if mtype == "openai-chat-completions":
            model_args = f"model={hf_id}"
        else:
            remote_code = "True" if m.get("trust_remote_code", False) else "False"
            model_args = f"pretrained={hf_id},dtype=float16,trust_remote_code={remote_code}"
            if needs_4bit:
                model_args += ",load_in_4bit=True"

        # Determine optimal fixed batch size to bypass slow auto-search phase
        if mtype == "hf":
            if m.get("params_B", 0.0) > 15.0:
                batch_size = "1"
            else:
                batch_size = "16"
        else:
            batch_size = "1"

        cmd = [
            "lm_eval",
            "--model", mtype,
            "--model_args", model_args,
            "--tasks", task_name,
            "--batch_size", batch_size,
            "--log_samples",
            "--output_path", output_dir,
        ]

        print(f"    Command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if res.returncode != 0:
            print(f"    ⚠️  Command failed with exit code {res.returncode}")
            if res.stdout:
                print("    [STDOUT]")
                print(res.stdout)
            if res.stderr:
                print("    [STDERR]")
                print(res.stderr)
        return res

    start_time = time.time()

    # Try primary model type
    result = _run_eval(model_type)

    # Fallback: if hf fails, try hf-multimodal (and vice versa)
    if result.returncode != 0 and model_type == "hf":
        print(f"    ⚠️  --model hf failed, retrying with hf-multimodal...")
        result = _run_eval("hf-multimodal")
    elif result.returncode != 0 and model_type == "hf-multimodal":
        print(f"    ⚠️  --model hf-multimodal failed, retrying with hf...")
        result = _run_eval("hf")

    elapsed = time.time() - start_time

    if result.returncode != 0:
        error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
        print(f"    ❌ FAILED in {elapsed:.0f}s: {error_msg[:200]}")
        return {
            "short_name": short_name, "hf_id": hf_id,
            "error": error_msg, "time_seconds": round(elapsed, 1),
        }

    # Parse the results JSON
    results_file = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if "results_" in f and ".json" in f:
                results_file = os.path.join(root, f)
                break

    if results_file is None:
        return {
            "short_name": short_name, "hf_id": hf_id,
            "error": "results JSON not found", "time_seconds": round(elapsed, 1),
        }

    with open(results_file) as f:
        results = json.load(f)

    task_results = results.get("results", {}).get(task_name, {})
    acc = task_results.get("acc,none", task_results.get("acc", None))
    acc_norm = task_results.get("acc_norm,none", task_results.get("acc_norm", None))

    print(f"  ✅ acc={acc}  acc_norm={acc_norm}  ({elapsed:.0f}s)")

    # Copy results to persistent volume
    import shutil
    os.makedirs(save_dir, exist_ok=True)
    # Copy all output files to the volume
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            src = os.path.join(root, f)
            # Preserve subdirectory structure
            rel_path = os.path.relpath(src, output_dir)
            dst = os.path.join(save_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    results_vol.commit()
    print(f"  💾 Saved to volume: {save_dir}")

    return {
        "short_name": short_name, "hf_id": hf_id,
        "acc": acc, "acc_norm": acc_norm,
        "time_seconds": round(elapsed, 1), "error": None,
    }
@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=7200,
    volumes={RESULTS_MOUNT: results_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
)
def evaluate_single_model_a100(m: dict, task_name: str) -> dict:
    """Evaluate a single model on A100-80GB GPU."""
    return evaluate_single_model.local(m, task_name)


# ══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT: Orchestrate parallel evaluation
# ══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    task_name: str = "arc_easy",
    skip_completed: bool = True,
    extra: bool = True,
):
    """Run benchmark evaluations on Modal GPUs.

    Args:
        task_name: lm-evaluation-harness task name (e.g. arc_easy, arc_challenge,
                   hellaswag, mmlu_stem).
        skip_completed: If True, skip models that already have results on the volume.
    """
    import pandas as pd

    task_display = task_name.replace("_", " ").title()
    print(f"╔{'═'*68}╗")
    print(f"║  Benchmark Evaluation on Modal")
    print(f"║  Task: {task_name} ({task_display})")
    print(f"╚{'═'*68}╝")

    all_models = MODELS 

    # Check which models are already completed on the volume
    completed = set()
    if skip_completed:
        completed = _get_completed_models(task_name)

    to_run = [m for m in all_models if m["short_name"] not in completed]

    print(f"\nTotal models defined: {len(all_models)}")
    print(f"Previously completed: {len(completed)}")
    if completed:
        for name in sorted(completed):
            print(f"  ✓ {name}")
    print(f"Models to run: {len(to_run)}")
    for m in to_run:
        print(f"  → {m['short_name']:>26s}  {m['params_B']:.2f}B  [{m['gpu']}]")

    if not to_run:
        print("\n✅ All models already completed! Nothing to do.")
        print(f"   Use --no-skip-completed to force re-run.")
        return

    # Launch evaluations in parallel using .map()
    print(f"\n🚀 Launching {len(to_run)} evaluations in parallel...")
    start = time.time()

    # Use starmap to pass (model_dict, task_name) pairs
    # Use starmap to pass (model_dict, task_name) pairs
    handles = []
    for m in to_run:
        # Route model to appropriate static GPU function
        gpu_type = m["gpu"]
        if "A100" in gpu_type:
            handle = evaluate_single_model_a100.spawn(m, task_name)
        else:
            handle = evaluate_single_model.spawn(m, task_name)
        handles.append((m, handle))

    # Collect results as they complete
    results = []
    for m, handle in handles:
        try:
            result = handle.get()
            status = "✅" if not result.get("error") else "❌"
            acc_str = f"acc={result.get('acc', 'N/A')}" if not result.get("error") else result["error"][:80]
            print(f"  {status} {result['short_name']:>26s}  {acc_str}  ({result.get('time_seconds', 0):.0f}s)")
            results.append(result)
        except Exception as e:
            print(f"  ❌ {m['short_name']:>26s}  Exception: {e}")
            results.append({
                "short_name": m["short_name"], "hf_id": m["hf_id"],
                "error": str(e), "time_seconds": 0,
            })

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  Done — {len(to_run)} models in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # Summary table
    df = pd.DataFrame([
        {
            "model": r["short_name"],
            "acc": r.get("acc"),
            "acc_norm": r.get("acc_norm"),
            "time_s": r.get("time_seconds"),
            "error": r.get("error", ""),
        }
        for r in results
    ])
    print("\n" + df.to_string(index=False))


def _get_completed_models(task_name: str) -> set:
    """Check the Modal volume for already-completed models."""
    completed = set()
    raw_dir = f"{RESULTS_MOUNT}/{task_name}_eval/raw_results"

    try:
        for entry in results_vol.listdir(f"{task_name}_eval/raw_results"):
            # entry.path is like "arc_easy_eval/raw_results/pythia-70m/..."
            parts = entry.path.split("/")
            if len(parts) >= 3:
                model_name = parts[2]
                completed.add(model_name)
    except Exception as e:
        print(f"  ⚠️  Error checking Volume for completed models: {e}")
        pass  # Volume might not have this directory yet

    return completed


# ══════════════════════════════════════════════════════════════════
#  DOWNLOAD RESULTS
# ══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def download_results(task_name: str = "arc_easy", local_dir: str = ""):
    """Download all results from the Modal volume to a local directory."""
    import subprocess

    if not local_dir:
        local_dir = f"./{task_name}_eval"

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading results for task '{task_name}' to {local_dir}/...")

    # Use modal volume get to download
    vol_path = f"{task_name}_eval/"
    cmd = ["modal", "volume", "get", "benchmark-eval-results", vol_path, local_dir]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Downloaded to {local_dir}/")
        print(result.stdout)
    else:
        print(f"❌ Download failed: {result.stderr}")
        print("  You can also download manually:")
        print(f"    modal volume get benchmark-eval-results {vol_path} {local_dir}")


# ══════════════════════════════════════════════════════════════════
#  UPLOAD EXISTING RESULTS
# ══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def upload_results(task_name: str = "arc_easy", local_dir: str = ""):
    """Upload existing results (e.g. from Colab/Drive) to the Modal volume."""
    import subprocess

    if not local_dir:
        local_dir = f"./{task_name}_eval/raw_results"

    if not os.path.exists(local_dir):
        print(f"❌ Local directory not found: {local_dir}")
        print(f"   Provide the path to your raw_results directory.")
        return

    vol_path = f"{task_name}_eval/raw_results"
    cmd = ["modal", "volume", "put", "benchmark-eval-results", local_dir, vol_path]
    print(f"Uploading {local_dir} → volume:{vol_path}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Uploaded successfully!")
        print(result.stdout)
    else:
        print(f"❌ Upload failed: {result.stderr}")
        print("  You can also upload manually:")
        print(f"    modal volume put benchmark-eval-results {local_dir} {vol_path}")


# ══════════════════════════════════════════════════════════════════
#  IRT ANALYSIS (runs remotely on Modal)
# ══════════════════════════════════════════════════════════════════

@app.function(
    image=irt_image,
    gpu="T4",
    timeout=600,
    volumes={RESULTS_MOUNT: results_vol},
    memory=16384,
)
def _run_irt_analysis_remote(task_name: str) -> dict:
    """Run IRT analysis on existing results stored in the volume."""
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import expit

    task_display = task_name.replace("_", " ").title()
    save_dir = f"{RESULTS_MOUNT}/{task_name}_eval"
    raw_dir = f"{save_dir}/raw_results"

    results_vol.reload()

    # ── Parse per-sample results ──────────────────────────────────
    def load_per_sample_results(output_dir, short_name):
        sample_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if task_name in f and (f.endswith(".jsonl") or f.endswith(".json")):
                    sample_files.append(os.path.join(root, f))
        if not sample_files:
            print(f"  ⚠️  No sample file found for {short_name}")
            return pd.DataFrame()

        sample_file = sample_files[0]
        print(f"  Loading: {short_name} ← {os.path.basename(sample_file)}")

        records = []
        with open(sample_file) as f:
            content = f.read().strip()
            if content.startswith("["):
                samples = json.loads(content)
            else:
                samples = [json.loads(line) for line in content.split("\n") if line.strip()]

        for i, sample in enumerate(samples):
            doc = sample.get("doc", {})
            item_id = doc.get("id", doc.get("question_id", f"item_{i}"))
            correct = sample.get("acc", sample.get("exact_match", None))
            log_likelihoods = sample.get("filtered_resps", sample.get("resps", None))
            records.append({
                "item_id": str(item_id),
                "correct": int(correct) if correct is not None else None,
                "log_likelihoods": json.dumps(log_likelihoods) if log_likelihoods else None,
                "question": doc.get("question", ""),
            })
        df = pd.DataFrame(records)
        df["model"] = short_name
        return df

    # Scan volume for completed models
    all_samples = []
    if os.path.exists(raw_dir):
        model_dirs = sorted([d for d in os.listdir(raw_dir)
                            if os.path.isdir(os.path.join(raw_dir, d))])
        print(f"Found {len(model_dirs)} model results on volume:")

        for short_name in model_dirs:
            model_path = os.path.join(raw_dir, short_name)
            df = load_per_sample_results(model_path, short_name)
            if len(df) > 0:
                all_samples.append(df)

    if not all_samples:
        return {"error": "No sample data found!"}

    samples_df = pd.concat(all_samples, ignore_index=True)
    print(f"\n✅ Parsed {len(samples_df)} total sample records "
          f"({samples_df['model'].nunique()} models × "
          f"{samples_df['item_id'].nunique()} items)")

    # ── Build response matrix ─────────────────────────────────────
    response_matrix = samples_df.pivot_table(
        index="model", columns="item_id", values="correct", aggfunc="first"
    )
    model_acc = response_matrix.mean(axis=1).sort_values()
    response_matrix = response_matrix.loc[model_acc.index]

    print(f"Response matrix shape: {response_matrix.shape}")

    # ── Save response matrix and scores ───────────────────────────
    response_matrix.to_csv(f"{save_dir}/response_matrix.csv")
    scores_df = pd.DataFrame({"model": model_acc.index, "acc": model_acc.values})
    scores_df.to_csv(f"{save_dir}/model_scores.csv", index=False)
    samples_df.to_csv(f"{save_dir}/all_samples_long.csv", index=False)

    item_difficulty = response_matrix.mean(axis=0).sort_values()
    item_diff_df = pd.DataFrame({
        "item_id": item_difficulty.index, "p_value": item_difficulty.values,
    })
    item_diff_df.to_csv(f"{save_dir}/item_difficulty.csv", index=False)

    # ── IRT: Rasch (1PL) fitting ──────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = response_matrix.values.astype(float)
    n_models, n_items = R.shape
    model_names = list(response_matrix.index)
    item_ids = list(response_matrix.columns)

    item_var = np.nanvar(R, axis=0)
    valid_mask = item_var > 0
    R_valid = R[:, valid_mask]
    valid_item_ids = [item_ids[i] for i in range(n_items) if valid_mask[i]]
    n_valid = valid_mask.sum()
    print(f"Items with variance > 0: {n_valid} / {n_items}")

    import sys
    from unittest.mock import MagicMock
    sys.modules["tabpfn"] = MagicMock()
    sys.modules["pyro"] = MagicMock()
    sys.modules["pyro.distributions"] = MagicMock()
    from torch_measure.models import Rasch

    R_tensor = torch.tensor(R_valid, dtype=torch.float32, device=device)
    R_tensor = torch.nan_to_num(R_tensor, nan=0.0)

    irt_model = Rasch(n_subjects=n_models, n_items=n_valid, device=device)
    history = irt_model.fit(R_tensor, method="mle", max_epochs=2000, lr=0.01, verbose=True)

    with torch.no_grad():
        thetas = irt_model.ability.cpu().numpy()
        b_raw = irt_model.difficulty.cpu().numpy()
        # Build grid query to predict for all subject-item combinations
        sub_grid = torch.arange(n_models, device=device)[:, None].expand(n_models, n_valid).flatten()
        item_grid = torch.arange(n_valid, device=device)[None, :].expand(n_models, n_valid).flatten()
        query = {"subject_idx": sub_grid, "item_idx": item_grid}
        predicted_probs = irt_model.predict(query).reshape(n_models, n_valid).cpu().numpy()

    # Parameterization check
    manual_traditional = expit(thetas[:, None] - b_raw[None, :])
    manual_intercept   = expit(thetas[:, None] + b_raw[None, :])
    err_traditional = np.abs(predicted_probs - manual_traditional).mean()
    err_intercept   = np.abs(predicted_probs - manual_intercept).mean()

    if err_intercept < err_traditional:
        b_params = -b_raw
    else:
        b_params = b_raw

    a_params = np.ones(n_valid)
    c_params = np.zeros(n_valid)

    # ── Fisher Information and SEs ────────────────────────────────
    def icc_rasch(theta, b):
        return expit(theta - b)

    def item_information_rasch(theta, b):
        p = icc_rasch(theta, b)
        return p * (1 - p)

    def test_information(theta_grid, a_params, b_params, c_params):
        I_total = np.zeros_like(theta_grid)
        for b in b_params:
            I_total += item_information_rasch(theta_grid, b)
        return I_total

    def ability_se(theta, a_params, b_params, c_params):
        I = test_information(np.array([theta]), a_params, b_params, c_params)[0]
        return 1.0 / np.sqrt(I) if I > 0 else np.inf

    theta_ses = np.array([ability_se(t, a_params, b_params, c_params) for t in thetas])

    # ── Save IRT results ──────────────────────────────────────────
    irt_df = pd.DataFrame({
        "model": model_names, "theta": thetas, "se_theta": theta_ses,
        "ci95_lo": thetas - 1.96 * theta_ses, "ci95_hi": thetas + 1.96 * theta_ses,
    }).sort_values("theta")
    irt_df.to_csv(f"{save_dir}/irt_ability_estimates.csv", index=False)

    item_params_df = pd.DataFrame({
        "item_id": valid_item_ids, "b_difficulty": b_params,
    })
    item_params_df.to_csv(f"{save_dir}/irt_item_parameters.csv", index=False)

    # ── Generate plots ────────────────────────────────────────────
    N_ITEMS = response_matrix.shape[1]
    theta_grid = np.linspace(-4, 4, 500)

    # Plot 1: Summary plots (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Accuracy by size
    ax = axes[0]
    all_model_defs = MODELS
    model_info = []
    for m_def in all_model_defs:
        short = m_def["short_name"]
        params = m_def["params_B"]
        if short in model_acc.index:
            p = model_acc[short]
            se = np.sqrt(p * (1 - p) / N_ITEMS)
            ci95 = 1.96 * se
            is_base = any(tag in short for tag in ["pythia", "gpt2"])
            is_instruct = not is_base
            model_info.append((params, p, ci95, short, is_base, is_instruct))

    if model_info:
        model_info.sort(key=lambda x: x[0])
        for label, color, marker, sel in [
            ("Base (Pythia/GPT-2)", "#4A90D9", "o", lambda b, i: b),
            ("Instruction-tuned", "#E74C3C", "^", lambda b, i: i),
        ]:
            subset = [(p, acc, ci, n) for p, acc, ci, n, b, i in model_info if sel(b, i)]
            if not subset:
                continue
            ps, accs, cis, names = zip(*subset)
            ax.errorbar(ps, accs, yerr=cis, fmt=marker, color=color,
                       markersize=7, capsize=3, capthick=1, elinewidth=1,
                       label=label, zorder=3, alpha=0.85)
            for p, a, n in zip(ps, accs, names):
                ax.annotate(n, (p, a), fontsize=5, rotation=25,
                           xytext=(4, 6), textcoords="offset points", alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)", fontsize=11)
    ax.set_ylabel(f"{task_display} Accuracy", fontsize=11)
    ax.set_title("Accuracy vs Model Size (95% CI)", fontsize=12)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylim(0.15, 1.02)

    # Item difficulty distribution
    ax = axes[1]
    ax.hist(item_difficulty.values, bins=30, color="#2ECC71", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Item p-value (fraction correct)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Item Difficulty Distribution", fontsize=12)
    ax.axvline(x=item_difficulty.mean(), color="red", linestyle="--",
              label=f"Mean = {item_difficulty.mean():.3f}")
    ax.legend(fontsize=8)

    # Response matrix heatmap
    ax = axes[2]
    n_show = min(100, response_matrix.shape[1])
    item_order = item_difficulty.index
    step = max(1, len(item_order) // n_show)
    sampled_items = item_order[::step][:n_show]
    mat_show = response_matrix[sampled_items].values.astype(float)
    im = ax.imshow(mat_show, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel(f"Items (sampled {n_show}, sorted by difficulty →)", fontsize=10)
    ax.set_ylabel("Models (sorted by accuracy →)", fontsize=10)
    ax.set_yticks(range(len(response_matrix)))
    ax.set_yticklabels(response_matrix.index, fontsize=5)
    ax.set_title("Response Matrix", fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/summary_plots.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: IRT analysis (4-panel)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Test Information Function
    ax = axes[0, 0]
    I_total = test_information(theta_grid, a_params, b_params, c_params)
    ax.plot(theta_grid, I_total, color="#2C3E50", linewidth=2)
    ax.fill_between(theta_grid, I_total, alpha=0.15, color="#3498DB")
    for idx in range(n_models):
        t = thetas[idx]
        I_at_t = test_information(np.array([t]), a_params, b_params, c_params)[0]
        color = "#E74C3C" if t > 1.5 else "#4A90D9"
        ax.plot(t, I_at_t, 'o', color=color, markersize=5, zorder=3)
        ax.annotate(model_names[idx], (t, I_at_t), fontsize=4.5,
                   rotation=40, xytext=(3, 5), textcoords="offset points")
    ax.set_xlabel("Ability (θ)", fontsize=11)
    ax.set_ylabel("Test Information I(θ)", fontsize=11)
    ax.set_title("Test Information Function (Rasch/1PL)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Ability estimates with CIs
    ax = axes[0, 1]
    order = np.argsort(thetas)
    for i, idx in enumerate(order):
        name = model_names[idx]
        ci = 1.96 * theta_ses[idx]
        color = "#4A90D9" if any(tag in name for tag in ["pythia", "gpt2"]) else "#E74C3C"
        ax.errorbar(thetas[idx], i, xerr=ci,
                   fmt='o', color=color, markersize=6,
                   capsize=4, capthick=1.2, elinewidth=1.2, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([model_names[idx] for idx in order], fontsize=7)
    ax.set_xlabel("Ability (θ)", fontsize=11)
    ax.set_title("IRT Ability Estimates (95% CI)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

    # SE function
    ax = axes[1, 0]
    SE_grid = 1.0 / np.sqrt(np.maximum(I_total, 1e-10))
    ax.plot(theta_grid, SE_grid, color="#8E44AD", linewidth=2)
    ax.fill_between(theta_grid, SE_grid, alpha=0.1, color="#8E44AD")
    for idx in range(n_models):
        t = thetas[idx]
        se = theta_ses[idx]
        color = "#E74C3C" if t > 1.5 else "#4A90D9"
        ax.plot(t, se, 'o', color=color, markersize=5, zorder=3)
        ax.annotate(model_names[idx], (t, se), fontsize=4.5,
                   rotation=30, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Ability (θ)", fontsize=11)
    ax.set_ylabel("SE(θ) = 1/√I(θ)", fontsize=11)
    ax.set_title("Standard Error Function", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(2.0, SE_grid[50:-50].max() * 1.3))

    # Difficulty distribution
    ax = axes[1, 1]
    ax.hist(b_params, bins=30, color="#27AE60", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, label="θ = 0")
    ax.axvline(x=b_params.mean(), color="blue", linestyle="--", linewidth=1.5,
              label=f"Mean b = {b_params.mean():.2f}")
    for idx in range(n_models):
        ax.axvline(x=thetas[idx], color="#E74C3C", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Difficulty (b)", fontsize=11)
    ax.set_ylabel("Count (items)", fontsize=11)
    ax.set_title("Item Difficulty Distribution vs Model Abilities", fontsize=12)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/irt_rasch_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()

    results_vol.commit()

    # Print summary
    print(f"\n── IRT Ability Estimates ──\n")
    print(f"{'Model':>22s}  {'θ':>7s}  {'SE(θ)':>7s}  {'95% CI':>20s}")
    print("-" * 65)
    for _, row in irt_df.iterrows():
        print(f"{row['model']:>22s}  {row['theta']:7.3f}  {row['se_theta']:7.4f}  "
              f"[{row['ci95_lo']:7.3f}, {row['ci95_hi']:7.3f}]")

    return {
        "n_models": n_models, "n_items": n_items, "n_valid": int(n_valid),
        "theta_range": [float(thetas.min()), float(thetas.max())],
        "b_range": [float(b_params.min()), float(b_params.max())],
        "files_saved": [
            "response_matrix.csv", "model_scores.csv", "all_samples_long.csv",
            "item_difficulty.csv", "irt_ability_estimates.csv",
            "irt_item_parameters.csv", "summary_plots.png", "irt_rasch_analysis.png",
        ],
    }


@app.local_entrypoint()
def run_irt_analysis(task_name: str = "arc_easy"):
    """Run IRT analysis on existing results."""
    print(f"Running IRT analysis for task: {task_name}")
    result = _run_irt_analysis_remote.remote(task_name)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return

    print(f"\n✅ IRT analysis complete!")
    print(f"  Models: {result['n_models']}, Items: {result['n_items']} ({result['n_valid']} valid)")
    print(f"  θ range: [{result['theta_range'][0]:.3f}, {result['theta_range'][1]:.3f}]")
    print(f"  b range: [{result['b_range'][0]:.3f}, {result['b_range'][1]:.3f}]")
    print(f"\nFiles saved to volume (download with):")
    print(f"  modal run benchmark_eval_modal.py::download_results --task-name {task_name}")
