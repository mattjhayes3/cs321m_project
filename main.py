"""
Active Question Generation Loop
=================================
Runs the full active learning loop inside a single Modal A100 container:

    repeat for max_rounds:
        1. Select target difficulty (CPU, <1ms)
        2. Generate 1-3 questions via LLM API (network, ~2s)
        3. For each model: load → score questions → unload (GPU, ~1-2 min/model)
        4. Anchored IRT refit: update θ, estimate new b, hold old b fixed (CPU, <1ms)
        5. Compute separability confidence (Wald test on θ difference)
        6. Log θ changes, difficulty calibration error, and confidence

Everything stays in one container to avoid paying N cold-start penalties
per iteration. The GPU is only active during model scoring.

Each run creates a timestamped log folder on the Modal volume with:
    config.json         — full run configuration
    round_N.json        — structured output for each round
    summary.json        — final summary with θ trajectories
    stdout.log          — captured stdout/stderr

Usage:
    modal run main.py                          # run with defaults
    modal run main.py --max-rounds 10          # 10 iterations
    modal run main.py --questions-per-round 1  # 1 question per round
    modal run main.py --seed 42                # fixed seed for reproducibility
"""

import os
import sys
import io
import json
import time
import random
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any

import modal
import torch
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

from interfaces import (
    Question, Benchmark, TargetProfile, QuestionGeneratorConfig,
    NearbyExamplePrompterConfig, ScaledExamplePrompterConfig, PresentationStyle,
    TrivialVerifierConfig, MidpointTargetSelectorConfig, LLMTrace
)
from irt_model import RaschModel
from question_generator import QuestionGenerator
from evaluate_generated import (
    MODEL_REGISTRY, _score_options, _format_arc_easy_prompt
)

# ────────────────────────────────────────────────────────────────
# MODEL SUBSET — uncomment models to include in evaluation
# ────────────────────────────────────────────────────────────────

MODEL_SUBSET = [
    # Model Name            # ARC-Easy  ARC-Challenge  GPQA (Zero-Shot)
    "gemma1-2b",            # ✓         ✓              ✓
    # "gemma1-2b-it",         # ✓         ✓              
    "gemma1-7b",            # ✓         ✓              ✓
    # "gemma1-7b-it",         # ✓         ✓              
    "gemma2-2b",            # ✓         ✓              ✓
    # "gemma2-2b-it",         # ✓         ✓              
    "gemma2-9b-it",         # ✓         ✓              ✓
    # "gemma3-12b-it",        # ✓         ✓              ✓
    "gemma3-1b-it",         # ✓         ✓              ✓
    # "gemma3-4b-it",         # ✓                        
    "gpt2-large",           # ✓         ✓              ✓
    "gpt2-small",           # ✓         ✓              ✓
    "llama2-13b",           # ✓         ✓              ✓
    "llama2-7b",            # ✓         ✓              ✓
    "llama3-8b-inst",       # ✓         ✓              ✓
    "llama3.1-8b-inst",     # ✓         ✓              ✓
    "llama3.2-1b-inst",     # ✓         ✓              ✓
    "llama3.2-3b-inst",     # ✓         ✓              ✓
    "mistral-7b-inst",      # ✓         ✓              ✓
    "mistral-nemo",         # ✓         ✓              ✓
    "phi3-mini",            # ✓         ✓              ✓
    "pythia-1.4b",          # ✓         ✓              ✓
    "pythia-12b",           # ✓         ✓              ✓
    "pythia-160m",          # ✓         ✓              ✓
    "pythia-1b",            # ✓         ✓              ✓
    "pythia-2.8b",          # ✓         ✓              ✓
    "pythia-410m",          # ✓         ✓              ✓
    "pythia-6.9b",          # ✓         ✓              ✓
    "pythia-70m",           # ✓         ✓              ✓
    "qwen2.5-14b-inst",     # ✓         ✓              ✓
    "qwen2.5-32b-inst",     # ✓         ✓              ✓
    "qwen2.5-3b-inst",      # ✓         ✓              ✓
    "qwen2.5-7b-inst",      # ✓         ✓              ✓
    "qwen2.5-coder-14b",    # ✓         ✓              ✓
    "qwen3-14b",            # ✓         ✓              ✓
    # "qwen3-32b",            # ✓         ✓              
    "qwen3.5-27b",          # ✓         ✓              ✓
    "qwen3.5-35b"           # ✓         ✓              ✓
]

# ────────────────────────────────────────────────────────────────
# MODAL INFRASTRUCTURE
# ────────────────────────────────────────────────────────────────

app = modal.App("active-question-generation")

results_vol = modal.Volume.from_name("benchmark-eval-results", create_if_missing=True)
RESULTS_MOUNT = "/results"

hf_cache_vol = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)

# Image defined in modal_image.py so edits to main.py don't trigger rebuilds
from modal_image import loop_image

# ────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ────────────────────────────────────────────────────────────────

def set_all_seeds(seed: int):
    """Set random seeds for numpy, Python random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ────────────────────────────────────────────────────────────────
# BENCHMARK LOADING
# ────────────────────────────────────────────────────────────────

class RealBenchmark(Benchmark):
    """Concrete Benchmark backed by real ARC-Easy calibrated data."""
    def get_response_matrix(self, model_ids: List[str]) -> pd.DataFrame:
        return pd.DataFrame()


def load_calibrated_benchmark(volume):
    """
    Load calibrated IRT parameters + original response matrix from Volume.

    Returns:
        (benchmark, original_response_matrix)
    """
    from datasets import load_dataset

    print("=== Loading Calibrated Benchmark ===")

    # Read ability estimates (using optimal Rasch parameters)
    ability_bytes = b""
    for chunk in volume.read_file("arc_easy_eval/theta_comparison_comprehensive_acc_norm.csv"):
        ability_bytes += chunk
    ability_df = pd.read_csv(io.BytesIO(ability_bytes))

    # Read item parameters
    try:
        item_bytes = b""
        for chunk in volume.read_file("arc_easy_eval/item_parameters_comprehensive_acc_norm.csv"):
            item_bytes += chunk
        item_params_df = pd.read_csv(io.BytesIO(item_bytes))
        print("  Loaded comprehensive psychometric item parameters.")
    except Exception as e:
        raise FileNotFoundError(
            f"❌ Critical Error: Could not load comprehensive psychometric item parameters "
            f"(item_parameters_comprehensive_acc_norm.csv) from volume: {e}"
        ) from e

    # Read original response matrix
    try:
        resp_bytes = b""
        for chunk in volume.read_file("arc_easy_eval/response_matrix_cumulative_acc_norm.csv"):
            resp_bytes += chunk
        original_response_matrix = pd.read_csv(io.BytesIO(resp_bytes), index_col=0)
        print("  Loaded cumulative acc_norm response matrix.")
    except Exception as e:
        raise FileNotFoundError(
            f"❌ Critical Error: Could not load cumulative response matrix "
            f"(response_matrix_cumulative_acc_norm.csv) from volume: {e}"
        ) from e

    # Load ARC-Easy questions from HuggingFace
    print("  Loading ARC-Easy questions from HuggingFace...")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train+validation+test")

    # Build RaschModel
    irt = RaschModel()

    for _, row in ability_df.iterrows():
        model_name = str(row["Model"])
        if MODEL_SUBSET and model_name not in MODEL_SUBSET:
            continue
        # Load Rasch_Theta (1PL MLE ability estimate) as standard subject ability
        irt.thetas[model_name] = float(row["Rasch_Theta"])

    # Map Rasch difficulties and 2PL EM discriminations (for discernability filtering)
    for _, row in item_params_df.iterrows():
        item_id = str(row["item_id"])
        irt.difficulties[item_id] = float(row["Rasch_Difficulty"])
        irt.discriminations[item_id] = float(row["EM_Discrimination"])

    irt.valid_items = list(irt.difficulties.keys())

    # Reconstruct question pool
    calibrated_questions = []
    calibrated_set = set(irt.valid_items)

    for row in arc_ds:
        item_id = str(row["id"])
        if item_id in calibrated_set:
            choices = row["choices"]
            options = list(choices["text"])
            correct_ans = str(row["answerKey"])
            if correct_ans.isdigit():
                correct_ans = chr(65 + int(correct_ans) - 1)

            q = Question(
                id=item_id,
                question_text=str(row["question"]),
                options=options,
                correct_answer=correct_ans,
                difficulty=irt.difficulties[item_id],
                discrimination=irt.discriminations[item_id],
                factor_loadings=None,
                calibrated=True
            )
            calibrated_questions.append(q)

    print(f"  Loaded {len(calibrated_questions)} calibrated questions, "
          f"{len(irt.thetas)} model abilities")

    # Filter original response matrix to MODEL_SUBSET
    if original_response_matrix is not None:
        if MODEL_SUBSET:
            available = [m for m in MODEL_SUBSET if m in original_response_matrix.index]
            original_response_matrix = original_response_matrix.loc[available]
        print(f"  Original response matrix: {original_response_matrix.shape}")

    benchmark = RealBenchmark(calibrated_questions, irt)
    return benchmark, original_response_matrix


def _bin_pack_models(
    model_names: List[str],
    vram_budget_gb: float = 80.0,
    safety_margin_gb: float = 4.0,
) -> List[List[str]]:
    """
    First-fit-decreasing bin-packing of models into VRAM batches.

    Groups models so that each batch's total estimated VRAM fits within
    (vram_budget_gb - safety_margin_gb). Models that exceed the budget
    alone get their own singleton batch.

    Returns a list of batches, where each batch is a list of model names.
    """
    usable = vram_budget_gb - safety_margin_gb

    # Sort by VRAM descending (first-fit-decreasing heuristic)
    models_with_vram = []
    for name in model_names:
        reg = MODEL_REGISTRY.get(name)
        vram = reg.get("vram_gb", 10.0) if reg else 10.0
        models_with_vram.append((name, vram))
    models_with_vram.sort(key=lambda x: x[1], reverse=True)

    batches: List[List[str]] = []
    batch_usage: List[float] = []

    for name, vram in models_with_vram:
        placed = False
        for i, used in enumerate(batch_usage):
            if used + vram <= usable:
                batches[i].append(name)
                batch_usage[i] += vram
                placed = True
                break
        if not placed:
            batches.append([name])
            batch_usage.append(vram)

    return batches


def evaluate_questions_local(
    questions: List[Dict[str, Any]],
    model_names: List[str],
    safety_margin_gb: float = 4.0,
) -> pd.DataFrame:
    """
    Evaluates generated questions across models using VRAM bin-packing.

    Models are grouped into batches that fit within the A100-80GB budget.
    All models in a batch are loaded simultaneously, scored, then freed
    together — minimizing redundant load/unload cycles for small models.

    If a batch triggers an OOM during loading, all loaded models are flushed
    and the remaining unscored models from that batch are evaluated
    one-at-a-time as a fallback.
    """
    import torch
    import gc
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    all_score_details = {}  # model_name -> {question_id -> scoring breakdown}

    def _load_model(model_name):
        """Load a single model. Returns (hf_model, tokenizer) or raises."""
        reg = MODEL_REGISTRY[model_name]
        hf_id = reg["hf_id"]
        trust_remote_code = reg.get("trust_remote_code", False)

        tokenizer_kwargs = {}
        if "mistral" in hf_id.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code, **tokenizer_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        hf_model.eval()
        return hf_model, tokenizer

    def _score_model(model_name, hf_model, tokenizer):
        """Score all questions on a single model. Returns result_row dict + detailed scoring."""
        result_row = {}
        score_details = {}  # question_id -> full scoring breakdown
        n_correct = 0
        n_correct_norm = 0

        for q in questions:
            score_result = _score_options(
                hf_model, tokenizer, q["question_text"], q["options"], device
            )
            pred_idx = score_result["pred_idx_norm"]
            pred_letter = chr(65 + pred_idx)
            expected = q["correct_answer"].strip().upper()
            correct = int(pred_letter == expected)
            n_correct_norm += correct
            result_row[q["id"]] = correct

            pred_letter_raw = chr(65 + score_result["pred_idx"])
            n_correct += int(pred_letter_raw == expected)

            # Preserve full scoring breakdown for this (model, question) pair
            score_details[q["id"]] = {
                "scores": score_result["scores"],            # raw log-likelihoods per option
                "scores_norm": score_result["scores_norm"],  # length-normalized log-likelihoods
                "pred_idx": score_result["pred_idx"],        # argmax raw
                "pred_idx_norm": score_result["pred_idx_norm"],  # argmax norm
                "pred_letter": pred_letter,                  # predicted answer (acc_norm)
                "pred_letter_raw": pred_letter_raw,          # predicted answer (acc)
                "expected": expected,
                "correct_norm": correct,
                "correct_raw": int(pred_letter_raw == expected),
            }

        acc_norm = n_correct_norm / len(questions) if questions else 0.0
        acc_raw = n_correct / len(questions) if questions else 0.0
        return result_row, acc_norm, acc_raw, score_details

    def _flush_gpu(loaded_dict):
        """Unload all models and clear GPU memory."""
        for k in list(loaded_dict.keys()):
            del loaded_dict[k]
        loaded_dict.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _eval_single_fallback(model_name):
        """Fallback: load→score→unload a single model."""
        nonlocal models_done
        reg = MODEL_REGISTRY.get(model_name)
        if not reg:
            rows[model_name] = {q["id"]: float("nan") for q in questions}
            return
        t0 = time.time()
        try:
            hf_model, tokenizer = _load_model(model_name)
            load_time = time.time() - t0
            result_row, acc_norm, acc_raw, sd = _score_model(model_name, hf_model, tokenizer)
            models_done += 1
            elapsed = time.time() - t0
            print(f"    ✅ [{models_done}/{len(model_names)}] {model_name}: {acc_norm:.0%} (acc_norm) [{acc_raw:.0%} raw] in {elapsed:.1f}s (fallback, load={load_time:.1f}s)")
            rows[model_name] = result_row
            all_score_details[model_name] = sd
        except Exception as e:
            print(f"    ❌ {model_name} failed in fallback: {e}")
            rows[model_name] = {q["id"]: float("nan") for q in questions}
        finally:
            for v in ['hf_model', 'tokenizer']:
                if v in locals():
                    del locals()[v]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    batches = _bin_pack_models(model_names, vram_budget_gb=80.0, safety_margin_gb=safety_margin_gb)
    print(f"\n  🚀 VRAM bin-packing: {len(model_names)} models → {len(batches)} batches")
    for bi, batch in enumerate(batches):
        vram_est = sum(MODEL_REGISTRY.get(m, {}).get("vram_gb", 10.0) for m in batch)
        print(f"    Batch {bi+1}: {batch} (est. {vram_est:.1f}GB)")

    models_done = 0
    eval_start = time.time()

    for bi, batch in enumerate(batches):
        batch_vram = sum(MODEL_REGISTRY.get(m, {}).get("vram_gb", 10.0) for m in batch)
        print(f"\n  ── Batch {bi+1}/{len(batches)}: loading {len(batch)} models (est. {batch_vram:.1f}GB) ──")
        batch_start = time.time()

        # Phase 1: Try to load all models in this batch
        loaded = {}  # model_name -> (hf_model, tokenizer)
        fallback_needed = []
        oom_hit = False

        for model_name in batch:
            if model_name not in MODEL_REGISTRY:
                print(f"    ⚠️  Unknown model: {model_name}, skipping.")
                continue
            if oom_hit:
                # Previous load OOM'd; defer remaining models to fallback
                fallback_needed.append(model_name)
                continue

            t0 = time.time()
            try:
                hf_model, tokenizer = _load_model(model_name)
                load_time = time.time() - t0
                print(f"    ✓ Loaded {model_name} in {load_time:.1f}s")
                loaded[model_name] = (hf_model, tokenizer)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"    ⚠️ OOM loading {model_name} — flushing batch, falling back to sequential")
                    oom_hit = True
                    fallback_needed.append(model_name)
                    # Flush everything loaded so far, score what we have first
                else:
                    print(f"    ❌ Failed to load {model_name}: {e}")
                    rows[model_name] = {q["id"]: float("nan") for q in questions}
            except Exception as e:
                print(f"    ❌ Failed to load {model_name}: {e}")
                rows[model_name] = {q["id"]: float("nan") for q in questions}

        # Phase 2: Score all successfully loaded models
        for model_name, (hf_model, tokenizer) in loaded.items():
            models_done += 1
            t0 = time.time()
            try:
                result_row, acc_norm, acc_raw, sd = _score_model(model_name, hf_model, tokenizer)
                elapsed = time.time() - t0
                print(f"    ✅ [{models_done}/{len(model_names)}] {model_name}: {acc_norm:.0%} (acc_norm) [{acc_raw:.0%} raw] scored in {elapsed:.1f}s")
                rows[model_name] = result_row
                all_score_details[model_name] = sd
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"    ⚠️ OOM scoring {model_name}, will retry in fallback")
                fallback_needed.append(model_name)
                models_done -= 1  # will re-increment in fallback

        # Phase 3: Unload entire batch
        _flush_gpu(loaded)

        # Phase 4: Fallback — evaluate OOM'd models one-at-a-time
        if fallback_needed:
            print(f"    🔄 Fallback: evaluating {len(fallback_needed)} models sequentially")
            for model_name in fallback_needed:
                _eval_single_fallback(model_name)

        batch_elapsed = time.time() - batch_start
        print(f"  ── Batch {bi+1} done in {batch_elapsed:.1f}s ──")

    total_elapsed = time.time() - eval_start
    print(f"\n  🏁 All {len(model_names)} models evaluated in {total_elapsed:.1f}s ({len(batches)} batches)")


    return pd.DataFrame.from_dict(rows, orient="index"), all_score_details


# ────────────────────────────────────────────────────────────────
# LOGGING HELPERS
# ────────────────────────────────────────────────────────────────

def create_run_dir(base_path: str, run_id: str) -> str:
    """Create a timestamped log directory for this run."""
    run_dir = os.path.join(base_path, "active_loop_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path: str, data: dict):
    """Write a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ────────────────────────────────────────────────────────────────
# MAIN ACTIVE LOOP
# ────────────────────────────────────────────────────────────────

@app.function(
    image=loop_image,
    gpu="A100-80GB",
    timeout=14400,  # 4 hours max
    volumes={
        RESULTS_MOUNT: results_vol,
        "/hf_cache": hf_cache_vol
    },
    env={"HF_HOME": "/hf_cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("openai"),
    ],
    memory=16384,
)
def run_active_loop(
    max_rounds: int = 3,
    questions_per_round: int = 5,
    num_generation_steps: int = 10,
    generator_model: str = "openai/gpt-5.5",
    seed: int = 42,
) -> str:
    """
    Full active question generation loop on a single A100.

    Each round:
      1. Identify target pairs in the Saturation Band (θ > 0.5, confidence < 95%)
      2. Spaced target midpoint selection
      3. Generate questions via LLM API
      4. Evaluate on all models in MODEL_SUBSET (sequential GPU scoring)
      5. Anchored Rasch (1PL) refit (update θ, estimate new b, hold old b fixed)
      6. Compute separability transitions for all target pairs and log results

    Returns JSON summary of all rounds.
    """
    from huggingface_hub import login
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)

    # Symlink pre-baked models from docker image to HF_HOME volume path
    src_cache = "/root/.cache/huggingface/hub"
    dst_cache = "/hf_cache/hub"
    if os.path.exists(src_cache):
        os.makedirs(dst_cache, exist_ok=True)
        for item in os.listdir(src_cache):
            src_item = os.path.join(src_cache, item)
            dst_item = os.path.join(dst_cache, item)
            if not os.path.exists(dst_item):
                print(f"  🔗 Symlinking pre-baked model {item} to cache volume...")
                try:
                    os.symlink(src_item, dst_item)
                except Exception as e:
                    print(f"  ⚠️ Failed to symlink {item}: {e}")

    # ── Reproducibility ─────────────────────────────────────────
    set_all_seeds(seed)

    # ── Create timestamped run directory ────────────────────────
    run_id = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = create_run_dir(RESULTS_MOUNT, run_id)
    print(f"\n  Run directory: active_loop_runs/{run_id}/")

    # ── Load benchmark first ────────────────────────────────────
    volume = modal.Volume.from_name("benchmark-eval-results")
    benchmark, original_response_matrix = load_calibrated_benchmark(volume)
    irt = benchmark.calibrated_model

    # Identify the Saturation Band dynamically (Rasch MLE abilities > 0.5)
    saturation_models = [m for m, th in irt.thetas.items() if th > 0.5]
    print(f"\n🎯 [Saturation Band] Identified {len(saturation_models)} models with θ > 0.5:")
    print(f"   {sorted(saturation_models)}")

    # ── Save config ─────────────────────────────────────────────
    run_config = {
        "max_rounds": max_rounds,
        "questions_per_round": questions_per_round,
        "num_generation_steps": num_generation_steps,
        "generator_model": generator_model,
        "seed": seed,
        "model_subset": MODEL_SUBSET,
        "saturation_models": saturation_models,
        "timestamp": run_id,
    }
    save_json(os.path.join(run_dir, "config.json"), run_config)

    # ── 1. Calculate Initial Target Pairs inside the Saturation Band ──
    initial_target_separabilities = {}
    target_pairs = []
    for i in range(len(saturation_models)):
        for j in range(i + 1, len(saturation_models)):
            m_x, m_y = saturation_models[i], saturation_models[j]
            sep_xy = irt.compute_separability(m_x, m_y)
            if sep_xy["confidence"] < 0.95:
                target_pairs.append((m_x, m_y, sep_xy["confidence"]))
                initial_target_separabilities[f"{m_x}_vs_{m_y}"] = sep_xy

    # Compute initial Standard Errors for Saturation models
    initial_sat_ses = {m: irt.compute_ability_se(m) for m in saturation_models}
    avg_initial_sat_se = np.mean(list(initial_sat_ses.values()))

    # Track cumulative state across rounds
    all_generated_questions = []
    all_generation_responses = []      # full GenerationResponse objects per step (LLMTraces, exemplars, etc.)
    all_scoring_details = {}           # round_num -> {model_name -> {qid -> scoring breakdown}}
    irt_snapshots = []                 # IRT state (θs + difficulties) after each round refit
    cumulative_new_response_rows = {}  # model -> {item_id: 0/1}
    calibrated_gen_ids = set()          # generated item IDs already calibrated (become anchors)
    round_summaries = []

    print(f"\n{'='*80}")
    print(f"  ACTIVE MULTI-MIDPOINT GENERATION LOOP")
    print(f"  Initial Unresolved Target Pairs (confidence < 95%): {len(target_pairs)}")
    if target_pairs:
        print("  Closest initial target pairs:")
        for m_x, m_y, conf in sorted(target_pairs, key=lambda x: x[2])[:5]:
            print(f"    • ({m_x:>20s}, {m_y:>20s}) -> confidence = {conf:.2%}")
    print(f"  Initial Saturation Band Avg SE (θ logit scale):      {avg_initial_sat_se:.4f}")
    print(f"  Rounds: {max_rounds}, Midpoint Steps/round: {num_generation_steps}")
    print(f"  Generator: {generator_model}, Seed: {seed}")
    print(f"{'='*80}")

    # Baseline abilities mapping for scale drift MAE tracking (using saturation models)
    baseline_abilities = {m: float(irt.get_subject_ability(m)[0]) for m in saturation_models}

    loop_start = time.time()

    for round_num in range(1, max_rounds + 1):
        round_start = time.time()
        round_seed = seed + round_num * 100

        print(f"\n{'─'*80}")
        print(f"  ROUND {round_num}/{max_rounds}")
        print(f"{'─'*80}")

        # ── 1. Calculate Target Pairs inside the Saturation Band ──
        target_pairs = []
        for i in range(len(saturation_models)):
            for j in range(i + 1, len(saturation_models)):
                m_x, m_y = saturation_models[i], saturation_models[j]
                sep_xy = irt.compute_separability(m_x, m_y)
                if sep_xy["confidence"] < 0.95:
                    target_pairs.append((m_x, m_y, sep_xy["confidence"]))

        if not target_pairs:
            print("🎉 Success! All pairs in the saturation band are successfully separated!")
            break

        # ── 2. Multi-Step Question Generation Loops ──
        round_new_questions = []
        round_step_details = []

        for step_idx in range(1, num_generation_steps + 1):
            if not target_pairs: break
            state = np.random.RandomState(round_seed + step_idx)
            tgt_a, tgt_b, current_conf = target_pairs[state.choice(len(target_pairs))]
            step_target_diff = (irt.get_subject_ability(tgt_a)[0] + irt.get_subject_ability(tgt_b)[0]) / 2.0

            config = QuestionGeneratorConfig(
                prompter=ScaledExamplePrompterConfig(
                    generator_model=generator_model,
                    temperature=0.7,
                    thinking_budget=2048,
                    max_tokens=16384,
                    p=2.0,
                    num_examples=4,
                    num_questions=questions_per_round,
                    min_difficulty=step_target_diff - 1.5,
                    presentation=PresentationStyle.POSITIVE_UNBOUNDED,
                    double_ended=True,
                    min_discernability=1,
                    max_discernability=10,
                    seed=round_seed + step_idx,
                ),
                verifier=TrivialVerifierConfig(),
                target_selector=MidpointTargetSelectorConfig(model_a=tgt_a, model_b=tgt_b),
            )
            response = QuestionGenerator(config).generate(benchmark)
            step_new_questions = response.verified_questions if response else []
            round_new_questions.extend(step_new_questions)
            all_generation_responses.append(response)

            # Serialize full step details including LLM traces and exemplars
            step_detail = {
                "step": step_idx,
                "model_a": tgt_a,
                "model_b": tgt_b,
                "midpoint_diff": float(step_target_diff),
                "questions_generated": len(step_new_questions),
            }
            if response:
                # LLM Trace from prompter
                pr = response.prompter_response
                if pr and pr.trace:
                    step_detail["llm_trace"] = {
                        "system_prompt": pr.trace.system_prompt,
                        "user_prompt": pr.trace.user_prompt,
                        "thinking": pr.trace.thinking,
                        "raw_output": pr.trace.raw_output,
                    }
                # Exemplars used
                if pr and pr.exemplars:
                    step_detail["exemplars"] = [
                        {
                            "id": eq.id,
                            "question_text": eq.question_text,
                            "options": eq.options,
                            "correct_answer": eq.correct_answer,
                            "difficulty": float(eq.difficulty),
                            "discrimination": float(eq.discrimination) if eq.discrimination else None,
                            "calibrated": eq.calibrated,
                        }
                        for eq in pr.exemplars
                    ]
                # Verifier responses
                if response.verifier_responses:
                    step_detail["verifier_responses"] = [
                        {
                            "success": vr.success,
                            "question_id": vr.question.id if vr.question else None,
                            "trace": {
                                "system_prompt": vr.trace.system_prompt,
                                "user_prompt": vr.trace.user_prompt,
                                "thinking": vr.trace.thinking,
                                "raw_output": vr.trace.raw_output,
                            } if vr.trace else None,
                        }
                        for vr in response.verifier_responses
                    ]
                # Target profile
                if response.target_profile:
                    step_detail["target_profile"] = {
                        "target_difficulty": float(response.target_profile.target_difficulty),
                        "scale": float(response.target_profile.scale),
                    }
            round_step_details.append(step_detail)

        if not round_new_questions: continue
        all_generated_questions.extend(round_new_questions)

        # ── 3. Evaluation ─────────────
        round_response_matrix, round_scoring_details = evaluate_questions_local([{"id": q.id, "question_text": q.question_text, "options": q.options, "correct_answer": q.correct_answer} for q in round_new_questions], MODEL_SUBSET)
        all_scoring_details[round_num] = round_scoring_details
        
        # Filter out degenerate (0% or 100% correct) questions across evaluated models
        active_round_questions = []
        for q in round_new_questions:
            if q.id in round_response_matrix.columns:
                col_vals = round_response_matrix[q.id].dropna()
                if len(col_vals) > 0:
                    mean_val = col_vals.mean()
                    if mean_val == 0.0 or mean_val == 1.0:
                        print(f"  ⚠️ Question {q.id} rejected: degenerate score ({mean_val:.0%}) across models")
                        continue
            active_round_questions.append(q)

        round_new_questions = active_round_questions
        if not round_new_questions:
            print("  ⚠️ All questions generated in this round were degenerate. Skipping refit for this round.")
            continue

        for model_name in round_response_matrix.index:
            if model_name not in cumulative_new_response_rows: cumulative_new_response_rows[model_name] = {}
            for qid in [q.id for q in round_new_questions]:
                if qid in round_response_matrix.columns and not np.isnan(round_response_matrix.loc[model_name, qid]):
                    cumulative_new_response_rows[model_name][qid] = int(round_response_matrix.loc[model_name, qid])
        
        # ── 4. Anchored Rasch (1PL) Refit ──
        # Split cumulative responses: previously-calibrated generated items become
        # anchors (appended to original_response_matrix), only this round's new
        # items go through new_response_matrix for parameter estimation.
        this_round_ids = [q.id for q in round_new_questions]
        cum_resp_df = pd.DataFrame.from_dict(cumulative_new_response_rows, orient="index")

        # Build augmented anchor matrix: original ARC items + previously calibrated generated items
        aug_anchor_matrix = original_response_matrix.copy()
        if calibrated_gen_ids:
            prev_cal_ids = [qid for qid in calibrated_gen_ids if qid in cum_resp_df.columns]
            if prev_cal_ids:
                prev_cal_resp = cum_resp_df[prev_cal_ids].reindex(aug_anchor_matrix.index)
                aug_anchor_matrix = pd.concat([aug_anchor_matrix, prev_cal_resp], axis=1)

        # New response matrix: only this round's truly new items
        new_only_resp = cum_resp_df[this_round_ids].reindex(index=[m for m in MODEL_SUBSET if m in cum_resp_df.index])

        # Fit standard Rasch (1PL) anchored calibration (only b estimated for new, θ jointly re-fit)
        new_difficulties = irt.fit_anchored(
            new_response_matrix=new_only_resp,
            original_response_matrix=aug_anchor_matrix
        )

        # Calculate Difficulty Targeting MAE (average absolute error vs target midpoint)
        targeting_abs_errors = []
        question_details = []
        for q in round_new_questions:
            target_b = q.difficulty  # original target midpoint stored in prompter
            fitted_b = irt.difficulties[q.id]
            targeting_abs_errors.append(abs(fitted_b - target_b))
            
            question_details.append({
                "id": q.id,
                "question_text": q.question_text,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "target_difficulty": float(target_b),
                "calibrated_difficulty": float(fitted_b)
            })
        targeting_mae = float(np.mean(targeting_abs_errors)) if targeting_abs_errors else 0.0
        print(f"  🎯 Targeting Precision - Difficulty MAE (vs target):  {targeting_mae:.4f}")

        # ── 5. Promote calibrated questions into benchmark pool ──
        # These become exemplar candidates for future rounds and anchors for future refits.
        for q in round_new_questions:
            q.difficulty = irt.difficulties[q.id]
            # For generated questions, assume baseline 1.0 discrimination (1PL/Rasch assumption)
            q.discrimination = 1.0
            irt.discriminations[q.id] = 1.0
            q.calibrated = True
            benchmark.questions.append(q)
            calibrated_gen_ids.add(q.id)

        avg_disc = 1.0
        print(f"  📏 Average discernability of round {round_num} generated questions: {avg_disc:.3f}")

        # Compute post-refit separabilities for all initial target pairs
        target_pairs_separability = {}
        newly_resolved = 0
        unresolved_remaining = 0
        
        print(f"\n  📊 Updated Target Pairs Separability (Round {round_num}):")
        for pair_key in initial_target_separabilities.keys():
            m_x, m_y = pair_key.split("_vs_")
            sep_xy = irt.compute_separability(m_x, m_y)
            target_pairs_separability[pair_key] = sep_xy
            
            init_conf = initial_target_separabilities[pair_key]["confidence"]
            curr_conf = sep_xy["confidence"]
            
            if curr_conf >= 0.95 and init_conf < 0.95:
                newly_resolved += 1
                print(f"    🎉 RESOLVED: ({m_x:>20s}, {m_y:>20s}) separated with {curr_conf:.2%} confidence (was {init_conf:.2%})")
            else:
                unresolved_remaining += 1
                # For unresolved, print confidence changes if they are close
                if curr_conf < 0.95:
                    print(f"    • ({m_x:>20s}, {m_y:>20s}) -> curr_conf = {curr_conf:.2%} (was {init_conf:.2%}, z_shift = {sep_xy['z'] - initial_target_separabilities[pair_key]['z']:+.2f})")

        print(f"  📊 Target pairs separation update: resolved {newly_resolved} pairs, {unresolved_remaining} remain unresolved.")

        # Compute standard error progress of Saturation models
        curr_sat_ses = {m: irt.compute_ability_se(m) for m in saturation_models}
        avg_curr_sat_se = np.mean(list(curr_sat_ses.values()))
        print(f"  📉 Saturation Band Avg SE (θ scale): {avg_curr_sat_se:.4f} (Initial: {avg_initial_sat_se:.4f}, shift: {avg_curr_sat_se - avg_initial_sat_se:+.4f})")

        # Compute Mean Absolute Error (MAE) drift between baseline and updated abilities
        abs_errors = []
        for m in saturation_models:
            orig_th = baseline_abilities[m]
            curr_th = float(irt.get_subject_ability(m)[0])
            abs_errors.append(abs(curr_th - orig_th))
        round_mae = float(np.mean(abs_errors))
        print(f"  📈 Scale Stability - Ability MAE (vs baseline):     {round_mae:.4f}")

        # Snapshot IRT state after this round's refit
        irt_snapshot = {
            "round": round_num,
            "thetas": {m: float(irt.thetas[m]) for m in irt.thetas},
            "new_item_difficulties": {q.id: float(irt.difficulties[q.id]) for q in round_new_questions},
            "new_item_discriminations": {q.id: float(irt.discriminations.get(q.id, 1.0)) for q in round_new_questions},
        }
        irt_snapshots.append(irt_snapshot)

        summary = {
            "round": round_num,
            "total_questions": len(all_generated_questions),
            "calibrated_pool_size": len(benchmark.questions),
            "avg_discernability": float(avg_disc),
            "steps": round_step_details,
            "question_details": question_details,
            "scoring_details": round_scoring_details,  # full per-model logits/predictions
            "irt_snapshot": irt_snapshot,                # θ + b after refit
            "target_pairs_separability": target_pairs_separability,
            "avg_saturation_se": float(avg_curr_sat_se),
            "saturation_model_ses": {m: float(se) for m, se in curr_sat_ses.items()},
            "ability_mae": round_mae,
            "targeting_mae": targeting_mae,
            "round_time_s": time.time() - round_start,
        }
        round_summaries.append(summary)
        save_json(os.path.join(run_dir, f"round_{round_num}.json"), summary)
        results_vol.commit()

    # ── Final summary ───────────────────────────────────────────
    total_elapsed = time.time() - loop_start
    final_sat_ses = {m: irt.compute_ability_se(m) for m in saturation_models}
    avg_final_sat_se = np.mean(list(final_sat_ses.values()))

    # Compute final Mean Absolute Error (MAE) across all rounds
    final_abs_errors = []
    for m in saturation_models:
        orig_th = baseline_abilities[m]
        final_th = float(irt.get_subject_ability(m)[0])
        final_abs_errors.append(abs(final_th - orig_th))
    final_mae = float(np.mean(final_abs_errors))
    mae_progression = [r["ability_mae"] for r in round_summaries]
    targeting_mae_progression = [r["targeting_mae"] for r in round_summaries]
    avg_targeting_mae = float(np.mean(targeting_mae_progression)) if targeting_mae_progression else 0.0

    print(f"\n{'='*80}")
    print(f"  ACTIVE LOOP COMPLETE")
    print(f"{'='*80}")
    print(f"  Total rounds:     {len(round_summaries)}")
    print(f"  Total questions:  {len(all_generated_questions)}")
    print(f"  Total time:       {total_elapsed/60:.1f} minutes")
    print(f"")
    print(f"  Ability Scale Stability (MAE Drift vs Baseline):")
    print(f"    Final Avg MAE:   {final_mae:.4f}")
    print(f"    MAE Progression: " + " → ".join([f"{x:.4f}" for x in mae_progression]))
    print(f"")
    print(f"  Difficulty Targeting Precision (MAE vs Midpoints):")
    print(f"    Avg targeting MAE:     {avg_targeting_mae:.4f}")
    print(f"    Targeting Progression: " + " → ".join([f"{x:.4f}" for x in targeting_mae_progression]))
    print(f"")
    print(f"  Saturation Band Standard Error Reduction:")
    print(f"    Initial Avg SE:  {avg_initial_sat_se:.4f}")
    print(f"    Final Avg SE:    {avg_final_sat_se:.4f}")
    print(f"    Average Shift:   {avg_final_sat_se - avg_initial_sat_se:+.4f} ({ (avg_initial_sat_se - avg_final_sat_se)/avg_initial_sat_se:.1%} reduction)")
    print(f"")
    print(f"  θ trajectories (Anchored IRT Refit):")
    for model_name in sorted(saturation_models):
        print(f"    {model_name:>25s}: {irt.thetas[model_name]:.4f} (Fisher SE: {initial_sat_ses[model_name]:.4f} → {final_sat_ses[model_name]:.4f})")
    print(f"{'='*80}")

    # ── Unanchored Rasch (1PL) Sanity Check ─────────────────────────────────
    print(f"\n=== 🧪 UNANCHORED RASCH (1PL) SANITY CHECK ===")
    unanchored_results = {}
    try:
        from torch_measure.models import Rasch
        from scipy.special import expit

        # 1. Load original matrix and filter
        orig_df = original_response_matrix.copy()
        valid_subset = [m for m in MODEL_SUBSET if m in orig_df.index]
        orig_df = orig_df.loc[valid_subset]
        
        # Build cum_df by merging with cumulative_new_response_rows
        cum_df = orig_df.copy()
        cum_new_df = pd.DataFrame.from_dict(cumulative_new_response_rows, orient="index")
        if not cum_new_df.empty:
            cum_new_df = cum_new_df.reindex(index=valid_subset)
            cum_df = pd.concat([cum_df, cum_new_df], axis=1)

        # Drop zero-variance items independently to keep calibration stable
        orig_vars = orig_df.var(axis=0)
        orig_active = orig_df.drop(columns=orig_vars[orig_vars == 0].index)
        
        cum_vars = cum_df.var(axis=0)
        cum_active = cum_df.drop(columns=cum_vars[cum_vars == 0].index)

        print(f"  Calibrating initial Unanchored Rasch (items={orig_active.shape[1]})...")
        R_orig_np = orig_active.to_numpy()
        R_orig_t = torch.tensor(R_orig_np, dtype=torch.float32, device=device)
        R_orig_t = torch.nan_to_num(R_orig_t, nan=0.0)
        model_orig = Rasch(n_subjects=len(valid_subset), n_items=orig_active.shape[1], device=device)
        model_orig.fit(R_orig_t, method="mle", max_epochs=1000, lr=0.05, verbose=False)

        print(f"  Calibrating final Unanchored Rasch (items={cum_active.shape[1]})...")
        R_cum_np = cum_active.to_numpy()
        R_cum_t = torch.tensor(R_cum_np, dtype=torch.float32, device=device)
        R_cum_t = torch.nan_to_num(R_cum_t, nan=0.0)
        model_cum = Rasch(n_subjects=len(valid_subset), n_items=cum_active.shape[1], device=device)
        model_cum.fit(R_cum_t, method="mle", max_epochs=1000, lr=0.05, verbose=False)

        # Normalize and retrieve ability parameters
        with torch.no_grad():
            theta_orig = model_orig.ability.detach().cpu()
            b_orig = model_orig.difficulty.detach().cpu().numpy()
            
            theta_orig.sub_(theta_orig.mean())
            std_orig = theta_orig.std()
            if std_orig > 1e-5:
                theta_orig.div_(std_orig)
                b_orig /= std_orig.item()
            theta_orig_np = theta_orig.numpy()

            theta_cum = model_cum.ability.detach().cpu()
            b_cum = model_cum.difficulty.detach().cpu().numpy()
            
            theta_cum.sub_(theta_cum.mean())
            std_cum = theta_cum.std()
            if std_cum > 1e-5:
                theta_cum.div_(std_cum)
                b_cum /= std_cum.item()
            theta_cum_np = theta_cum.numpy()

        # Helper function to compute SE for a Rasch subject
        def get_model_se(theta_val, b_arr, mask_np):
            p_vals = expit(theta_val - b_arr)
            info = np.sum(p_vals * (1.0 - p_vals))
            return 1.0 / np.sqrt(info) if info > 0 else 1.0

        # Helper to compute separability stats
        def compute_sep(th_val_a, se_val_a, th_val_b, se_val_b):
            se_diff = np.sqrt(se_val_a**2 + se_val_b**2)
            z_val = abs(th_val_a - th_val_b) / se_diff
            p_val = 2 * (1 - norm.cdf(z_val))
            return {
                "theta_a": float(th_val_a),
                "theta_b": float(th_val_b),
                "se_a": float(se_val_a),
                "se_b": float(se_val_b),
                "se_diff": float(se_diff),
                "z": float(z_val),
                "p_value": float(p_val),
                "confidence": float(1.0 - p_val)
            }

        unanchored_pair_results = {}
        print(f"\n  📊 Unanchored Rasch (1PL) Separability Transitions for Target Pairs:")
        
        for pair_key in initial_target_separabilities.keys():
            m_x, m_y = pair_key.split("_vs_")
            idx_x = valid_subset.index(m_x)
            idx_y = valid_subset.index(m_y)
            
            # Initial unanchored SEs
            valid_orig_x = orig_active.iloc[idx_x].notna().to_numpy()
            se_orig_x = get_model_se(theta_orig_np[idx_x], b_orig[valid_orig_x], valid_orig_x)
            valid_orig_y = orig_active.iloc[idx_y].notna().to_numpy()
            se_orig_y = get_model_se(theta_orig_np[idx_y], b_orig[valid_orig_y], valid_orig_y)
            
            # Final unanchored SEs
            valid_cum_x = cum_active.iloc[idx_x].notna().to_numpy()
            se_cum_x = get_model_se(theta_cum_np[idx_x], b_cum[valid_cum_x], valid_cum_x)
            valid_cum_y = cum_active.iloc[idx_y].notna().to_numpy()
            se_cum_y = get_model_se(theta_cum_np[idx_y], b_cum[valid_cum_y], valid_cum_y)
            
            sep_init = compute_sep(theta_orig_np[idx_x], se_orig_x, theta_orig_np[idx_y], se_orig_y)
            sep_fin = compute_sep(theta_cum_np[idx_x], se_cum_x, theta_cum_np[idx_y], se_cum_y)
            
            unanchored_pair_results[pair_key] = {
                "initial": sep_init,
                "final": sep_fin
            }
            
            print(f"    • ({m_x:>20s}, {m_y:>20s}) -> Final Conf = {sep_fin['confidence']:.2%} (Initial = {sep_init['confidence']:.2%}, shift = {sep_fin['confidence'] - sep_init['confidence']:+.2%})")

        unanchored_results = {
            "success": True,
            "pairs": unanchored_pair_results,
        }

    except Exception as e:
        print(f"  ⚠️  Unanchored Rasch (1PL) Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        unanchored_results = {
            "success": False,
            "error": str(e)
        }
    print(f"{'='*80}\n")

    # Save final summary
    final_target_separabilities = {}
    newly_resolved_final = 0
    for pair_key in initial_target_separabilities.keys():
        m_x, m_y = pair_key.split("_vs_")
        sep_xy = irt.compute_separability(m_x, m_y)
        final_target_separabilities[pair_key] = sep_xy
        if sep_xy["confidence"] >= 0.95 and initial_target_separabilities[pair_key]["confidence"] < 0.95:
            newly_resolved_final += 1

    # ── Save comprehensive raw data artifacts ────────────────────
    raw_data_dir = os.path.join(run_dir, "raw_data")
    os.makedirs(raw_data_dir, exist_ok=True)

    # 1. Full cumulative response matrix (binary)
    cum_resp_df = pd.DataFrame.from_dict(cumulative_new_response_rows, orient="index")
    if not cum_resp_df.empty:
        cum_resp_df.to_csv(os.path.join(raw_data_dir, "cumulative_response_matrix.csv"))

    # 2. All generated questions with full details
    all_questions_data = []
    for q in all_generated_questions:
        all_questions_data.append({
            "id": q.id,
            "question_text": q.question_text,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "target_difficulty": float(q.difficulty),
            "calibrated_difficulty": float(irt.difficulties.get(q.id, q.difficulty)),
            "discrimination": float(q.discrimination) if q.discrimination else None,
            "calibrated": q.calibrated,
        })
    save_json(os.path.join(raw_data_dir, "all_generated_questions.json"), all_questions_data)

    # 3. IRT parameter snapshots (θ trajectory + item params per round)
    save_json(os.path.join(raw_data_dir, "irt_snapshots.json"), irt_snapshots)

    # 4. Final complete IRT state
    final_irt_state = {
        "thetas": {m: float(irt.thetas[m]) for m in irt.thetas},
        "difficulties": {qid: float(b) for qid, b in irt.difficulties.items()},
        "discriminations": {qid: float(a) for qid, a in irt.discriminations.items()},
        "valid_items_count": len(irt.valid_items),
    }
    save_json(os.path.join(raw_data_dir, "final_irt_state.json"), final_irt_state)

    # 5. Baseline abilities (initial θ values before loop)
    save_json(os.path.join(raw_data_dir, "baseline_abilities.json"), baseline_abilities)

    # 6. All scoring details (per-model logits/predictions per round)
    save_json(os.path.join(raw_data_dir, "all_scoring_details.json"), all_scoring_details)

    print(f"  💾 Raw data artifacts saved to {raw_data_dir}/")

    results_payload = {
        "run_id": run_id,
        "config": run_config,
        "initial_target_separabilities": initial_target_separabilities,
        "final_target_separabilities": final_target_separabilities,
        "initial_saturation_model_ses": {m: float(se) for m, se in initial_sat_ses.items()},
        "final_saturation_model_ses": {m: float(se) for m, se in final_sat_ses.items()},
        "avg_initial_saturation_se": float(avg_initial_sat_se),
        "avg_final_saturation_se": float(avg_final_sat_se),
        "final_ability_mae": final_mae,
        "ability_mae_trajectory": mae_progression,
        "final_targeting_mae": avg_targeting_mae,
        "targeting_mae_trajectory": targeting_mae_progression,
        "unanchored_rasch_1pl": unanchored_results,
        "rounds": round_summaries,
        "irt_snapshots": irt_snapshots,
        "total_time_s": total_elapsed,
        "model_subset": MODEL_SUBSET,
    }

    save_json(os.path.join(run_dir, "summary.json"), results_payload)
    results_vol.commit()
    print(f"\n  Results saved to volume: active_loop_runs/{run_id}/")

    return json.dumps(results_payload, indent=2)


# ────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT — kicks off the remote function
# ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    max_rounds: int = 10,
    questions_per_round: int = 5,
    num_generation_steps: int = 10,
    generator_model: str = "openai/gpt-5.5",
    seed: int = 42,
):
    """
    Launch the active question generation loop.
    All computation happens on a single remote A100 container.
    """
    print(f"\nLaunching dynamic active learning loop:")
    print(f"  Rounds: {max_rounds}, Steps/round: {num_generation_steps}, Questions/step: {questions_per_round}")
    print(f"  Generator: {generator_model}, Seed: {seed}\n")

    results_json = run_active_loop.remote(
        max_rounds=max_rounds,
        questions_per_round=questions_per_round,
        num_generation_steps=num_generation_steps,
        generator_model=generator_model,
        seed=seed,
    )

    results = json.loads(results_json)
    initial_unresolved = len(results.get("initial_target_separabilities", {}))
    final_unresolved = sum(1 for k, v in results.get("final_target_separabilities", {}).items() if v.get("confidence", 0.0) < 0.95)
    
    print(f"\n✅ Active loop complete!")
    print(f"   Run ID: {results.get('run_id', 'unknown')}")
    print(f"   Initial unresolved target pairs: {initial_unresolved}")
    print(f"   Final unresolved target pairs:   {final_unresolved}")
    print(f"   Successfully separated pairs:    {initial_unresolved - final_unresolved}")
    print(f"   Total questions generated: "
      f"{sum(r.get('total_questions', 0) for r in results['rounds'])}")
    print(f"   Total time: {results['total_time_s']/60:.1f} minutes")


# ────────────────────────────────────────────────────────────────
# CHEAP CPU PRE-DOWNLOADER
# ────────────────────────────────────────────────────────────────

@app.function(
    image=loop_image,
    volumes={
        "/hf_cache": hf_cache_vol
    },
    env={"HF_HOME": "/hf_cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=14400,  # 4 hours max
)
def pre_download_models_remote():
    """Pre-download all models in MODEL_SUBSET to the persistent cache volume on a cheap CPU worker."""
    import os
    import time
    from huggingface_hub import snapshot_download, login

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)

    # Clean up stale locks to prevent FileExistsError during snapshot downloads
    locks_dir = "/hf_cache/hub/.locks"
    if os.path.lexists(locks_dir):
        print(f"🧹 Cleaning up stale HF download locks directory/link: {locks_dir}")
        import shutil
        try:
            if os.path.islink(locks_dir):
                os.unlink(locks_dir)
            else:
                shutil.rmtree(locks_dir, ignore_errors=True)
        except Exception as del_err:
            print(f"  ⚠️  Failed to delete locks directory: {del_err}")

    print(f"\n=== PRE-DOWNLOADING {len(MODEL_SUBSET)} MODELS TO CACHE VOLUME ===")
    start_time = time.time()

    for i, model_short_name in enumerate(MODEL_SUBSET):
        reg = MODEL_REGISTRY.get(model_short_name)
        if not reg:
            print(f"  ❌ Unknown model: {model_short_name}")
            continue
        hf_id = reg["hf_id"]
        # HF cache dir for this model: /hf_cache/hub/models--{org}--{model}
        cache_dir_name = f"models--{hf_id.replace('/', '--')}"
        cache_path = os.path.join("/hf_cache", "hub", cache_dir_name)

        print(f"\n[{i+1}/{len(MODEL_SUBSET)}] Downloading: {model_short_name} ({hf_id})...")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=hf_id,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_files_only=False,
            )
            print(f"  ✅ Success in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  ⚠️  First attempt failed: {e}")
            # Check for corrupted cache dir (missing blobs/ or refs/, or broken symlinks)
            if os.path.lexists(cache_path):
                print(f"  🔧 Removing corrupted cache entry/link: {cache_path}")
                import shutil
                try:
                    if os.path.islink(cache_path):
                        os.unlink(cache_path)
                    else:
                        shutil.rmtree(cache_path, ignore_errors=True)
                except Exception as del_err:
                    print(f"  ⚠️  Failed to delete {cache_path}: {del_err}")
                
                # Retry
                try:
                    snapshot_download(
                        repo_id=hf_id,
                        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                        local_files_only=False,
                    )
                    print(f"  ✅ Retry success in {time.time() - t0:.1f}s")
                except Exception as e2:
                    print(f"  ❌ Retry also failed for {hf_id}: {e2}")
            else:
                print(f"  ❌ Failed to download {hf_id}: {e}")

    total_time = time.time() - start_time
    print(f"\n=== PRE-DOWNLOAD COMPLETE: {len(MODEL_SUBSET)} models in {total_time/60:.1f} minutes ===")
    
    print("Committing all changes to hf-cache-volume to persist model downloads...")
    hf_cache_vol.commit()


@app.local_entrypoint()
def pre_download():
    """Kicks off the CPU pre-downloader."""
    print("Launching cheap CPU pre-downloader on Modal...")
    pre_download_models_remote.remote()