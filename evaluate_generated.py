"""
Evaluate Generated Questions on Modal
=======================================
Lightweight evaluation script that replicates the exact ARC-Easy format
(multiple-choice log-likelihood scoring) without the lm_eval overhead.

For each generated question and each model in the subset, we:
1. Format: "Question: <text>\nAnswer: <option_text>"  for each option
2. Compute the conditional log-likelihood of each option continuation
3. Pick the option with the highest log-likelihood as the model's answer
4. Score correct=1 if the chosen option matches the correct answer

This matches lm_eval's "acc" metric for arc_easy exactly.

Architecture: All models are evaluated SEQUENTIALLY on a single A100-80GB
container to avoid paying N separate cold-start penalties. With only a handful
of questions per model, the actual inference is trivial (~2s per model) and
the dominant cost is model loading — which we'd pay either way.

Usage:
    # From main.py pipeline (programmatic):
    from evaluate_generated import evaluate_questions_on_models, build_response_matrix

    # Standalone test:
    modal run evaluate_generated.py
"""

import json
import os
import time
from typing import List, Dict, Any

import modal
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

# ══════════════════════════════════════════════════════════════════
#  MODAL INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════

app = modal.App("eval-generated-questions")

results_vol = modal.Volume.from_name("benchmark-eval-results", create_if_missing=True)
RESULTS_MOUNT = "/results"

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "huggingface_hub>=1.5.0",
        "pandas",
        "numpy",
        "scipy",
        "einops",
    )
)

# ══════════════════════════════════════════════════════════════════
#  MODEL REGISTRY — maps short names to HF IDs
#  (must match the names used in the pre-analysis evaluation)
# ══════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "gemma1-2b":           {"hf_id": "google/gemma-2b",                    "vram_gb": 5.0},
    "gemma1-2b-it":        {"hf_id": "google/gemma-2-2b-it",                 "vram_gb": 5.0},
    "gemma1-7b":           {"hf_id": "google/gemma-7b",                    "vram_gb": 15.0},
    "gemma1-7b-it":        {"hf_id": "google/gemma-7b-it",                 "vram_gb": 15.0},
    "gemma2-2b":           {"hf_id": "google/gemma-2-2b",                  "vram_gb": 5.0},
    "gemma2-2b-it":        {"hf_id": "google/gemma-2-2b-it",               "vram_gb": 5.0},
    "gemma2-9b-it":        {"hf_id": "google/gemma-2-9b-it",               "vram_gb": 19.0},
    "gemma3-1b-it":        {"hf_id": "google/gemma-3-1b-it",               "vram_gb": 3.0},
    "gemma3-4b-it":        {"hf_id": "google/gemma-3-4b-it",               "vram_gb": 9.5},
    "gemma3-12b-it":       {"hf_id": "google/gemma-3-12b-it",              "vram_gb": 26.0},
    "gpt2-small":          {"hf_id": "openai-community/gpt2",              "vram_gb": 1.0},
    "gpt2-large":          {"hf_id": "openai-community/gpt2-large",        "vram_gb": 2.0},
    "llama2-7b":           {"hf_id": "meta-llama/Llama-2-7b-hf",           "vram_gb": 15.0},
    "llama2-13b":          {"hf_id": "meta-llama/Llama-2-13b-hf",          "vram_gb": 28.0},
    "llama2-7b-chat":      {"hf_id": "meta-llama/Llama-2-7b-chat-hf",      "vram_gb": 15.0},
    "llama2-13b-chat":     {"hf_id": "meta-llama/Llama-2-13b-chat-hf",     "vram_gb": 28.0},
    "llama3-8b-inst":      {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",   "vram_gb": 17.0},
    "llama3.1-8b-inst":    {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",   "vram_gb": 17.0},
    "llama3.2-1b-inst":    {"hf_id": "meta-llama/Llama-3.2-1B-Instruct",   "vram_gb": 3.0},
    "llama3.2-3b-inst":    {"hf_id": "meta-llama/Llama-3.2-3B-Instruct",   "vram_gb": 7.0},
    "mistral-7b-inst":     {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "vram_gb": 15.0},
    "mistral-nemo":        {"hf_id": "mistralai/Mistral-Nemo-Instruct-2407", "vram_gb": 25.0},
    "phi3-mini":           {"hf_id": "microsoft/Phi-3-mini-4k-instruct",   "vram_gb": 8.0},
    "pythia-70m":          {"hf_id": "EleutherAI/pythia-70m",              "vram_gb": 0.5},
    "pythia-160m":         {"hf_id": "EleutherAI/pythia-160m",             "vram_gb": 0.8},
    "pythia-410m":         {"hf_id": "EleutherAI/pythia-410m",             "vram_gb": 1.2},
    "pythia-1b":           {"hf_id": "EleutherAI/pythia-1b",               "vram_gb": 2.5},
    "pythia-1.4b":         {"hf_id": "EleutherAI/pythia-1.4b",             "vram_gb": 3.5},
    "pythia-2.8b":         {"hf_id": "EleutherAI/pythia-2.8b",             "vram_gb": 7.0},
    "pythia-6.9b":         {"hf_id": "EleutherAI/pythia-6.9b",             "vram_gb": 15.0},
    "pythia-12b":          {"hf_id": "EleutherAI/pythia-12b",              "vram_gb": 26.0},
    "qwen2.5-3b-inst":     {"hf_id": "Qwen/Qwen2.5-3B-Instruct",           "vram_gb": 7.0},
    "qwen2.5-7b-inst":     {"hf_id": "Qwen/Qwen2.5-7B-Instruct",           "vram_gb": 15.0},
    "qwen2.5-14b-inst":    {"hf_id": "Qwen/Qwen2.5-14B-Instruct",          "vram_gb": 30.0},
    "qwen2.5-32b-inst":    {"hf_id": "Qwen/Qwen2.5-32B-Instruct",          "vram_gb": 66.0},
    "qwen2.5-coder-14b":   {"hf_id": "Qwen/Qwen2.5-Coder-14B-Instruct",   "vram_gb": 30.0},
    "qwen3-14b":           {"hf_id": "Qwen/Qwen3-14B",                     "vram_gb": 30.0, "trust_remote_code": True},
    "qwen3-32b":           {"hf_id": "Qwen/Qwen3-32B",                     "vram_gb": 66.0, "trust_remote_code": True},
    "qwen3.5-27b":         {"hf_id": "Qwen/Qwen3.5-27B",                   "vram_gb": 56.0, "trust_remote_code": True},
    "qwen3.5-35b":         {"hf_id": "Qwen/Qwen3.5-35B-A3B",               "vram_gb": 72.0, "trust_remote_code": True},
}

# ══════════════════════════════════════════════════════════════════
#  CORE EVALUATION LOGIC
# ══════════════════════════════════════════════════════════════════

def _format_arc_easy_prompt(question_text: str) -> str:
    """
    Format a question context exactly as lm_eval does for arc_easy.

    lm_eval's arc_easy uses the template:
        "Question: <question>\nAnswer:"

    The model scores the log-likelihood of " <option_text>" as continuation.
    """
    return f"Question: {question_text}\nAnswer:"


def _score_options(model, tokenizer, question_text: str, options: List[str], device: str) -> dict:
    """
    Score each option by conditional log-likelihood and return scoring results.

    This replicates lm_eval's multiple-choice scoring for both acc and acc_norm:
    - context = "Question: <text>\nAnswer:"
    - continuation = " <option_text>"
    - acc:      argmax sum(log-probs of continuation tokens)
    - acc_norm: argmax sum(log-probs) / num_continuation_tokens

    Returns:
        dict with keys:
            pred_idx:      int — best option index using raw log-likelihood (acc)
            pred_idx_norm: int — best option index using length-normalized ll (acc_norm)
            scores:        list[float] — raw log-likelihood per option
            scores_norm:   list[float] — length-normalized log-likelihood per option
    """
    import torch

    # Resolve BPE tokenizer settings: match lm_eval's add_special_tokens behavior dynamically
    add_special = getattr(tokenizer, "add_bos_token", True)

    context = _format_arc_easy_prompt(question_text)
    context_ids = tokenizer.encode(context, add_special_tokens=add_special, return_tensors="pt").to(device)
    context_len = context_ids.shape[1]

    scores = []
    scores_norm = []

    for idx, option in enumerate(options):
        # Full sequence: context + " option_text"
        full_text = context + " " + option
        full_ids = tokenizer.encode(full_text, add_special_tokens=add_special, return_tensors="pt").to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda"):
                    outputs = model(full_ids)
            else:
                outputs = model(full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Log-probabilities of continuation tokens only
        # Shift: logits[t] predicts token[t+1]
        log_probs = torch.log_softmax(logits[0], dim=-1)

        # Sum log-probs for continuation tokens (from context_len onward)
        score = 0.0
        n_cont_tokens = full_ids.shape[1] - context_len
        for t in range(context_len - 1, full_ids.shape[1] - 1):
            next_token = full_ids[0, t + 1]
            score += log_probs[t, next_token].item()

        scores.append(score)
        # Length-normalized score (acc_norm): divide by number of continuation tokens
        scores_norm.append(score / n_cont_tokens if n_cont_tokens > 0 else score)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_idx_norm = max(range(len(scores_norm)), key=lambda i: scores_norm[i])

    return {
        "pred_idx": best_idx,
        "pred_idx_norm": best_idx_norm,
        "scores": scores,
        "scores_norm": scores_norm,
    }


def _evaluate_single_model(model_short_name: str, questions: List[dict]) -> dict:
    """
    Load a single model, score all questions, return results dict.
    Called inside the remote Modal container.
    """
    import torch
    import gc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reg = MODEL_REGISTRY.get(model_short_name)
    if not reg:
        return {"model": model_short_name, "results": [], "accuracy": 0.0,
                "error": f"Unknown model: {model_short_name}"}

    hf_id = reg["hf_id"]
    trust_remote_code = reg.get("trust_remote_code", False)

    print(f"\n  ── {model_short_name} ({hf_id}) ──")
    load_start = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer_kwargs = {}
        if "mistral" in hf_id.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code, **tokenizer_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        load_elapsed = time.time() - load_start
        print(f"    Loaded in {load_elapsed:.1f}s")

        results = []
        n_correct = 0
        n_correct_norm = 0
        infer_start = time.time()

        for q in questions:
            score_result = _score_options(
                model, tokenizer, q["question_text"], q["options"], device
            )
            # Use acc_norm (length-normalized) by default
            predicted_idx = score_result["pred_idx_norm"]
            predicted_letter = chr(65 + predicted_idx)
            expected = q["correct_answer"].strip().upper()
            correct = int(predicted_letter == expected)
            n_correct_norm += correct

            # Also track raw acc
            predicted_letter_raw = chr(65 + score_result["pred_idx"])
            correct_raw = int(predicted_letter_raw == expected)
            n_correct += correct_raw

            results.append({
                "item_id": q["id"],
                "correct": correct,
                "correct_raw": correct_raw,
                "predicted": predicted_letter,
                "predicted_raw": predicted_letter_raw,
                "expected": expected,
            })

        infer_elapsed = time.time() - infer_start
        accuracy_norm = n_correct_norm / len(questions) if questions else 0.0
        accuracy_raw = n_correct / len(questions) if questions else 0.0
        print(f"    {n_correct_norm}/{len(questions)} correct ({accuracy_norm:.0%} acc_norm, {accuracy_raw:.0%} acc) — "
              f"inference {infer_elapsed:.1f}s, load {load_elapsed:.1f}s")

        # Free GPU memory before loading the next model
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "model": model_short_name,
            "results": results,
            "accuracy": accuracy_norm,
            "accuracy_raw": accuracy_raw,
            "error": None,
        }

    except Exception as e:
        print(f"    ❌ Error: {e}")
        # Clean up on error too
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "model": model_short_name,
            "results": [],
            "accuracy": 0.0,
            "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════
#  MODAL FUNCTION: Single container evaluates ALL models sequentially
# ══════════════════════════════════════════════════════════════════

@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours — enough for ~40 models sequentially
    volumes={RESULTS_MOUNT: results_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=65536,
    env={"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
)
def evaluate_all_models_sequential(
    questions_json: str,
    model_names_json: str,
) -> str:
    """
    Evaluate ALL models sequentially on a single A100-80GB container.

    This avoids N separate cold-start penalties. With only a few questions
    per model, inference is trivial (~2s) and model loading dominates.
    Sequential on one machine means we pay for one cold start instead of N.

    Models are loaded one at a time in fp16 with device_map="auto", scored,
    then freed from GPU memory before loading the next.
    """
    from huggingface_hub import login

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)

    questions = json.loads(questions_json)
    model_names = json.loads(model_names_json)

    print(f"\n{'='*70}")
    print(f"  Sequential Evaluation: {len(questions)} questions × {len(model_names)} models")
    print(f"  Container: A100-80GB (single instance)")
    print(f"{'='*70}")

    all_results = {}
    total_start = time.time()

    for i, model_name in enumerate(model_names):
        print(f"\n  [{i+1}/{len(model_names)}]", end="")
        result = _evaluate_single_model(model_name, questions)
        all_results[model_name] = result

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  Done — {len(model_names)} models in {total_elapsed/60:.1f} minutes")
    n_success = sum(1 for r in all_results.values() if not r.get("error"))
    n_failed = len(model_names) - n_success
    print(f"  Success: {n_success}  |  Failed: {n_failed}")
    print(f"{'='*70}")

    return json.dumps(all_results)


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API: Called from main.py
# ══════════════════════════════════════════════════════════════════

def evaluate_questions_on_models(
    questions: List[Dict[str, Any]],
    model_names: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate a list of generated questions across all specified models.

    Dispatches everything to a single A100-80GB container that loads
    models sequentially. Avoids N separate cold-start penalties.

    Args:
        questions: List of question dicts with id, question_text, options, correct_answer
        model_names: List of model short names to evaluate on

    Returns:
        Dict mapping model_name -> list of per-question result dicts
        Each result dict has: item_id, correct (0/1), predicted, expected
    """
    questions_json = json.dumps(questions)
    model_names_json = json.dumps(model_names)

    print(f"\n🚀 Evaluating {len(questions)} questions on {len(model_names)} models "
          f"(single A100, sequential)...")
    start = time.time()

    # Single remote call — one container, all models
    raw_results_json = evaluate_all_models_sequential.remote(
        questions_json, model_names_json
    )
    raw_results = json.loads(raw_results_json)

    elapsed = time.time() - start
    print(f"\n  Done — {len(model_names)} models in {elapsed/60:.1f} minutes")

    # Extract per-model result lists
    all_results = {}
    for name in model_names:
        model_result = raw_results.get(name, {})
        if model_result.get("error"):
            print(f"  ❌ {name}: {model_result['error'][:100]}")
            all_results[name] = []
        else:
            acc = model_result.get("accuracy", 0)
            print(f"  ✅ {name}: acc={acc:.1%}")
            all_results[name] = model_result.get("results", [])

    return all_results


def build_response_matrix(
    eval_results: Dict[str, List[Dict[str, Any]]],
    question_ids: List[str],
) -> "pd.DataFrame":
    """
    Build a binary response matrix (models × items) from evaluation results.

    Returns:
        pd.DataFrame with model names as index, question IDs as columns,
        values are 0/1 (correct/incorrect).
    """
    import pandas as pd

    rows = {}
    for model_name, results in eval_results.items():
        result_map = {r["item_id"]: r["correct"] for r in results}
        rows[model_name] = {qid: result_map.get(qid, float("nan")) for qid in question_ids}

    return pd.DataFrame.from_dict(rows, orient="index")


# ══════════════════════════════════════════════════════════════════
#  STANDALONE TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main():
    """Standalone test: evaluate a few dummy questions on a small model."""
    test_questions = [
        {
            "id": "test_q1",
            "question_text": "What is the chemical symbol for water?",
            "options": ["CO2", "NaCl", "O2", "H2O"],
            "correct_answer": "D",
        },
        {
            "id": "test_q2",
            "question_text": "What planet is closest to the Sun?",
            "options": ["Venus", "Mercury", "Earth", "Mars"],
            "correct_answer": "B",
        },
    ]
    test_models = ["pythia-70m", "gpt2-small"]

    print("\n=== FORMATTED PROMPTS SEEN BY THE MODELS ===")
    for q in test_questions:
        context = f"Question: {q['question_text']}\nAnswer:"
        print(f"\n• [Question ID: {q['id']}]")
        print("  --- Context ---")
        print(f"  {context!r}")
        print("  --- Continuation Options (to evaluate log-likelihood) ---")
        for opt in q['options']:
            print(f"    - {f' {opt}'!r}")
    print("============================================\n")

    results = evaluate_questions_on_models(test_questions, test_models)

    import pandas as pd
    response_df = build_response_matrix(results, [q["id"] for q in test_questions])
    print("\nResponse Matrix:")
    print(response_df)
