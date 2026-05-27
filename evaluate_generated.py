"""
Parallel Evaluation Scorer Backend for Active Question Generation Loop.
Hosts the Modal App definition, model registries, parallel lm_eval GPU containers,
and the evaluate_questions_local orchestrator.

Usage:
  Imported by main.py to offload evaluation workloads onto on-demand GPU containers.
"""

import os
import time
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import modal

# ────────────────────────────────────────────────────────────────
# MODAL INFRASTRUCTURE & IMAGES
# ────────────────────────────────────────────────────────────────

# Initialize the shared Modal App here to avoid circular imports
app = modal.App("active-question-generation")

results_vol = modal.Volume.from_name("benchmark-eval-results", create_if_missing=True)
USE_ACC_NORM = True
hf_cache_vol = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)
RESULTS_MOUNT = "/results"

# Image used for the GPU scoring containers (reuses loop dependencies)
from modal_image import loop_image

# ────────────────────────────────────────────────────────────────
# MODEL REGISTRY — maps short names to HF IDs
# Hardcodes -deduped for Pythias; contains exactly our 32 active models.
# ────────────────────────────────────────────────────────────────

# Models to to evaluate and their approximate VRAM usage for scheduling.
# Commented-out models either did not load correctly (getting near guessing performance) 
# or were nearly identical to another variant (e.g. chat vs it models), left here for posterity.
MODEL_REGISTRY = {
    "gemma1-2b":           {"hf_id": "google/gemma-2b",                    "vram_gb": 5.0},
    # "gemma1-2b-it":        {"hf_id": "google/gemma-2-2b-it",                 "vram_gb": 5.0},
    "gemma1-7b":           {"hf_id": "google/gemma-7b",                    "vram_gb": 15.0},
    # "gemma1-7b-it":        {"hf_id": "google/gemma-7b-it",                 "vram_gb": 15.0},
    "gemma2-2b":           {"hf_id": "google/gemma-2-2b",                  "vram_gb": 5.0},
    # "gemma2-2b-it":        {"hf_id": "google/gemma-2-2b-it",               "vram_gb": 5.0},
    "gemma2-9b-it":        {"hf_id": "google/gemma-2-9b-it",               "vram_gb": 19.0},
    "gemma3-1b-it":        {"hf_id": "google/gemma-3-1b-it",               "vram_gb": 3.0},
    # "gemma3-4b-it":        {"hf_id": "google/gemma-3-4b-it",               "vram_gb": 9.5},
    # "gemma3-12b-it":       {"hf_id": "google/gemma-3-12b-it",              "vram_gb": 26.0},
    "gpt2-small":          {"hf_id": "openai-community/gpt2",              "vram_gb": 1.0},
    "gpt2-large":          {"hf_id": "openai-community/gpt2-large",        "vram_gb": 2.0},
    "llama2-7b":           {"hf_id": "meta-llama/Llama-2-7b-hf",           "vram_gb": 15.0},
    "llama2-13b":          {"hf_id": "meta-llama/Llama-2-13b-hf",          "vram_gb": 28.0},
    # "llama2-7b-chat":      {"hf_id": "meta-llama/Llama-2-7b-chat-hf",      "vram_gb": 15.0},
    # "llama2-13b-chat":     {"hf_id": "meta-llama/Llama-2-13b-chat-hf",     "vram_gb": 28.0},
    "llama3-8b-inst":      {"hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",   "vram_gb": 17.0},
    "llama3.1-8b-inst":    {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",   "vram_gb": 17.0},
    "llama3.2-1b-inst":    {"hf_id": "meta-llama/Llama-3.2-1B-Instruct",   "vram_gb": 3.0},
    "llama3.2-3b-inst":    {"hf_id": "meta-llama/Llama-3.2-3B-Instruct",   "vram_gb": 7.0},
    "mistral-7b-inst":     {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "vram_gb": 15.0},
    "mistral-nemo":        {"hf_id": "mistralai/Mistral-Nemo-Instruct-2407", "vram_gb": 25.0},
    "phi3-mini":           {"hf_id": "microsoft/Phi-3-mini-4k-instruct",   "vram_gb": 8.0},
    "pythia-70m":          {"hf_id": "EleutherAI/pythia-70m-deduped",      "vram_gb": 0.5},
    "pythia-160m":         {"hf_id": "EleutherAI/pythia-160m-deduped",     "vram_gb": 0.8},
    "pythia-410m":         {"hf_id": "EleutherAI/pythia-410m-deduped",     "vram_gb": 1.2},
    "pythia-1b":           {"hf_id": "EleutherAI/pythia-1b-deduped",       "vram_gb": 2.5},
    "pythia-1.4b":         {"hf_id": "EleutherAI/pythia-1.4b-deduped",     "vram_gb": 3.5},
    "pythia-2.8b":         {"hf_id": "EleutherAI/pythia-2.8b-deduped",     "vram_gb": 7.0},
    "pythia-6.9b":         {"hf_id": "EleutherAI/pythia-6.9b-deduped",     "vram_gb": 15.0},
    "pythia-12b":          {"hf_id": "EleutherAI/pythia-12b-deduped",      "vram_gb": 26.0},
    "qwen2.5-3b-inst":     {"hf_id": "Qwen/Qwen2.5-3B-Instruct",           "vram_gb": 7.0},
    "qwen2.5-7b-inst":     {"hf_id": "Qwen/Qwen2.5-7B-Instruct",           "vram_gb": 15.0},
    "qwen2.5-14b-inst":    {"hf_id": "Qwen/Qwen2.5-14B-Instruct",          "vram_gb": 30.0},
    "qwen2.5-32b-inst":    {"hf_id": "Qwen/Qwen2.5-32B-Instruct",          "vram_gb": 66.0},
    "qwen2.5-coder-14b":   {"hf_id": "Qwen/Qwen2.5-Coder-14B-Instruct",   "vram_gb": 30.0},
    "qwen3-14b":           {"hf_id": "Qwen/Qwen3-14B",                     "vram_gb": 30.0, "trust_remote_code": True},
    # "qwen3-32b":           {"hf_id": "Qwen/Qwen3-32B",                     "vram_gb": 66.0, "trust_remote_code": True},
    "qwen3.5-27b":         {"hf_id": "Qwen/Qwen3.5-27B",                   "vram_gb": 56.0, "trust_remote_code": True},
    "qwen3.5-35b":         {"hf_id": "Qwen/Qwen3.5-35B-A3B",               "vram_gb": 72.0, "trust_remote_code": True},
}

MODEL_SUBSET = list(MODEL_REGISTRY.keys())

# ────────────────────────────────────────────────────────────────
# PARALLEL EVALUATION WORKERS
# ────────────────────────────────────────────────────────────────

@app.function(image=loop_image, volumes={RESULTS_MOUNT: results_vol})
def setup_lmeval_files(questions_list: List[dict], run_id: str, round_num: int):
    """Sets up round-specific JSONL and YAML task files on the persistent Volume."""
    import os
    import json
    
    run_dir = f"{RESULTS_MOUNT}/active_loop_runs/{run_id}"
    
    # Write JSONL specifically for this round
    jsonl_path = f"{RESULTS_MOUNT}/active_loop_runs/{run_id}/raw_data/round_{round_num}_questions.jsonl"
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w") as f:
        for q in questions_list:
            labels = [chr(65 + i) for i in range(len(q["options"]))]
            doc = {
                "question": q["question_text"],
                "choices": {
                    "text": q["options"],
                    "label": labels,
                },
                "answerKey": q["correct_answer"],
                "id": q["id"],
            }
            f.write(json.dumps(doc) + "\n")
            
    # Clean run_id for YAML task name compliance (letters, numbers, underscores only)
    clean_id = run_id.replace("-", "_").replace("T", "_").replace(":", "_")
    task_name = f"gen_arc_task_{clean_id}_r{round_num}"
    
    # Write round task YAML pointing to this round's JSONL
    yaml_path = f"{RESULTS_MOUNT}/{task_name}.yaml"
    yaml_content = f"""task: {task_name}
dataset_path: json
dataset_kwargs:
  data_files: {{"test": "{jsonl_path}"}}
test_split: test
output_type: multiple_choice
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{choices.label.index(answerKey)}}}}"
doc_to_choice: "{{{{choices.text}}}}"
metric_list:
  - metric: acc
  - metric: acc_norm
metadata:
  version: 1.0
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    results_vol.commit()
        
    print(f"Successfully setup unique YAML and JSONL files for {len(questions_list)} questions in Round {round_num}.")
    return len(questions_list)


def run_lmeval_for_model(m_name: str, hf_id: str, trust_remote_code: bool, run_id: str, round_num: int, use_acc_norm: bool = True) -> dict:
    """Helper inside the remote container that runs lm_eval CLI."""
    global USE_ACC_NORM
    USE_ACC_NORM = use_acc_norm
    import subprocess
    from huggingface_hub import login

    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"], add_to_git_credential=False)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    output_dir = f"/tmp/lmeval_out/{m_name}"
    os.makedirs(output_dir, exist_ok=True)

    remote_code = "True" if trust_remote_code else "False"
    model_args = f"pretrained={hf_id},dtype=float16,trust_remote_code={remote_code}"
    
    clean_id = run_id.replace("-", "_").replace("T", "_").replace(":", "_")
    task_name = f"gen_arc_task_{clean_id}_r{round_num}"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task_name,
        "--include_path", RESULTS_MOUNT,
        "--batch_size", "auto",
        "--log_samples",
        "--output_path", output_dir,
    ]

    print(f"[{m_name}] Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        print(f"[{m_name}] ⚠️ Error code {res.returncode}: {res.stderr}")
        return {"error": res.stderr or "Unknown error"}

    # Parse samples JSONL
    samples_file = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith(f"samples_{task_name}_") and f.endswith(".jsonl"):
                samples_file = os.path.join(root, f)
                break
    
    if not samples_file or not os.path.exists(samples_file):
        return {"error": "Could not find samples JSONL output file!"}

    results = {}
    score_details = {}
    
    with open(samples_file) as f:
        for line in f:
            if not line.strip():
                continue
            doc_res = json.loads(line)
            doc_id = doc_res.get("doc", {}).get("id")
            acc_norm = doc_res.get("acc_norm")
            acc = doc_res.get("acc")
            filtered = doc_res.get("filtered_resps", [])
            choices = doc_res.get("doc", {}).get("choices", {}).get("text", [])
            target = doc_res.get("target", "0")
            
            val_to_use = acc_norm if USE_ACC_NORM else acc
            if doc_id is not None and val_to_use is not None:
                results[doc_id] = int(val_to_use)
                
                # Reconstruct score details block matching custom scorer exactly!
                try:
                    pred_idx_norm = int(np.argmax([float(r[0])/len(opt) for r, opt in zip(filtered, choices)]))
                except Exception:
                    pred_idx_norm = int(target)
                pred_letter = chr(65 + pred_idx_norm)
                
                try:
                    pred_idx = int(np.argmax([float(r[0]) for r in filtered]))
                except Exception:
                    pred_idx = int(target)
                pred_letter_raw = chr(65 + pred_idx)
                
                expected = chr(65 + int(target))
                
                score_details[doc_id] = {
                    "scores": [float(r[0]) for r in filtered] if filtered else [0.0],
                    "scores_norm": [float(r[0])/len(opt) for r, opt in zip(filtered, choices)] if filtered else [0.0],
                    "pred_idx": pred_idx,
                    "pred_idx_norm": pred_idx_norm,
                    "pred_letter": pred_letter,
                    "pred_letter_raw": pred_letter_raw,
                    "expected": expected,
                    "correct_norm": int(acc_norm),
                    "correct_raw": int(acc),
                }

    # Also copy results to the run directory for archiving
    archive_dir = f"{RESULTS_MOUNT}/active_loop_runs/{run_id}/raw_data/lmeval_parallel/round_{round_num}/{m_name}"
    os.makedirs(archive_dir, exist_ok=True)
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            import shutil
            shutil.copy2(os.path.join(root, f), os.path.join(archive_dir, f))
            
    return {"results": results, "score_details": score_details, "error": None}


@app.function(
    image=loop_image,
    gpu="A10G",
    volumes={
        RESULTS_MOUNT: results_vol,
        "/hf_cache": hf_cache_vol
    },
    env={"HF_HOME": "/hf_cache"},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
    memory=16384,
)
def evaluate_single_model_a10g(m_name: str, hf_id: str, trust_remote_code: bool, run_id: str, round_num: int, use_acc_norm: bool = True) -> dict:
    return run_lmeval_for_model(m_name, hf_id, trust_remote_code, run_id, round_num, use_acc_norm)


@app.function(
    image=loop_image,
    gpu="A100-80GB",
    volumes={
        RESULTS_MOUNT: results_vol,
        "/hf_cache": hf_cache_vol
    },
    env={"HF_HOME": "/hf_cache"},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
    memory=32768,
)
def evaluate_single_model_a100(m_name: str, hf_id: str, trust_remote_code: bool, run_id: str, round_num: int, use_acc_norm: bool = True) -> dict:
    return run_lmeval_for_model(m_name, hf_id, trust_remote_code, run_id, round_num, use_acc_norm)


def evaluate_questions_local(
    questions: List[Dict[str, Any]],
    model_names: List[str],
    run_id: str,
    round_num: int,
    use_acc_norm: bool = True,
):
    """
    Evaluates generated questions in parallel on on-demand GPU containers on Modal.
    Replaces the legacy sequential Dynamic VRAM Packing scheduler with massive throughput parallel scoring.
    """
    setup_lmeval_files.remote(questions, run_id, round_num)
    
    print(f"\n Launching parallel evaluations for {len(model_names)} models on Modal...")
    # Sort model names in decreasing order of estimated VRAM to minimize tail latency under concurrent container limits
    sorted_model_names = sorted(
        model_names,
        key=lambda m: MODEL_REGISTRY.get(m, {}).get("vram_gb", 10.0),
        reverse=True
    )
    
    handles = []
    for m_name in sorted_model_names:
        cfg = MODEL_REGISTRY[m_name]
        hf_id = cfg["hf_id"]
        trust_remote_code = cfg.get("trust_remote_code", False)
        
        # GPU Routing Logic
        is_large = any(tag in hf_id.lower() or tag in m_name.lower() for tag in [
            "35b", "32b", "27b", "14b", "13b", "12b", "nemo"
        ])
        
        # Check if results already exist on volume and are complete!
        clean_id = run_id.replace("-", "_").replace("T", "_").replace(":", "_")
        task_name = f"gen_arc_task_{clean_id}_r{round_num}"
        
        archive_dir = f"{RESULTS_MOUNT}/active_loop_runs/{run_id}/raw_data/lmeval_parallel/round_{round_num}/{m_name}"
        samples_exist = False
        if os.path.exists(archive_dir):
            for f in os.listdir(archive_dir):
                if f.startswith(f"samples_{task_name}_") and f.endswith(".jsonl"):
                    try:
                        # Count valid JSON lines in the file
                        with open(os.path.join(archive_dir, f)) as file_obj:
                            lines = sum(1 for line in file_obj if line.strip())
                        if lines == len(questions):
                            samples_exist = True
                            break
                        else:
                            print(f"  ⚠️ {m_name:<22} cached file has incomplete count ({lines}/{len(questions)}), re-evaluating...")
                    except Exception:
                        pass
                    
        if samples_exist:
            # Re-parse the existing file instead of re-running!
            print(f"  ⏭️ {m_name:<22} [{hf_id}] skipping (cached results found)")
            # We will parse this asynchronously in a lightweight mock handle
            class MockHandle:
                def __init__(self, name, u_norm):
                    self.name = name
                    self.u_norm = u_norm
                def get(self):
                    return parse_existing_results(self.name, run_id, round_num, self.u_norm)
            handle = MockHandle(m_name, use_acc_norm)
        else:
            if is_large:
                print(f"  → {m_name:<22} [{hf_id}] spawning on A100-80GB...")
                handle = evaluate_single_model_a100.spawn(m_name, hf_id, trust_remote_code, run_id, round_num, use_acc_norm)
            else:
                print(f"  → {m_name:<22} [{hf_id}] spawning on A10G...")
                handle = evaluate_single_model_a10g.spawn(m_name, hf_id, trust_remote_code, run_id, round_num, use_acc_norm)
                
        handles.append((m_name, handle))
        
    print(f"⌛ Waiting for parallel evaluations to complete...")
    rows = {}
    all_score_details = {}
    errors = {}
    
    for m_name, handle in handles:
        try:
            res = handle.get()
            if res.get("error"):
                print(f"  ❌ {m_name:<22} FAILED: {res['error'][:100]}")
                errors[m_name] = res["error"]
            else:
                rows[m_name] = res["results"]
                all_score_details[m_name] = res["score_details"]
                print(f"  ✅ {m_name:<22} COMPLETED ({len(res['results'])} items)")
        except Exception as e:
            print(f"  ❌ {m_name:<22} EXCEPTION: {e}")
            errors[m_name] = str(e)
            
    print(f"\n🏁 Evaluations completed: {len(rows)} successful, {len(errors)} failed.")
    return pd.DataFrame.from_dict(rows, orient="index"), all_score_details


def parse_existing_results(m_name: str, run_id: str, round_num: int, use_acc_norm: bool = True) -> dict:
    """Helper to parse already completed results from volume instead of re-running."""
    global USE_ACC_NORM
    USE_ACC_NORM = use_acc_norm
    clean_id = run_id.replace("-", "_").replace("T", "_").replace(":", "_")
    task_name = f"gen_arc_task_{clean_id}_r{round_num}"
    
    archive_dir = f"{RESULTS_MOUNT}/active_loop_runs/{run_id}/raw_data/lmeval_parallel/round_{round_num}/{m_name}"
    samples_file = None
    for f in os.listdir(archive_dir):
        if f.startswith(f"samples_{task_name}_") and f.endswith(".jsonl"):
            samples_file = os.path.join(archive_dir, f)
            break
            
    results = {}
    score_details = {}
    with open(samples_file) as f:
        for line in f:
            if not line.strip(): continue
            doc_res = json.loads(line)
            doc_id = doc_res.get("doc", {}).get("id")
            acc_norm = doc_res.get("acc_norm")
            acc = doc_res.get("acc")
            filtered = doc_res.get("filtered_resps", [])
            choices = doc_res.get("doc", {}).get("choices", {}).get("text", [])
            target = doc_res.get("target", "0")
            
            val_to_use = acc_norm if USE_ACC_NORM else acc
            if doc_id is not None and val_to_use is not None:
                results[doc_id] = int(val_to_use)
                try:
                    pred_idx_norm = int(np.argmax([float(r[0])/len(opt) for r, opt in zip(filtered, choices)]))
                except Exception:
                    pred_idx_norm = int(target)
                pred_letter = chr(65 + pred_idx_norm)
                
                try:
                    pred_idx = int(np.argmax([float(r[0]) for r in filtered]))
                except Exception:
                    pred_idx = int(target)
                pred_letter_raw = chr(65 + pred_idx)
                
                expected = chr(65 + int(target))
                
                score_details[doc_id] = {
                    "scores": [float(r[0]) for r in filtered] if filtered else [0.0],
                    "scores_norm": [float(r[0])/len(opt) for r, opt in zip(filtered, choices)] if filtered else [0.0],
                    "pred_idx": pred_idx,
                    "pred_idx_norm": pred_idx_norm,
                    "pred_letter": pred_letter,
                    "pred_letter_raw": pred_letter_raw,
                    "expected": expected,
                    "correct_norm": int(acc_norm),
                    "correct_raw": int(acc),
                }
    return {"results": results, "score_details": score_details, "error": None}
