# IRT-Driven Active Benchmarking for LLM Evaluation

**CS 321M Project — Stanford University**

An active learning system that uses **Item Response Theory (IRT)** to dynamically generate and calibrate new benchmark questions for large language models. The system identifies measurement gaps in existing benchmarks (e.g., ARC-Easy), generates targeted questions via LLM prompting, evaluates them across a panel of 30+ models on remote GPUs via [Modal](https://modal.com), and iteratively refits the psychometric model to improve discrimination in ability-saturated regions.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone & Install Dependencies](#1-clone--install-dependencies)
  - [2. Environment Variables & Secrets](#2-environment-variables--secrets)
  - [3. Modal Setup](#3-modal-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Quick Start: Run a Test Loop](#quick-start-run-a-test-loop)
  - [Full Active Loop](#full-active-loop)
  - [Generation Strategies](#generation-strategies)
  - [Run All Experiments](#run-all-experiments)
  - [Download Results from Modal](#download-results-from-modal)
  - [Produce Figures](#produce-figures)
- [End-to-End Example Walkthrough](#end-to-end-example-walkthrough)
- [Key CLI Arguments](#key-cli-arguments)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL MACHINE                                │
│  main.py (local entrypoint) ──► modal run ──► Remote A100 Container │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  MODAL CLOUD (per round)                             │
│                                                                     │
│  1. Load calibrated benchmark from Modal Volume                     │
│  2. Identify saturation band (models with θ > 0.5)                  │
│  3. Select target model pairs (confidence < 95%)                    │
│  4. Generate questions via OpenAI/Anthropic API                     │
│  5. Fan out parallel GPU evaluation (A10G / A100)                   │
│     ├── Model 1 (lm-evaluation-harness)                             │
│     ├── Model 2 ...                                                 │
│     └── Model N ...                                                 │
│  6. Anchored Rasch (1PL) refit                                      │
│  7. Log results to Modal Volume                                     │
│                                                                     │
│  Repeat for max_rounds                                              │
└─────────────────────────────────────────────────────────────────────┘
```

The system evaluates generated questions across **30+ open-weight models** (Pythia 70M → Qwen3.5-35B) using EleutherAI's `lm-evaluation-harness`, parallelized across Modal GPU containers. 

This will max out your model GPU containers and it's recommended to be careful scaling up. However, as huggingface models are pre-downloaded to modal volume on a CPU worker, startup overhead costs are minimal. Cost is actually comparable and slightly faster than a highly optimized custom GPU-packing algorithm which attempted to replicate lm_eval's methodology, an approach that was abandoned in favor of consistency. Unfortuntely the dominant cost is loading and unloading models to VRAM, which you can't get around without moving to paid hosted APIs.

Single round ablations from the paper should cost roughly $1.50 on modal and should take <10 minutes if nothing else is contending for modal containers. Full 10 round runs cost around $15 each and should take 1-2 hours without contention.  Total OpenAI costs should be <$25 to replicate all experiments. 

---

## Prerequisites

- **Python 3.11+**
- **Modal account** — [sign up at modal.com](https://modal.com) (free tier available)
- **OpenAI API key** — for question generation (GPT-4o / GPT-5.5)
- **Anthropic API key** *(optional)* — for Claude-based generation
- **Hugging Face account** — for gated model access (LLaMA, Gemma, etc.)

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/mattjhayes3/cs321m_project.git
cd cs321m_project

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `lm-eval` (lm-evaluation-harness) and heavy ML dependencies are primarily used inside Modal containers. For local-only development, you may skip `torch`, `transformers`, `accelerate`, and `lm-eval`.

### 2. Environment Variables & Secrets

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-...          # Required for question generation
ANTHROPIC_API_KEY=sk-ant-...   # Optional, for Claude-based generation
MODAL_TOKEN_ID=ak-...          # From `modal token new`
MODAL_TOKEN_SECRET=as-...      # From `modal token new`
```

You also need to configure **Modal Secrets** (these are injected into remote containers):

```bash
# Hugging Face token (required for gated models like LLaMA, Gemma)
modal secret create huggingface HF_TOKEN=hf_...

# OpenAI (for question generation inside the container)
modal secret create openai OPENAI_API_KEY=sk-...

# Anthropic (optional)
modal secret create anthropic ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Modal Setup

```bash
# Install the Modal CLI
pip install modal

# Authenticate (opens browser for login)
modal setup

# Verify your setup
modal run --help
```

Modal Volumes are created automatically on first run:
- `benchmark-eval-results` — stores calibrated IRT data and run outputs
- `hf-cache-volume` — caches Hugging Face model weights across runs

> **Note:** The pre-scraped, calibrated baseline data (ARC-Easy response matrices, Rasch difficulties, 2PL EM discriminations, and model abilities) is pre-packaged directly inside this repository under `baseline_data/`. On your very first run, the system will **automatically populate** your Modal Volume with these baselines. You do **not** need to run the initial calibration scripts in `Pre-analysis/` from scratch.

---

## Project Structure

```
Project/
├── main.py                  # Main entry point — orchestrates the active loop on Modal
├── evaluate_generated.py    # Parallel GPU evaluation backend (Modal workers)
├── modal_image.py           # Modal container image definition (dependencies)
├── interfaces.py            # Core data structures (Question, Benchmark, configs)
├── irt_model.py             # IRT model fitting (Rasch 1PL/2PL, MIRT, Fisher info)
├── prompter.py              # LLM prompt templates for all 4 generation strategies
├── question_generator.py    # Orchestrates question generation pipeline
├── target_selector.py       # Selects optimal difficulty targets via Fisher info
├── verifier.py              # Validates generated question format and quality
├── call_llm.py              # Unified OpenAI / Anthropic API interface
├── benchmark.py             # Loads calibrated benchmark from Modal Volume
├── utils.py                 # Logging, JSON I/O, scoring utilities
├── download_run.py          # Downloads run results from Modal Volume to local
├── produce_figures.py       # Generates all paper figures from saved results
├── run_all_experiments.py   # Batch launcher for all experiment configurations
├── project.py               # Project directory path constant
├── active_loop_runs/        # Downloaded run results (local)
├── results/                 # Aggregated analysis outputs
├── custom_tasks/            # YAML task definitions for lm-evaluation-harness
├── Pre-analysis/            # Initial calibration and exploration scripts
```

---

## Usage

### Quick Start: Run a Test Loop

The `--test-run` flag restricts evaluation to small models (< 7GB VRAM) for fast iteration:

```bash
modal run main.py --test-run --max-rounds 1 --num-generation-steps 2 --questions-per-target 2
```

This completes in under 5 minutes and is useful for verifying your setup.

### Full Active Loop

Run the default 10-round active loop with the `ScaledExample` strategy:

```bash
modal run main.py
```

Or with explicit arguments:

```bash
modal run main.py \
  --max-rounds 10 \
  --questions-per-target 5 \
  --num-generation-steps 10 \
  --prompter-type scaled_example \
  --generator-model "openai/gpt-5.5" \
  --seed 42
```

### Generation Strategies

The system supports four question generation strategies, each with different strengths:

| Strategy | Description | Flag |
|---|---|---|
| **ScaledExample** | Selects calibrated exemplars near the target difficulty and asks the LLM to generate questions at a scaled difficulty level | `--prompter-type scaled_example` |
| **AddOption** | Takes an existing question and adds a new distractor option to increase difficulty | `--prompter-type add_option` |
| **IncreaseDifficulty** | Rewrites an existing question to be harder by a target percentage | `--prompter-type increase_difficulty` |
| **NearbyExample** | Generates new questions similar in topic/style to exemplars near the target difficulty | `--prompter-type nearby_example` |

### Run All Experiments

To launch the full experiment suite (all strategies × ablation configurations):

```bash
python run_all_experiments.py
```

This sequentially runs:
1. `ScaledExample` (double-ended disabled)
2. `ScaledExample` (no discernability filter)
3. `IncreaseDifficulty` (25 steps × 2 questions)
4. `AddOption` (25 steps × 2 questions)

### Download Results from Modal

After runs complete, download the results from the Modal Volume:

```bash
# List all available runs
python download_run.py --list

# Download a specific run
python download_run.py 2026-05-27T11-56-00_add_option

# Download multiple runs
python download_run.py <run_id_1> <run_id_2>
```

Results are saved to `active_loop_runs/<run_id>/` with:
- `config.json` — full run configuration
- `round_N.json` — detailed output per round (questions, scores, IRT fits)
- `summary.json` — final summary with θ trajectories and separability results
- `raw_data/` — raw lm-eval JSONL outputs per model

### Produce Figures

Generate all paper figures from downloaded results:

```bash
python produce_figures.py
```

Figures are saved to `results/` and `plots_comparison/`.

---

## End-to-End Example Walkthrough

Here is a complete walkthrough from setup to results:

```bash
# ── Step 1: Environment setup ──
source .venv/bin/activate
modal setup

# ── Step 2: Verify Modal secrets are configured ──
modal secret list
# Should show: huggingface, openai, anthropic

# ── Step 3: Run a quick test to verify everything works ──
modal run main.py --test-run --max-rounds 1 --num-generation-steps 2

# ── Step 4: Run the AddOption strategy for 10 rounds ──
modal run main.py \
  --prompter-type add_option \
  --max-rounds 10 \
  --num-generation-steps 25 \
  --questions-per-target 2 \
  --seed 42

# ── Step 5: Download results ──
python download_run.py --list
python download_run.py <your-run-id>

# ── Step 6: Generate figures ──
python produce_figures.py
```

---

## Key CLI Arguments

All arguments are passed via `modal run main.py --<arg>`:

| Argument | Default | Description |
|---|---|---|
| `--max-rounds` | `10` | Number of active loop iterations |
| `--questions-per-target` | `5` | Questions generated per target (use 1 for AddOption, 2 for IncreaseDifficulty, and 5 for everything else) |
| `--num-generation-steps` | `10` | Target-pair generation steps per round (default: 10 targets × 5 questions = 50 questions/round) |
| `--generator-model` | `openai/gpt-5.5` | LLM used for question generation |
| `--prompter-type` | `scaled_example` | Generation strategy (`scaled_example`, `add_option`, `increase_difficulty`, `nearby_example`) |
| `--double-ended` | `True` | Enable double-ended difficulty targeting (ScaledExample only) |
| `--use-discernability` | `True` | Enable discernability filtering in prompts |
| `--delta-percent` | `0.25` | Target difficulty increase fraction (IncreaseDifficulty only) |
| `--selector-offset` | `0.0` | Difficulty offset for target selection (AddOption only) |
| `--seed` | `42` | Random seed for reproducibility |
| `--use-acc-norm` | `False` | Use length-normalized accuracy scoring |
| `--test-run` | `False` | Restrict to small models for fast testing |

---

## Expected Runtime & Computational Requirements

- **Quick Test Run** (`--test-run` with `--max-rounds 1`): Completes in **under 5 minutes**.
- **Full Active Learning Loop (10 Rounds)**:
  - **Wall-Clock Time**: Approximately **1 to 2 hours** if running without container contention on Modal.
  - **Per-Round Breakdown**:
    - OpenAI Generation step: ~10–30 seconds.
    - Parallel GPU Evaluation fan-out (lm-evaluation-harness on 30+ models): ~5–8 minutes.
    - Anchored IRT Refit: ~5 seconds.
  - **Modal GPU Billing**: Around **$1.50** per ablation/single-round test run, and **$15.00** for a full 10-round active learning loop.
  - **OpenAI API Billing**: Approximately **<$25.00** to replicate all experiments in the paper.

---

## Figure & Table Reproducibility Mapping

The plotting script `produce_figures.py` parses the JSON round snapshots and baseline calibrations, generating all primary and appendix figures/tables directly into `final_manuscript/figures/`:

| Paper Result | Script Function | Output Artifact |
|---|---|---|
| **Figure 1** (Ability Estimate Comparison) | `plot_ability_comparison` | `ability_comparison.pdf` |
| **Figure 2** (Ability Trajectory & SE Ribbons) | `plot_ability_trajectory` | `ability_trajectory.pdf` |
| **Figure 3** (Avg SE & Targeting MAE Progression) | `plot_se_and_mae_trajectory` | `se_mae_trajectory.pdf` |
| **Figure 4** (Pair Separability Bar & Gains) | `plot_separability_heatmap` | `separability.pdf` |
| **Figure 5** (Kendall Rank Correlation vs. Benchmarks) | `plot_kendall_correlation` | `kendall_correlation.pdf` |
| **Figure 6** (Generated Question Examples Table) | `save_question_examples` | `question_examples.md` |
| **Figure 7** (Difficulty Shift Histogram) | `plot_difficulty_shift` | `targeting_error_histogram.pdf` |
| **Figure 8** (Pairs Resolved Trajectory) | `plot_pairs_resolved` | `pairs_resolved.pdf` |
| **Figure 9** (Ability MAE Scale Stability Drift) | `plot_ability_mae_drift` | `ability_mae_drift.pdf` |
| **Figure 10** (Summary Dashboard Dashboard) | `plot_summary_dashboard` | `summary_dashboard.pdf` |
| **Figure 11** (Items Needed Under Original Dist) | `plot_items_needed` | `items_needed.pdf` |

