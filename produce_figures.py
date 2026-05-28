#!/usr/bin/env python3
"""
produce_figures.py — Generate all manuscript figures from active-learning run data.

Usage:
    python produce_figures.py --run-dir active_loop_runs/2026-05-27T11-56-00_add_option
    python produce_figures.py --run-dir active_loop_runs/2026-05-27T11-56-00_add_option \
                              --compare-dir active_loop_runs/2026-05-27T12-40-29_add_option

Generates:
    1. Ability estimate comparison (baseline vs final, with error bars)
    2. Ability trajectory plots (θ over rounds, with SE ribbons)
    3. Standard error reduction trajectory
    4. Targeting precision (MAE trajectory)
    5. Pair separability heatmap (before/after)
    6. Kendall rank correlation with external benchmarks (arc_challenge, gpqa)
    7. Generated question examples (formatted table)
    8. Difficulty shift histogram (AddOption systematic shift analysis)

All figures saved to: final_manuscript/figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Optional imports (graceful fallback)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from scipy.stats import kendalltau, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ════════════════════════════════════════════════════════════════════
# Style Configuration
# ════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "serif",
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (8, 5),
})

# Color palette
COLORS = {
    "baseline": "#4C72B0",
    "final": "#DD8452",
    "se_ribbon": "#55A868",
    "targeting": "#C44E52",
    "resolved": "#8172B3",
    "highlight": "#CCB974",
    "offset_run": "#4C72B0",
    "no_offset_run": "#DD8452",
}

MODEL_DISPLAY_ORDER = [
    "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b",
    "pythia-2.8b", "pythia-6.9b", "pythia-12b",
    "gpt2-small", "gpt2-large",
    "gemma1-2b", "gemma1-7b", "gemma2-2b", "gemma2-9b-it",
    "gemma3-1b-it", "gemma3-12b-it",
    "llama2-7b", "llama2-13b",
    "llama3-8b-inst", "llama3.1-8b-inst",
    "llama3.2-1b-inst", "llama3.2-3b-inst",
    "mistral-7b-inst", "mistral-nemo",
    "phi3-mini",
    "qwen2.5-3b-inst", "qwen2.5-7b-inst", "qwen2.5-14b-inst",
    "qwen2.5-32b-inst", "qwen2.5-coder-14b",
    "qwen3-14b", "qwen3-32b",
    "qwen3.5-27b", "qwen3.5-35b",
]


# ════════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════════

def load_run(run_dir: str) -> dict:
    """Load all data from an active-learning run directory."""
    run_dir = Path(run_dir)
    
    data = {"run_dir": str(run_dir)}
    
    # Load summary
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data["summary"] = json.load(f)
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data["config"] = json.load(f)
    
    # Load per-round data
    data["rounds"] = []
    for i in range(1, 100):
        round_path = run_dir / f"round_{i}.json"
        if round_path.exists():
            with open(round_path) as f:
                data["rounds"].append(json.load(f))
        else:
            break
    
    # Load raw data artifacts if available
    raw_dir = run_dir / "raw_data"
    if raw_dir.exists():
        for fname in ["irt_snapshots.json", "baseline_abilities.json",
                       "all_generated_questions.json", "final_irt_state.json"]:
            fpath = raw_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    key = fname.replace(".json", "")
                    data[key] = json.load(f)
    
    n_rounds = len(data["rounds"])
    print(f"  Loaded run from {run_dir}: {n_rounds} rounds")
    
    return data


def load_baseline_thetas(project_dir: str) -> dict:
    """Load baseline (pre-loop) Rasch theta estimates from the calibration CSV."""
    csv_path = os.path.join(project_dir, "results", "arc_easy_eval", "plots",
                            "theta_comparison_comprehensive.csv")
    if not os.path.exists(csv_path):
        # Fall back to simpler CSV
        csv_path = os.path.join(project_dir, "results", "arc_easy_eval", "plots",
                                "theta_comparison.csv")
    
    if os.path.exists(csv_path) and HAS_PANDAS:
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            result[row["Model"]] = {
                "theta": row["Rasch_Theta"],
                "se": row["Rasch_SE"],
            }
        print(f"  Loaded {len(result)} baseline thetas from {csv_path}")
        return result
    
    print("  Warning: No baseline theta CSV found")
    return {}


# ════════════════════════════════════════════════════════════════════
# Figure 1: Ability Estimate Comparison (Baseline vs Final)
# ════════════════════════════════════════════════════════════════════

def plot_ability_comparison(run_data: dict, baseline_thetas: dict, 
                            out_dir: str, label: str = ""):
    """Bar chart comparing baseline and final ability estimates with error bars."""
    summary = run_data.get("summary", {})
    
    # Extract baseline and final abilities
    irt_snapshots = run_data.get("irt_snapshots", [])
    if not irt_snapshots and "rounds" in run_data:
        irt_snapshots = [r.get("irt_snapshot", {}) for r in run_data["rounds"]]
    
    if not irt_snapshots:
        print("  Skipping ability comparison: no IRT snapshots")
        return
    
    # Use the last snapshot for final thetas
    final_thetas = irt_snapshots[-1].get("thetas", {})
    
    # Get baseline from summary or from loaded baseline data
    base_abilities = summary.get("config", {}).get("baseline_abilities", {})
    if not base_abilities and "baseline_abilities" in run_data:
        base_abilities = run_data["baseline_abilities"]
    
    # Get initial/final SEs from summary
    initial_ses = summary.get("initial_saturation_model_ses", {})
    final_ses = summary.get("final_saturation_model_ses", {})
    
    # Determine which models to show (saturation band only)
    models = sorted(final_thetas.keys(), 
                    key=lambda m: MODEL_DISPLAY_ORDER.index(m) 
                    if m in MODEL_DISPLAY_ORDER else 999)
    
    # Filter to models we have both baseline and final for
    valid = [m for m in models if m in base_abilities and m in final_thetas]
    if not valid:
        # Try building from baseline_thetas
        valid = [m for m in models if m in baseline_thetas and m in final_thetas]
        base_abilities = {m: baseline_thetas[m]["theta"] for m in valid}
    
    if not valid:
        print("  Skipping ability comparison: no matched models")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(valid))
    width = 0.35
    
    baseline_vals = [float(base_abilities.get(m, 0)) for m in valid]
    final_vals = [float(final_thetas.get(m, 0)) for m in valid]
    baseline_errs = [float(initial_ses.get(m, baseline_thetas.get(m, {}).get("se", 0.07))) 
                     for m in valid]
    final_errs = [float(final_ses.get(m, 0.07)) for m in valid]
    
    ax.bar(x - width/2, baseline_vals, width, label="Baseline θ",
           color=COLORS["baseline"], alpha=0.8, yerr=baseline_errs, 
           capsize=3, error_kw={"linewidth": 0.8})
    ax.bar(x + width/2, final_vals, width, label=f"Final θ ({len(run_data['rounds'])} rounds)",
           color=COLORS["final"], alpha=0.8, yerr=final_errs, 
           capsize=3, error_kw={"linewidth": 0.8})
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Ability (θ, logit scale)")
    ax.set_title("Ability Estimates: Baseline vs. After Active Learning" + 
                 (f" ({label})" if label else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(valid, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    
    fig.tight_layout()
    fname = f"ability_comparison{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 2: Ability Trajectory (θ over rounds with SE ribbons)
# ════════════════════════════════════════════════════════════════════

def plot_ability_trajectory(run_data: dict, out_dir: str, label: str = ""):
    """Line plot showing how θ estimates evolve round by round."""
    rounds = run_data.get("rounds", [])
    irt_snapshots = run_data.get("irt_snapshots", [])
    
    if not irt_snapshots and rounds:
        irt_snapshots = [r.get("irt_snapshot", {}) for r in rounds]
    
    if not irt_snapshots:
        print("  Skipping ability trajectory: no IRT snapshots")
        return
    
    # Extract theta trajectories per model
    all_models = set()
    for snap in irt_snapshots:
        all_models.update(snap.get("thetas", {}).keys())
    
    # Get SE data per round from round summaries
    round_ses = []
    for r in rounds:
        round_ses.append(r.get("saturation_model_ses", {}))
    
    # Identify saturation-band models (high ability)
    if irt_snapshots:
        last_thetas = irt_snapshots[-1].get("thetas", {})
        saturation_models = [m for m in sorted(all_models) 
                             if last_thetas.get(m, 0) > 0.5]
    else:
        saturation_models = sorted(all_models)
    
    if len(saturation_models) == 0:
        print("  Skipping ability trajectory: no saturation models")
        return
    
    # Build trajectory arrays
    n_rounds = len(irt_snapshots)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(saturation_models) - 1)) 
              for i in range(len(saturation_models))]
    
    # Panel 1: θ trajectory
    for i, model in enumerate(sorted(saturation_models)):
        thetas = [irt_snapshots[r].get("thetas", {}).get(model, np.nan) 
                  for r in range(n_rounds)]
        ax1.plot(range(1, n_rounds + 1), thetas, "-o", color=colors[i],
                 markersize=3, linewidth=1.2, label=model, alpha=0.8)
    
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Ability (θ)")
    ax1.set_title("Ability Trajectories (Saturation Band)")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Legend outside
    handles, labels = ax1.get_legend_handles_labels()
    if len(labels) <= 15:
        ax1.legend(loc="best", fontsize=7, ncol=2)
    else:
        ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, ncol=1)
    
    # Panel 2: SE trajectory
    for i, model in enumerate(sorted(saturation_models)):
        ses = [round_ses[r].get(model, np.nan) if r < len(round_ses) else np.nan
               for r in range(n_rounds)]
        ax2.plot(range(1, n_rounds + 1), ses, "-o", color=colors[i],
                 markersize=3, linewidth=1.2, label=model, alpha=0.8)
    
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Standard Error (SE)")
    ax2.set_title("Standard Error Trajectories")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    fig.tight_layout()
    fname = f"ability_trajectory{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 3: Standard Error & Targeting MAE Progression
# ════════════════════════════════════════════════════════════════════

def plot_se_and_mae_trajectory(run_data: dict, out_dir: str, 
                                compare_data: dict = None, label: str = ""):
    """Dual-axis plot: avg SE reduction + targeting MAE over rounds."""
    summary = run_data.get("summary", {})
    rounds = run_data.get("rounds", [])
    
    if not rounds:
        print("  Skipping SE/MAE trajectory: no round data")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ── Panel 1: Average SE reduction ──
    avg_ses = [r.get("avg_saturation_se", np.nan) for r in rounds]
    initial_se = summary.get("avg_initial_saturation_se", avg_ses[0] if avg_ses else np.nan)
    
    se_vals = [initial_se] + avg_ses
    se_rounds = list(range(0, len(se_vals)))
    
    ax1.plot(se_rounds, se_vals, "-o", color=COLORS["se_ribbon"], 
             markersize=5, linewidth=2, label="Offset = −0.2" if not label else label)
    
    if compare_data and compare_data.get("rounds"):
        c_rounds = compare_data["rounds"]
        c_summary = compare_data.get("summary", {})
        c_ses = [r.get("avg_saturation_se", np.nan) for r in c_rounds]
        c_init = c_summary.get("avg_initial_saturation_se", c_ses[0] if c_ses else np.nan)
        c_vals = [c_init] + c_ses
        ax1.plot(range(len(c_vals)), c_vals, "-s", color=COLORS["no_offset_run"],
                 markersize=5, linewidth=2, label="Offset = 0.0")
        ax1.legend()
    
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Avg Standard Error (logit scale)")
    ax1.set_title("Saturation Band: Average SE Reduction")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Add percentage annotation
    if len(se_vals) >= 2 and not np.isnan(se_vals[0]) and not np.isnan(se_vals[-1]):
        pct_reduction = (se_vals[0] - se_vals[-1]) / se_vals[0] * 100
        ax1.annotate(f"−{pct_reduction:.1f}%", 
                     xy=(se_rounds[-1], se_vals[-1]),
                     xytext=(se_rounds[-1] - 1.5, se_vals[-1] + 0.002),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     fontsize=10, fontweight="bold", color=COLORS["se_ribbon"])
    
    # ── Panel 2: Targeting MAE ──
    mae_vals = [r.get("targeting_mae", r.get("targeting_mae_anchored", np.nan)) 
                for r in rounds]
    
    ax2.plot(range(1, len(mae_vals) + 1), mae_vals, "-o", 
             color=COLORS["targeting"], markersize=5, linewidth=2,
             label="Offset = −0.2" if not label else label)
    
    if compare_data and compare_data.get("rounds"):
        c_mae = [r.get("targeting_mae", r.get("targeting_mae_anchored", np.nan)) 
                 for r in compare_data["rounds"]]
        ax2.plot(range(1, len(c_mae) + 1), c_mae, "-s",
                 color=COLORS["no_offset_run"], markersize=5, linewidth=2,
                 label="Offset = 0.0")
        ax2.legend()
    
    # Add mean line
    if mae_vals:
        mean_mae = np.nanmean(mae_vals)
        ax2.axhline(mean_mae, color=COLORS["targeting"], linestyle="--", 
                     alpha=0.5, linewidth=1)
        ax2.text(0.97, mean_mae, f"Mean: {mean_mae:.3f}", 
                 transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color=COLORS["targeting"])
    
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Targeting MAE (logits)")
    ax2.set_title("Difficulty Targeting Precision")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    fig.tight_layout()
    fname = f"se_mae_trajectory{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 4: Pair Separability Heatmap (Before/After)
# ════════════════════════════════════════════════════════════════════

def plot_separability_heatmap(run_data: dict, out_dir: str, label: str = ""):
    """Heatmap showing confidence transitions for target pairs."""
    summary = run_data.get("summary", {})
    
    initial_seps = summary.get("initial_target_separabilities", {})
    final_seps = summary.get("final_target_separabilities", {})
    
    if not initial_seps or not final_seps:
        # Try to reconstruct from rounds
        rounds = run_data.get("rounds", [])
        if rounds:
            first_r = rounds[0]
            last_r = rounds[-1]
            initial_seps = first_r.get("target_pairs_separability", {})
            final_seps = last_r.get("target_pairs_separability", {})
    
    if not initial_seps:
        print("  Skipping separability heatmap: no pair data")
        return
    
    # Sort pairs by initial confidence
    pairs = sorted(initial_seps.keys(), 
                   key=lambda p: initial_seps[p].get("confidence", 0))
    
    if len(pairs) > 40:
        # Show only the most interesting ones (initially lowest confidence)
        pairs = pairs[:40]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(pairs) * 0.22)))
    
    # Extract values
    init_confs = [initial_seps[p].get("confidence", 0) * 100 for p in pairs]
    final_confs = [final_seps.get(p, {}).get("confidence", 0) * 100 for p in pairs]
    
    # Format pair labels
    pair_labels = [p.replace("_vs_", " vs\n") for p in pairs]
    
    y = np.arange(len(pairs))
    height = 0.35
    
    # Bar chart version (clearer than heatmap for this data)
    bars1 = ax1.barh(y - height/2, init_confs, height, label="Initial",
                      color=COLORS["baseline"], alpha=0.7)
    bars2 = ax1.barh(y + height/2, final_confs, height, label="Final",
                      color=COLORS["final"], alpha=0.7)
    
    ax1.axvline(95, color="red", linewidth=1, linestyle="--", alpha=0.5, label="95% threshold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(pair_labels, fontsize=6)
    ax1.set_xlabel("Confidence (%)")
    ax1.set_title("Pair Separability: Initial vs Final")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(0, 105)
    
    # Panel 2: Confidence gain
    gains = [final_confs[i] - init_confs[i] for i in range(len(pairs))]
    colors_gain = [COLORS["resolved"] if g > 0 else COLORS["targeting"] for g in gains]
    ax2.barh(y, gains, color=colors_gain, alpha=0.7)
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(pair_labels, fontsize=6)
    ax2.set_xlabel("Confidence Change (%p)")
    ax2.set_title("Confidence Gain per Pair")
    
    # Count resolved
    newly_resolved = sum(1 for i in range(len(pairs)) 
                         if init_confs[i] < 95 and final_confs[i] >= 95)
    ax2.text(0.02, 0.98, f"Newly resolved (≥95%): {newly_resolved}/{len(pairs)}",
             transform=ax2.transAxes, fontsize=9, fontweight="bold",
             verticalalignment="top")
    
    fig.tight_layout()
    fname = f"separability{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 5: Kendall Rank Correlation with External Benchmarks
# ════════════════════════════════════════════════════════════════════

def plot_kendall_correlation(run_data: dict, baseline_thetas: dict,
                             external_data: dict, out_dir: str, label: str = ""):
    """Scatter plots + Kendall τ of our ability estimates vs external benchmarks."""
    if not HAS_SCIPY:
        print("  Skipping Kendall correlation: scipy not available")
        return
    
    if not external_data:
        print("  Skipping Kendall correlation: no external benchmark data")
        return
    
    summary = run_data.get("summary", {})
    irt_snapshots = run_data.get("irt_snapshots", [])
    
    # Get final thetas
    if irt_snapshots:
        final_thetas = irt_snapshots[-1].get("thetas", {})
    elif "rounds" in run_data and run_data["rounds"]:
        final_thetas = run_data["rounds"][-1].get("irt_snapshot", {}).get("thetas", {})
    else:
        final_thetas = {}
    
    benchmarks = list(external_data.keys())
    n_benchmarks = len(benchmarks)
    
    if n_benchmarks == 0:
        print("  Skipping Kendall correlation: no benchmarks")
        return
    
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(6 * n_benchmarks, 5))
    if n_benchmarks == 1:
        axes = [axes]
    
    for ax, bench_name in zip(axes, benchmarks):
        bench_scores = external_data[bench_name]  # {model: score}
        
        # Find common models
        common = sorted(set(final_thetas.keys()) & set(bench_scores.keys()))
        if len(common) < 3:
            ax.text(0.5, 0.5, f"Too few common\nmodels ({len(common)})",
                    transform=ax.transAxes, ha="center")
            ax.set_title(bench_name)
            continue
        
        our_vals = [float(final_thetas[m]) for m in common]
        ext_vals = [float(bench_scores[m]) for m in common]
        
        # Also compute baseline correlation for comparison
        base_common = sorted(set(baseline_thetas.keys()) & set(bench_scores.keys()))
        
        # Scatter plot
        ax.scatter(our_vals, ext_vals, c=COLORS["baseline"], s=40, alpha=0.7, 
                   edgecolors="white", linewidth=0.5)
        
        # Label points
        for m, x, y in zip(common, our_vals, ext_vals):
            ax.annotate(m, (x, y), fontsize=5, alpha=0.6,
                        xytext=(3, 3), textcoords="offset points")
        
        # Kendall correlation
        tau, p_val = kendalltau(our_vals, ext_vals)
        rho, _ = spearmanr(our_vals, ext_vals)
        
        ax.set_xlabel("Our θ (ARC-Easy + generated items)")
        ax.set_ylabel(f"{bench_name} accuracy")
        ax.set_title(f"vs. {bench_name}\nKendall τ = {tau:.3f} (p={p_val:.3g}), Spearman ρ = {rho:.3f}")
    
    fig.tight_layout()
    fname = f"kendall_correlation{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 6: Generated Question Examples
# ════════════════════════════════════════════════════════════════════

def save_question_examples(run_data: dict, out_dir: str, label: str = "",
                           n_examples: int = 12):
    """Save a formatted markdown table of generated question examples.
    
    Selects a diverse set: best-targeted, typical, and worst-targeted items,
    deduplicating by question text and filtering degenerate calibrations.
    """
    # Force loading from round details to preserve true target difficulties
    all_questions = []
    for r in run_data.get("rounds", []):
        for q in r.get("question_details", []):
            all_questions.append(q)
            
    if not all_questions and run_data.get("all_generated_questions"):
        all_questions = run_data["all_generated_questions"]
    
    if not all_questions:
        print("  Skipping question examples: no generated questions")
        return
    
    # Helper to get calibrated difficulty
    def get_cal_b(q):
        for k in ["calibrated_difficulty_anchored", "calibrated_difficulty_fpc",
                  "calibrated_difficulty", "calibrated_difficulty_anchored_fpc"]:
            if q.get(k) is not None:
                return float(q[k])
        t = q.get("target_difficulty", q.get("difficulty"))
        return float(t) if t is not None else 0.0

    # --- Deduplication by question text (not just ID, which may be empty) ---
    seen_texts = set()
    unique_questions = []
    for q in all_questions:
        text = q.get("question_text", q.get("question", "")).strip()[:120]
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_questions.append(q)
    
    # --- Filter degenerate calibrations (items pinned at boundary values) ---
    cal_values = [get_cal_b(q) for q in unique_questions]
    if cal_values:
        from collections import Counter
        val_counts = Counter(round(v, 2) for v in cal_values)
        # A value appearing in >20% of items is likely a degenerate boundary
        n_total = len(unique_questions)
        degenerate_vals = {v for v, c in val_counts.items() if c > max(3, n_total * 0.1)}
        if degenerate_vals:
            filtered = [q for q in unique_questions 
                        if round(get_cal_b(q), 2) not in degenerate_vals]
            print(f"    Filtered {len(unique_questions) - len(filtered)} items with "
                  f"degenerate calibration values: {degenerate_vals}")
            unique_questions = filtered if filtered else unique_questions
    
    # --- Compute targeting errors ---
    for q in unique_questions:
        target = float(q.get("target_difficulty", q.get("difficulty", 0)))
        q["_targeting_error"] = abs(get_cal_b(q) - target)
    
    sorted_by_error = sorted(unique_questions, key=lambda q: q["_targeting_error"])
    
    # Select categorized examples
    n_best = 4
    n_typical = 4
    n_worst = 4
    
    best = sorted_by_error[:n_best]
    worst = sorted_by_error[-n_worst:]
    
    # Typical = items near the median targeting error
    median_idx = len(sorted_by_error) // 2
    half_window = n_typical // 2
    typical = sorted_by_error[max(0, median_idx - half_window):
                               median_idx + half_window + n_typical % 2]
    
    # Combine, dedup by text again
    examples = []
    seen = set()
    for category, items, cat_label in [
            ("Best-targeted", best, "best"),
            ("Typical", typical, "typical"),
            ("Worst-targeted", worst, "worst")]:
        for q in items:
            text = q.get("question_text", q.get("question", "")).strip()[:120]
            if text not in seen:
                seen.add(text)
                q["_category"] = category
                examples.append(q)
    
    examples = examples[:n_examples]
    
    # Write as markdown
    fname = f"question_examples{'_' + label if label else ''}.md"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as f:
        f.write("# Generated Question Examples\n\n")
        f.write(f"Selected from {len(all_questions)} total generated items "
                f"({len(unique_questions)} unique after deduplication).\n\n")
        
        current_cat = None
        for i, q in enumerate(examples, 1):
            cat = q.get("_category", "")
            if cat != current_cat:
                current_cat = cat
                f.write(f"### {cat} Items\n\n")
            
            target_b = float(q.get("target_difficulty", q.get("difficulty", 0)))
            cal_b = get_cal_b(q)
            error = q["_targeting_error"]
            
            f.write(f"**Example {i}** — "
                    f"target: {target_b:.3f}, calibrated: {cal_b:.3f}, "
                    f"error: {error:.3f} logits\n\n")
            f.write(f"> {q.get('question_text', q.get('question', 'N/A'))}\n\n")
            
            options = q.get("options", [])
            if isinstance(options, dict):
                for key, val in sorted(options.items()):
                    marker = "✓" if key == q.get("correct_answer", "") else " "
                    f.write(f"  {key}. {val} {marker}\n")
            elif isinstance(options, list):
                labels = "ABCDEFGH"
                correct = q.get("correct_answer", "")
                for j, opt in enumerate(options):
                    lbl = labels[j] if j < len(labels) else str(j)
                    marker = "✓" if lbl == correct or opt == correct else " "
                    f.write(f"  {lbl}. {opt} {marker}\n")
            
            responses = q.get("model_responses", {})
            if responses:
                n_correct = sum(1 for v in responses.values() if v == 1)
                f.write(f"\n  *{n_correct}/{len(responses)} models answered correctly*\n")
            f.write("\n---\n\n")
    
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 7: Difficulty Shift Histogram (AddOption)
# ════════════════════════════════════════════════════════════════════

def plot_difficulty_shift(run_data: dict, out_dir: str, label: str = ""):
    """Histogram of (calibrated_b - target_b) to show AddOption systematic shift."""
    rounds = run_data.get("rounds", [])
    
    all_shifts = []
    for r in rounds:
        stats = r.get("difficulty_shift_stats", {})
        anchored = stats.get("anchored", {})
        vals = anchored.get("values", [])
        all_shifts.extend(vals)
    
    if not all_shifts:
        # Try from question_details
        for r in rounds:
            for q in r.get("question_details", []):
                target = q.get("target_difficulty")
                cal = q.get("calibrated_difficulty")
                if target is not None and cal is not None:
                    all_shifts.append(float(cal) - float(target))
    
    if not all_shifts:
        print("  Skipping difficulty shift: no data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    shifts = np.array(all_shifts)
    ax.hist(shifts, bins=30, color=COLORS["targeting"], alpha=0.7, 
            edgecolor="white", linewidth=0.5, density=True)
    
    mean_shift = np.mean(shifts)
    median_shift = np.median(shifts)
    mae = np.mean(np.abs(shifts))
    
    ax.axvline(mean_shift, color="black", linewidth=2, linestyle="-",
               label=f"Mean: {mean_shift:+.3f}")
    ax.axvline(median_shift, color="gray", linewidth=2, linestyle="--",
               label=f"Median: {median_shift:+.3f}")
    ax.axvline(0, color="green", linewidth=1, linestyle=":", alpha=0.5)
    
    ax.set_xlabel("Targeting Error (calibrated_b − target_b, logits)")
    ax.set_ylabel("Density")
    ax.set_title(f"Difficulty Targeting Error Distribution (N={len(shifts)}, MAE={mae:.3f})")
    ax.legend()
    
    # Add text box with stats
    textstr = f"Mean: {mean_shift:+.3f}\nMedian: {median_shift:+.3f}\nMAE: {mae:.3f}\nStd: {np.std(shifts):.3f}\nN: {len(shifts)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)
    
    fig.tight_layout()
    fname = f"targeting_error_histogram{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 8: Pairs Resolved Trajectory
# ════════════════════════════════════════════════════════════════════

def plot_pairs_resolved(run_data: dict, out_dir: str, 
                        compare_data: dict = None, label: str = ""):
    """Cumulative count of newly resolved pairs over rounds."""
    rounds = run_data.get("rounds", [])
    summary = run_data.get("summary", {})
    initial_seps = summary.get("initial_target_separabilities", {})
    
    if not rounds or not initial_seps:
        print("  Skipping pairs resolved: no data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    def _count_resolved(rds, init_seps):
        counts = []
        for r in rds:
            final_seps_r = r.get("target_pairs_separability", {})
            n_resolved = sum(
                1 for pk in init_seps
                if init_seps[pk].get("confidence", 0) < 0.95
                and final_seps_r.get(pk, {}).get("confidence", 0) >= 0.95
            )
            counts.append(n_resolved)
        return counts
    
    counts = _count_resolved(rounds, initial_seps)
    total_unresolved = sum(1 for v in initial_seps.values() 
                           if v.get("confidence", 0) < 0.95)
    
    ax.plot(range(1, len(counts) + 1), counts, "-o", color=COLORS["resolved"],
            markersize=6, linewidth=2, label="Offset = −0.2" if not label else label)
    
    if compare_data and compare_data.get("rounds"):
        c_summary = compare_data.get("summary", {})
        c_init = c_summary.get("initial_target_separabilities", initial_seps)
        c_counts = _count_resolved(compare_data["rounds"], c_init)
        ax.plot(range(1, len(c_counts) + 1), c_counts, "-s", 
                color=COLORS["no_offset_run"],
                markersize=6, linewidth=2, label="Offset = 0.0")
        ax.legend()
    
    ax.axhline(total_unresolved, color="gray", linewidth=1, linestyle="--",
               alpha=0.5, label=f"Max possible: {total_unresolved}")
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Pairs Resolved (≥95% confidence)")
    ax.set_title("Model Pair Separability: Resolution Progress")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    
    fig.tight_layout()
    fname = f"pairs_resolved{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 9: Ability MAE Drift (Scale Stability)
# ════════════════════════════════════════════════════════════════════

def plot_ability_mae_drift(run_data: dict, out_dir: str, 
                           compare_data: dict = None, label: str = ""):
    """Plot showing how much ability estimates drift from baseline over rounds."""
    rounds = run_data.get("rounds", [])
    
    if not rounds:
        print("  Skipping ability MAE drift: no round data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    mae_vals = [r.get("ability_mae", np.nan) for r in rounds]
    ax.plot(range(1, len(mae_vals) + 1), mae_vals, "-o", color=COLORS["baseline"],
            markersize=6, linewidth=2, label="Offset = −0.2" if not label else label)
    
    if compare_data and compare_data.get("rounds"):
        c_mae = [r.get("ability_mae", np.nan) for r in compare_data["rounds"]]
        ax.plot(range(1, len(c_mae) + 1), c_mae, "-s", color=COLORS["no_offset_run"],
                markersize=6, linewidth=2, label="Offset = 0.0")
        ax.legend()
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Absolute Error (logits)")
    ax.set_title("Scale Stability: Ability Drift from Baseline")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Add annotation
    if mae_vals:
        ax.text(0.97, 0.97, f"Final MAE: {mae_vals[-1]:.4f}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                verticalalignment="top", horizontalalignment="right")
    
    fig.tight_layout()
    fname = f"ability_mae_drift{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 10: Summary Dashboard (Combined)
# ════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(run_data: dict, out_dir: str, label: str = ""):
    """4-panel summary dashboard of key metrics."""
    summary = run_data.get("summary", {})
    rounds = run_data.get("rounds", [])
    
    if not rounds:
        print("  Skipping summary dashboard: no round data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Panel 1: SE trajectory
    avg_ses = [r.get("avg_saturation_se", np.nan) for r in rounds]
    init_se = summary.get("avg_initial_saturation_se", avg_ses[0] if avg_ses else np.nan)
    se_vals = [init_se] + avg_ses
    axes[0, 0].plot(range(len(se_vals)), se_vals, "-o", color=COLORS["se_ribbon"],
                     markersize=5, linewidth=2)
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Avg SE")
    axes[0, 0].set_title("Avg Standard Error")
    axes[0, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Panel 2: Targeting MAE
    mae_vals = [r.get("targeting_mae", r.get("targeting_mae_anchored", np.nan)) 
                for r in rounds]
    axes[0, 1].plot(range(1, len(mae_vals) + 1), mae_vals, "-o",
                     color=COLORS["targeting"], markersize=5, linewidth=2)
    if mae_vals:
        axes[0, 1].axhline(np.nanmean(mae_vals), color=COLORS["targeting"],
                            linestyle="--", alpha=0.4)
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Targeting MAE (logits)")
    axes[0, 1].set_title("Targeting Precision")
    axes[0, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Panel 3: Ability MAE drift
    drift_vals = [r.get("ability_mae", np.nan) for r in rounds]
    axes[1, 0].plot(range(1, len(drift_vals) + 1), drift_vals, "-o",
                     color=COLORS["baseline"], markersize=5, linewidth=2)
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Ability MAE vs Baseline")
    axes[1, 0].set_title("Scale Stability (Ability Drift)")
    axes[1, 0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # Panel 4: Pairs resolved (from separability data)
    initial_seps = summary.get("initial_target_separabilities", {})
    if initial_seps:
        counts = []
        for r in rounds:
            final_seps_r = r.get("target_pairs_separability", {})
            n = sum(1 for pk in initial_seps
                    if initial_seps[pk].get("confidence", 0) < 0.95
                    and final_seps_r.get(pk, {}).get("confidence", 0) >= 0.95)
            counts.append(n)
        axes[1, 1].plot(range(1, len(counts) + 1), counts, "-o",
                         color=COLORS["resolved"], markersize=5, linewidth=2)
        total = sum(1 for v in initial_seps.values() if v.get("confidence", 0) < 0.95)
        axes[1, 1].axhline(total, color="gray", linestyle="--", alpha=0.4)
        axes[1, 1].text(0.5, total + 1, f"Max: {total}", fontsize=8, color="gray")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Pairs Resolved (≥95%)")
    axes[1, 1].set_title("Separability Resolution")
    axes[1, 1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    fig.suptitle(f"Active Learning Summary Dashboard ({len(rounds)} rounds)", 
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"summary_dashboard{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")


# ════════════════════════════════════════════════════════════════════
# Figure 11: Items Needed Under Original Distribution
# ════════════════════════════════════════════════════════════════════

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def plot_items_needed(run_data: dict, out_dir: str, label: str = ""):
    """Compute and plot how many original-distribution items would be needed
    to achieve the same SE reduction that the targeted generation achieved.
    
    Uses Rasch Fisher information: I_j(θ) = p_j(1 − p_j)
    where p_j = logistic(θ − b_j).
    
    For each saturation-band model, we compare:
    - avg info per item from the original ARC-Easy difficulty distribution
    - avg info per item from the generated (targeted) difficulty distribution
    and compute how many more original-dist items would be needed to reach
    the same total information (i.e. same final SE).
    """
    summary = run_data.get("summary", {})
    final_irt = run_data.get("final_irt_state", {})
    
    if not final_irt:
        # Try loading from raw_data
        raw_dir = Path(run_data["run_dir"]) / "raw_data" / "final_irt_state.json"
        if raw_dir.exists():
            with open(raw_dir) as f:
                final_irt = json.load(f)
    
    if not final_irt or "difficulties" not in final_irt:
        print("  Skipping items-needed: no final IRT state")
        return
    
    diffs = final_irt["difficulties"]
    final_thetas = final_irt["thetas"]
    
    # Separate original vs generated items
    orig_b = np.array([float(v) for k, v in diffs.items() if not k.startswith("gen_")])
    gen_b = np.array([float(v) for k, v in diffs.items() if k.startswith("gen_")])
    n_orig = len(orig_b)
    n_gen = len(gen_b)
    
    if n_orig == 0 or n_gen == 0:
        print("  Skipping items-needed: can't separate original/generated items")
        return
    
    init_ses = summary.get("initial_saturation_model_ses", {})
    final_ses = summary.get("final_saturation_model_ses", {})
    sat_models = list(init_ses.keys())
    
    if not sat_models:
        print("  Skipping items-needed: no saturation model SEs")
        return
    
    # Compute per-model statistics
    models_sorted = sorted(sat_models, key=lambda m: float(final_thetas.get(m, 0)))
    thetas_arr = []
    info_ratios = []
    additional_items = []
    se_reductions = []
    
    for m in models_sorted:
        theta = float(final_thetas[m])
        thetas_arr.append(theta)
        
        # Fisher info per item from each distribution
        p_orig = _sigmoid(theta - orig_b)
        info_per_orig = np.mean(p_orig * (1 - p_orig))
        
        p_gen = _sigmoid(theta - gen_b)
        info_per_gen = np.mean(p_gen * (1 - p_gen))
        
        ratio = info_per_gen / info_per_orig if info_per_orig > 0 else 1.0
        info_ratios.append(ratio)
        
        # Total info needed to reach final SE
        final_se = float(final_ses.get(m, 0))
        init_se = float(init_ses.get(m, 0))
        if final_se > 0:
            target_info = 1.0 / (final_se ** 2)
            n_needed = target_info / info_per_orig
            additional_items.append(max(0, n_needed - n_orig))
        else:
            additional_items.append(0)
        
        se_reductions.append((init_se - final_se) / init_se * 100 if init_se > 0 else 0)
    
    avg_additional = np.mean(additional_items)
    avg_ratio = np.mean(info_ratios)
    efficiency = avg_additional / n_gen if n_gen > 0 else 1.0
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Panel 1: Additional items needed (bar chart by model)
    x = np.arange(len(models_sorted))
    bars = ax1.bar(x, additional_items, color=COLORS["baseline"], alpha=0.7,
                   edgecolor="white", linewidth=0.5)
    ax1.axhline(n_gen, color=COLORS["targeting"], linewidth=2, linestyle="--",
                label=f"Actual generated items: {n_gen}")
    ax1.axhline(avg_additional, color="black", linewidth=1.5, linestyle=":",
                label=f"Avg needed: {avg_additional:.0f}")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_sorted, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("Additional original-dist items needed")
    ax1.set_title(f"Items Needed to Match SE Reduction\n"
                  f"(efficiency factor: {efficiency:.1f}×)")
    ax1.legend(loc="upper left", fontsize=8)
    
    # Panel 2: Information ratio vs ability
    ax2.scatter(thetas_arr, info_ratios, c=COLORS["final"], s=50, alpha=0.8,
                edgecolors="white", linewidth=0.5, zorder=5)
    for m, th, ratio in zip(models_sorted, thetas_arr, info_ratios):
        ax2.annotate(m, (th, ratio), fontsize=5, alpha=0.6,
                     xytext=(3, 3), textcoords="offset points")
    
    ax2.axhline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5,
                label="Equal efficiency")
    ax2.axhline(avg_ratio, color=COLORS["final"], linewidth=1.5, linestyle=":",
                label=f"Mean ratio: {avg_ratio:.2f}×")
    
    ax2.set_xlabel("Model ability (θ)")
    ax2.set_ylabel("Info ratio (generated / original per item)")
    ax2.set_title("Per-Item Information Gain\nfrom Targeted Generation")
    ax2.legend(loc="upper left", fontsize=8)
    
    # Add text box with summary
    textstr = (f"Original items: {n_orig}\n"
               f"Generated items: {n_gen}\n"
               f"Avg SE reduction: {np.mean(se_reductions):.1f}%\n"
               f"Avg additional orig items\n  for same reduction: {avg_additional:.0f}\n"
               f"Efficiency factor: {efficiency:.1f}×")
    props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    ax2.text(0.97, 0.03, textstr, transform=ax2.transAxes, fontsize=8,
             verticalalignment="bottom", horizontalalignment="right", bbox=props)
    
    fig.tight_layout()
    fname = f"items_needed_efficiency{'_' + label if label else ''}.pdf"
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)
    print(f"  Saved {fname}")
    
    # Also save a CSV with the per-model data
    csv_fname = f"items_needed_efficiency{'_' + label if label else ''}.csv"
    csv_path = os.path.join(out_dir, csv_fname)
    with open(csv_path, "w") as f:
        f.write("model,theta,init_se,final_se,se_reduction_pct,"
                "info_per_orig,info_per_gen,info_ratio,additional_items_needed\n")
        for i, m in enumerate(models_sorted):
            theta = thetas_arr[i]
            p_orig = _sigmoid(theta - orig_b)
            info_po = np.mean(p_orig * (1 - p_orig))
            p_gen = _sigmoid(theta - gen_b)
            info_pg = np.mean(p_gen * (1 - p_gen))
            f.write(f"{m},{theta:.4f},{init_ses.get(m,0):.5f},{final_ses.get(m,0):.5f},"
                    f"{se_reductions[i]:.1f},{info_po:.5f},{info_pg:.5f},"
                    f"{info_ratios[i]:.3f},{additional_items[i]:.0f}\n")
    print(f"  Saved {csv_fname}")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures from run data")
    parser.add_argument("--run-dir", required=True, 
                        help="Path to active learning run directory")
    parser.add_argument("--compare-dir", default=None,
                        help="Optional second run directory for comparison plots")
    parser.add_argument("--project-dir", 
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Project root directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for figures (default: final_manuscript/figures/)")
    parser.add_argument("--label", default="",
                        help="Label suffix for figure filenames")
    parser.add_argument("--external-benchmarks", default=None,
                        help="JSON file with external benchmark scores {bench: {model: score}}")
    
    args = parser.parse_args()
    
    # Set up output directory
    out_dir = args.out_dir or os.path.join(args.project_dir, "final_manuscript", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    # Resolve run directory
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(args.project_dir, run_dir)
    
    print(f"=" * 60)
    print(f"  Generating Manuscript Figures")
    print(f"  Run directory: {run_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"=" * 60)
    
    # Load primary run data
    print("\nLoading primary run data...")
    run_data = load_run(run_dir)
    
    # Load comparison run if provided
    compare_data = None
    if args.compare_dir:
        compare_dir = args.compare_dir
        if not os.path.isabs(compare_dir):
            compare_dir = os.path.join(args.project_dir, compare_dir)
        print(f"\nLoading comparison run data...")
        compare_data = load_run(compare_dir)
    
    # Load baseline thetas
    print("\nLoading baseline ability estimates...")
    baseline_thetas = load_baseline_thetas(args.project_dir)
    
    # Load external benchmark data
    external_data = {}
    if args.external_benchmarks and os.path.exists(args.external_benchmarks):
        with open(args.external_benchmarks) as f:
            external_data = json.load(f)
        print(f"  Loaded external benchmarks: {list(external_data.keys())}")
    
    # Generate all figures
    print("\n" + "─" * 60)
    print("  Generating figures...")
    print("─" * 60)
    
    plot_ability_comparison(run_data, baseline_thetas, out_dir, args.label)
    plot_ability_trajectory(run_data, out_dir, args.label)
    plot_se_and_mae_trajectory(run_data, out_dir, compare_data, args.label)
    plot_separability_heatmap(run_data, out_dir, args.label)
    plot_kendall_correlation(run_data, baseline_thetas, external_data, out_dir, args.label)
    save_question_examples(run_data, out_dir, args.label)
    plot_difficulty_shift(run_data, out_dir, args.label)
    plot_pairs_resolved(run_data, out_dir, compare_data, args.label)
    plot_ability_mae_drift(run_data, out_dir, compare_data, args.label)
    plot_summary_dashboard(run_data, out_dir, args.label)
    plot_items_needed(run_data, out_dir, args.label)
    
    print("\n" + "=" * 60)
    print(f"  All figures saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
