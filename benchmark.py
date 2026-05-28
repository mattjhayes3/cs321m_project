"""
Benchmark Calibration & Loading Module.
Implements the concrete RealBenchmark class and load_calibrated_benchmark function
to fetch calibrated psychometric data directly from the Modal persistent Volume.
"""

import io
import pandas as pd
from typing import List, Dict, Any, Tuple
from interfaces import Benchmark, Question, IRTModel
from irt_model import RaschModel

class RealBenchmark(Benchmark):
    """Concrete Benchmark backed by real ARC-Easy calibrated data."""
    def get_response_matrix(self, model_ids: List[str]) -> pd.DataFrame:
        # Dummy implementation as the response matrix is loaded directly from volume
        return pd.DataFrame()

def load_calibrated_benchmark(volume, model_subset: List[str] = None) -> Tuple[RealBenchmark, pd.DataFrame]:
    """
    Load calibrated IRT parameters + original response matrix from the persistent Volume.

    Args:
        volume: Modal Volume client object.
        model_subset: Optional list of model short names to filter.

    Returns:
        Tuple[RealBenchmark, pd.DataFrame] representing the loaded benchmark pool
        and the baseline response matrix.
    """
    print("=== Loading Calibrated Benchmark ===")

    # 1. Read ability estimates (Rasch Theta)
    ability_bytes = b""
    for chunk in volume.read_file("arc_easy_eval/theta_comparison_comprehensive_acc_norm.csv"):
        ability_bytes += chunk
    ability_df = pd.read_csv(io.BytesIO(ability_bytes))

    # 2. Read item parameters
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

    # 3. Read original response matrix
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

    # 4. Populate RaschModel & Question pool
    irt = RaschModel()
    
    # Map baseline theta to dict
    irt.thetas = dict(zip(ability_df["Model"], ability_df["Rasch_Theta"]))
    
    # Map baseline difficulties and EM discriminations (for discernability filtering)
    irt.difficulties = dict(zip(item_params_df["item_id"], item_params_df["Rasch_Difficulty"]))
    irt.discriminations = dict(zip(item_params_df["item_id"], item_params_df["EM_Discrimination"]))
    
    irt.valid_items = list(irt.difficulties.keys())

    # Filter original response matrix to the specified model subset
    if model_subset:
        available = [m for m in model_subset if m in original_response_matrix.index]
        original_response_matrix = original_response_matrix.loc[available]
    print(f"  Original response matrix: {original_response_matrix.shape}")

    # Load ARC-Easy questions from HuggingFace
    from datasets import load_dataset
    print("  Loading ARC-Easy questions from HuggingFace...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    
    calibrated_questions = []
    for row in ds:
        q_id = row["id"]
        if q_id in irt.difficulties:
            q = Question(
                id=q_id,
                question_text=row["question"],
                options=row["choices"]["text"],
                correct_answer=row["answerKey"],
                difficulty=irt.difficulties[q_id],
                discrimination=irt.discriminations[q_id],
                calibrated=True
            )
            calibrated_questions.append(q)
            
    print(f"  Loaded {len(calibrated_questions)} calibrated questions, {len(irt.thetas)} model abilities")

    benchmark = RealBenchmark(calibrated_questions, irt)
    return benchmark, original_response_matrix
