import numpy as np
import pandas as pd
from interfaces import TargetSelector, TargetProfile, Benchmark, MidpointTargetSelectorConfig

class MidpointTargetSelector(TargetSelector):
    """
    Concrete Target Selector that retrieves the calibrated abilities (thetas) of
    two specific target models (model_a and model_b) from the IRTModel calibration state,
    and targets the exact mathematical midpoint between them.
    """
    def __init__(self, config: MidpointTargetSelectorConfig):
        self.config = config

    def select_target(self, benchmark: Benchmark) -> TargetProfile:
        irt = benchmark.calibrated_model
        
        # Get capabilities of the two configured target models (1D midpoints read index 0)
        # TODO support factor model
        t1 = irt.get_subject_ability(self.config.model_a)[0]
        t2 = irt.get_subject_ability(self.config.model_b)[0]
        # Compute their exact midpoint difficulty
        target_diff = (t1 + t2) / 2.0
        
        return TargetProfile(target_difficulty=target_diff)
