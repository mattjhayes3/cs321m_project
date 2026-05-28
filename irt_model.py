import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.special import expit

from interfaces import IRTModel

class RaschModel(IRTModel):
    """
    Concrete Item Response Theory Rasch (1PL) model implementation
    using the torch_measure framework for model calibration.
    """
    def __init__(self):
        self.thetas: Dict[str, float] = {}
        self.difficulties: Dict[str, float] = {}
        self.discriminations: Dict[str, float] = {}
        self.valid_items: List[str] = []
        self.model_ids: List[str] = []
        self.item_ids: List[str] = []

    def fit(self, response_matrix: pd.DataFrame) -> None:
        import torch
        import sys
        from unittest.mock import MagicMock
        sys.modules["tabpfn"] = MagicMock()
        sys.modules["pyro"] = MagicMock()
        sys.modules["pyro.distributions"] = MagicMock()
        from torch_measure.models import Rasch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        R = response_matrix.values.astype(float)
        n_subjects, n_items = R.shape
        self.model_ids = list(response_matrix.index)
        self.item_ids = list(response_matrix.columns)

        # Filter items with positive variance (non-trivial items)
        item_var = np.nanvar(R, axis=0)
        self.valid_mask = item_var > 0
        R_valid = R[:, self.valid_mask]
        self.valid_items = [self.item_ids[i] for i in range(n_items) if self.valid_mask[i]]
        n_valid = len(self.valid_items)

        if n_valid == 0:
            raise ValueError("No items with positive variance found in response matrix.")

        R_tensor = torch.tensor(R_valid, dtype=torch.float32, device=device)
        R_tensor = torch.nan_to_num(R_tensor, nan=0.0)

        # Instantiate and fit Rasch model
        irt_model = Rasch(n_subjects=n_subjects, n_items=n_valid, device=device)
        irt_model.fit(R_tensor, method="mle", max_epochs=2000, lr=0.01, verbose=False)

        with torch.no_grad():
            thetas = irt_model.ability.cpu().numpy()
            b_raw = irt_model.difficulty.cpu().numpy()
            
            # Grid queries to predict subject-item parameters directiontraditional vs intercept
            sub_grid = torch.arange(n_subjects, device=device)[:, None].expand(n_subjects, n_valid).flatten()
            item_grid = torch.arange(n_valid, device=device)[None, :].expand(n_subjects, n_valid).flatten()
            query = {"subject_idx": sub_grid, "item_idx": item_grid}
            predicted_probs = irt_model.predict(query).reshape(n_subjects, n_valid).cpu().numpy()

        # Traditional Rasch parameters check: traditional is sig(theta - b), intercept is sig(theta + b)
        traditional_probs = expit(thetas[:, None] - b_raw[None, :])
        intercept_probs = expit(thetas[:, None] + b_raw[None, :])
        
        if np.abs(predicted_probs - intercept_probs).mean() < np.abs(predicted_probs - traditional_probs).mean():
            b_params = -b_raw
        else:
            b_params = b_raw

        # Store fitted capabilities (thetas) and item difficulties
        self.thetas = {self.model_ids[i]: float(thetas[i]) for i in range(n_subjects)}
        self.difficulties = {self.valid_items[j]: float(b_params[j]) for j in range(n_valid)}

    def get_subject_ability(self, subject_id: str) -> List[float]:
        val = self.thetas.get(subject_id, 0.0)
        return [float(val)]

    def get_item_information(self, item_id: str, theta: np.ndarray) -> np.ndarray:
        if item_id not in self.difficulties:
            return np.zeros_like(theta)
        b = self.difficulties[item_id]
        p = expit(theta - b)
        return p * (1.0 - p)

    def compute_fisher_information(self, theta: np.ndarray) -> np.ndarray:
        I_total = np.zeros_like(theta)
        for item_id in self.valid_items:
            I_total += self.get_item_information(item_id, theta)
        return I_total

    def compute_ability_se(self, subject_id: str) -> float:
        """
        Compute asymptotic standard error of θ for a given model.

        SE(θ) = 1 / sqrt(I(θ))  where I(θ) = Σ_i p_i(1-p_i)
        and p_i = sigmoid(θ - b_i).
        """
        theta = self.thetas.get(subject_id, 0.0)
        theta_arr = np.array([theta])
        info = self.compute_fisher_information(theta_arr)[0]
        if info <= 0:
            return float("inf")
        return 1.0 / np.sqrt(info)

    def compute_separability(self, model_a: str, model_b: str) -> Dict[str, float]:
        """
        Compute the statistical confidence that two models have different abilities.

        Uses a Wald test on the difference θ_a - θ_b:
          SE(θ_a - θ_b) = sqrt(SE(θ_a)² + SE(θ_b)²)
          z = (θ_a - θ_b) / SE(diff)
          p_value = 2 * Φ(-|z|)
          confidence = 1 - p_value

        Returns dict with: theta_a, theta_b, se_a, se_b, se_diff, z, p_value, confidence
        """
        from scipy.stats import norm

        theta_a = self.thetas.get(model_a, 0.0)
        theta_b = self.thetas.get(model_b, 0.0)
        se_a = self.compute_ability_se(model_a)
        se_b = self.compute_ability_se(model_b)
        se_diff = np.sqrt(se_a**2 + se_b**2)

        if se_diff <= 0 or not np.isfinite(se_diff):
            z = 0.0
            p_value = 1.0
        else:
            z = (theta_a - theta_b) / se_diff
            p_value = 2.0 * norm.sf(abs(z))  # two-tailed

        return {
            "theta_a": float(theta_a),
            "theta_b": float(theta_b),
            "se_a": float(se_a),
            "se_b": float(se_b),
            "se_diff": float(se_diff),
            "z": float(z),
            "p_value": float(p_value),
            "confidence": float(1.0 - p_value),
        }


    def fit_anchored(
        self,
        new_response_matrix: pd.DataFrame,
        original_response_matrix: pd.DataFrame = None,
        max_iter: int = 5000,
        lr: float = 0.001,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Jointly re-estimate model abilities (θ) and new item difficulties,
        while holding existing item difficulties FIXED.

        This implements anchored calibration where:
        - Existing item difficulties (b) are HELD CONSTANT — they are
          well-calibrated from the full benchmark (40+ models × 2000+ items)
        - New item difficulties are ESTIMATED from scratch
        - Model abilities (θ) are RE-ESTIMATED using ALL items (original + new)
          — this is the whole point: new items provide additional information
          to sharpen our confidence in model abilities

        For the Rasch model: P(correct | θ_j, b_i) = sigmoid(θ_j - b_i)

        We jointly optimize θ_j (for all models) and b_i (for new items only)
        via gradient descent on the negative log-likelihood.

        Args:
            new_response_matrix: DataFrame with model names as index (rows),
                new question IDs as columns, values are 0/1 (correct/incorrect).
            original_response_matrix: Optional DataFrame with model names as
                index, original item IDs as columns, values are 0/1.
                If provided, the original responses contribute to θ estimation
                (with their difficulties held fixed). If None, only the new
                responses are used (θ estimates may shift more).
            max_iter: Maximum gradient descent iterations.
            lr: Learning rate.
            tol: Convergence tolerance (max absolute gradient).

        Returns:
            Dict mapping new item IDs to their estimated difficulties.
            Also updates self.thetas with re-estimated model abilities.
        """
        # Identify models present in the new response matrix AND in our calibration
        model_names = [m for m in new_response_matrix.index if m in self.thetas]
        if not model_names:
            raise ValueError(
                "No models in the response matrix match calibrated model abilities. "
                f"Response matrix models: {list(new_response_matrix.index)}, "
                f"Calibrated models: {list(self.thetas.keys())}"
            )

        n_models = len(model_names)
        new_item_ids = list(new_response_matrix.columns)
        n_new = len(new_item_ids)

        # Initialize θ from existing calibration
        theta_arr = np.array([self.thetas[m] for m in model_names])

        # Collect new-item responses: shape (n_models, n_new)
        R_new = new_response_matrix.loc[model_names, new_item_ids].values.astype(float)
        mask_new = ~np.isnan(R_new)  # valid entries

        # Initialize new b values from observed proportion correct
        b_new = np.zeros(n_new)
        for i in range(n_new):
            valid = mask_new[:, i]
            if valid.sum() == 0:
                b_new[i] = 0.0
                continue
            p_obs = np.clip(R_new[valid, i].mean(), 0.01, 0.99)
            b_new[i] = theta_arr[valid].mean() - np.log(p_obs / (1.0 - p_obs))

        # Collect original-item responses if available
        R_orig = None
        mask_orig = None
        b_orig = None
        if original_response_matrix is not None:
            # Only use original items whose difficulties we already have
            # Deduplicate columns first to avoid quadratic duplicates replication from index slicing
            unique_cols = ~original_response_matrix.columns.duplicated()
            original_matrix_unique = original_response_matrix.loc[:, unique_cols]
            
            orig_items = [c for c in original_matrix_unique.columns
                          if c in self.difficulties]
            orig_models = [m for m in model_names
                           if m in original_matrix_unique.index]
            if orig_items and orig_models:
                # Reindex to match model_names ordering
                R_orig_df = original_matrix_unique.loc[orig_models, orig_items]
                # Map to same model ordering
                model_to_idx = {m: i for i, m in enumerate(model_names)}
                orig_model_idxs = [model_to_idx[m] for m in orig_models]

                R_orig_full = np.full((n_models, len(orig_items)), np.nan)
                R_orig_full[orig_model_idxs, :] = R_orig_df.values.astype(float)
                R_orig = R_orig_full
                mask_orig = ~np.isnan(R_orig)
                b_orig = np.array([self.difficulties[item] for item in orig_items])

        # Joint gradient descent
        old_thetas = theta_arr.copy()

        for iteration in range(max_iter):
            max_grad = 0.0

            # ── Gradients from NEW items (optimize both θ and b_new) ──
            P_new = expit(theta_arr[:, None] - b_new[None, :])  # (n_models, n_new)

            # Gradient for θ_j from new items (sum, not mean — standard Rasch JMLE)
            grad_theta = np.zeros(n_models)
            for j in range(n_models):
                valid = mask_new[j, :]
                if valid.any():
                    grad_theta[j] += np.sum(R_new[j, valid] - P_new[j, valid])

            # Gradient for b_i (new items) — sum gradient
            grad_b_new = np.zeros(n_new)
            for i in range(n_new):
                valid = mask_new[:, i]
                if valid.any():
                    grad_b_new[i] = np.sum(P_new[valid, i] - R_new[valid, i])

            # ── Gradients from ORIGINAL items (optimize θ only, b fixed) ──
            if R_orig is not None and mask_orig is not None:
                P_orig = expit(theta_arr[:, None] - b_orig[None, :])
                for j in range(n_models):
                    valid = mask_orig[j, :]
                    if valid.any():
                        grad_theta[j] += np.sum(R_orig[j, valid] - P_orig[j, valid])



            # Update parameters (gradient ascent on log-likelihood)
            theta_arr += lr * grad_theta
            b_new += lr * grad_b_new  # grad_b_new is the gradient of log-likelihood, so add

            max_grad = max(np.max(np.abs(grad_theta)), np.max(np.abs(grad_b_new)))
            if max_grad < tol:
                print(f"    Converged at iteration {iteration + 1} (max |grad| = {max_grad:.2e})")
                break

        # Report θ changes
        theta_changes = theta_arr - old_thetas
        print(f"\n  θ updates after anchored re-estimation:")
        for j, m in enumerate(model_names):
            print(f"    {m:>25s}: {old_thetas[j]:.3f} → {theta_arr[j]:.3f} (Δ = {theta_changes[j]:+.4f})")

        # Store updated θ values
        for j, m in enumerate(model_names):
            self.thetas[m] = float(theta_arr[j])

        # Store new item difficulties
        new_difficulties = {}
        for i, item_id in enumerate(new_item_ids):
            new_difficulties[item_id] = float(b_new[i])
            self.difficulties[item_id] = float(b_new[i])
            if item_id not in self.valid_items:
                self.valid_items.append(item_id)

        return new_difficulties


    def fit_anchored_2pl_em(
        self,
        new_response_matrix: pd.DataFrame,
        original_response_matrix: pd.DataFrame = None,
        max_epochs: int = 300,
        lr: float = 0.05,
        n_quadrature: int = 31,
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit 2PL EM model on cumulative matrix, holding original item parameters HELD FIXED.
        Uses PyTorch gradient masking to freeze anchor parameters during EM/MML calibration.

        Returns:
            Dict containing:
                "difficulties": Dict of new item difficulties
                "discriminations": Dict of new item discriminations
        """
        import torch
        import sys
        from unittest.mock import MagicMock
        sys.modules["tabpfn"] = MagicMock()
        sys.modules["pyro"] = MagicMock()
        sys.modules["pyro.distributions"] = MagicMock()
        from torch_measure.models import TwoPL
        from scipy.special import expit

        # Identify models present
        model_names = [m for m in new_response_matrix.index if m in self.thetas]
        n_models = len(model_names)
        new_item_ids = list(new_response_matrix.columns)
        n_new = len(new_item_ids)

        # Map original items
        orig_items = []
        if original_response_matrix is not None:
            orig_items = [c for c in original_response_matrix.columns
                          if c in self.difficulties and c in self.discriminations]

        # Build ordered item lists: original first, then new
        ordered_items = orig_items + new_item_ids
        n_orig = len(orig_items)
        n_total = len(ordered_items)

        print(f"  [Anchored 2PL EM] Calibrating cumulative items: {n_total} (anchor={n_orig}, new={n_new})")

        # Build cumulative response matrix
        cum_df = pd.DataFrame(np.nan, index=model_names, columns=ordered_items)
        if original_response_matrix is not None:
            for item in orig_items:
                for m in model_names:
                    if m in original_response_matrix.index:
                        cum_df.loc[m, item] = original_response_matrix.loc[m, item]

        for item in new_item_ids:
            for m in model_names:
                cum_df.loc[m, item] = new_response_matrix.loc[m, item]

        # Convert to PyTorch Tensor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        R_t = torch.tensor(cum_df.to_numpy(), dtype=torch.float32, device=device)
        mask = ~torch.isnan(R_t) & (R_t != -1)
        n_obs = mask.sum().to(dtype=R_t.dtype)

        # Initialize 2PL Model
        model = TwoPL(n_subjects=n_models, n_items=n_total, device=device)

        # Seed parameters from existing calibration
        with torch.no_grad():
            # Abilities
            for idx, m in enumerate(model_names):
                model.ability[idx] = self.thetas[m]
            # Anchor item parameters
            for idx, item in enumerate(orig_items):
                model.difficulty[idx] = self.difficulties[item]
                model.discrimination[idx] = self.discriminations[item]

        # ── Phase 1: EM Item Parameter Optimization (M-Step) ──
        theta_nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature)
        theta_nodes = torch.tensor(theta_nodes, dtype=torch.float32, device=device)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        weights = weights / weights.sum()
        log_weights = torch.log(weights)

        item_params = [p for name, p in model.named_parameters() if "ability" not in name]
        optimizer_item = torch.optim.Adam(item_params, lr=lr)
        loss_fn = lambda probs, targets: - (targets * torch.log(probs + 1e-7) + (1 - targets) * torch.log(1 - probs + 1e-7)).mean()

        # Optimization loop for items
        for epoch in range(max_epochs):
            optimizer_item.zero_grad()

            quad_log_terms = []
            for q in range(n_quadrature):
                with torch.no_grad():
                    model.ability.fill_(theta_nodes[q].item())

                # Compute probabilities
                logit = model.discrimination.unsqueeze(0) * (model.ability.unsqueeze(1) - model.difficulty.unsqueeze(0))
                probs = torch.sigmoid(logit)
                masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
                nll = loss_fn(masked_probs, R_t[mask])
                quad_log_terms.append(log_weights[q] - n_obs * nll)

            neg_marginal_ll = torch.logsumexp(torch.stack(quad_log_terms), dim=0)
            total_loss = -neg_marginal_ll

            total_loss.backward()

            # Mask gradients: ZERO OUT original anchor items gradients!
            if n_orig > 0:
                model.difficulty.grad[:n_orig] = 0.0
                model.discrimination.grad[:n_orig] = 0.0

            optimizer_item.step()

        # ── Phase 2: EM Ability Optimization ──
        for p in item_params:
            p.requires_grad_(False)
        model.ability.requires_grad_(True)

        # Re-initialize abilities
        with torch.no_grad():
            for idx, m in enumerate(model_names):
                model.ability[idx] = self.thetas[m]

        optimizer_ability = torch.optim.Adam([model.ability], lr=lr)

        for epoch in range(max_epochs):
            optimizer_ability.zero_grad()
            logit = model.discrimination.unsqueeze(0) * (model.ability.unsqueeze(1) - model.difficulty.unsqueeze(0))
            probs = torch.sigmoid(logit)
            masked_probs = probs[mask].clamp(1e-7, 1 - 1e-7)
            loss = loss_fn(masked_probs, R_t[mask])
            
            # Symmetrical regularization toward mean=0
            loss = loss + 0.01 * (model.ability.mean().abs())

            loss.backward()
            optimizer_ability.step()

        # Retrieve parameters back
        with torch.no_grad():
            theta_np = model.ability.detach().cpu().numpy()
            b_np = model.difficulty.detach().cpu().numpy()
            a_np = model.discrimination.detach().cpu().numpy()

        # Report ability shifts
        print(f"\n  θ updates after Anchored 2PL EM calibration:")
        for idx, m in enumerate(model_names):
            print(f"    {m:>25s}: {self.thetas[m]:.3f} → {theta_np[idx]:.3f} (Δ = {theta_np[idx] - self.thetas[m]:+.4f})")

        # Update model capacities in memory
        for idx, m in enumerate(model_names):
            self.thetas[m] = float(theta_np[idx])

        # Update item capacities
        new_diffs = {}
        new_discs = {}
        for idx, item in enumerate(new_item_ids):
            # Index of new item is n_orig + idx
            total_idx = n_orig + idx
            self.difficulties[item] = float(b_np[total_idx])
            self.discriminations[item] = float(a_np[total_idx])
            new_diffs[item] = float(b_np[total_idx])
            new_discs[item] = float(a_np[total_idx])
            if item not in self.valid_items:
                self.valid_items.append(item)

        return {
            "difficulties": new_diffs,
            "discriminations": new_discs
        }



