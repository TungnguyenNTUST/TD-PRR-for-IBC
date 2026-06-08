"""
Shared BO application helpers used by both LUT-mode and live-simulation-mode apps.

Previously copy-pasted across:
  - Bayesian_Optimization_APP.py
  - Bayesian_Optimization_APP_Paper01.py
  - Bayesian_Optimization_APP_Paper01_EI.py
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from discrete_bo_engine import DiscreteBO
from bo_gp_reporter import log_step


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_index_list(
    excel_path: str,
    sheet_name: str = "Mid_Z_Init_T2",
    column_name: str = "indices_1based",
) -> List[int]:
    """Read a 1-based index column from an Excel sheet and return as a list of ints."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=[column_name])
    return df[column_name].dropna().astype(int).tolist()


def load_input_and_index(
    file_path: str,
    input_sheet: str = "Mid_Input_encoding",
    index_sheet: str = "Mid_Input_Indexing",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the GP input matrix and the full design-option index matrix.

    Returns
    -------
    Input   : np.ndarray shape (D, N) -- transposed feature matrix for DiscreteBO
    Z_matrix: np.ndarray shape (N, n_vars) -- index matrix (1-based)
    """
    df_enc = pd.read_excel(file_path, sheet_name=input_sheet, header=0)
    df_idx = pd.read_excel(file_path, sheet_name=index_sheet, header=0)
    Input = df_enc.to_numpy(dtype=float).T        # (D, N)
    Z_matrix = df_idx.to_numpy(dtype=int)         # (N, n_vars)
    return Input, Z_matrix


def load_fixed_indices(
    file_path: str,
    sheet_name: str = "Mid_Z_Testing",
) -> np.ndarray:
    """Load the first column of a sheet as a 1-based integer index array."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    return df.iloc[:, 0].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# BO factory
# ---------------------------------------------------------------------------

def build_bo(
    Input: np.ndarray,
    *,
    sigma_f2: float = 1.0,
    sigma_n2: float = 0.05 ** 2,
    ei_acq: bool = False,
    xi: float = 0.01,
    kappa: float = 1.96,
    normalize_y: bool = True,
    random_state: int = 0,
    allow_hyperparam_optimization: bool = False,
    length_scale_bounds: Tuple[float, float] = (1e-9, 1e10),
    length_scales: Optional[Sequence[float]] = None,
) -> DiscreteBO:
    """
    Construct a DiscreteBO instance from the Input matrix.

    If length_scales is None, initialises all scales to 1.0 (uniform prior).
    Pass learned length-scales from Length_Scale_Learn.py to override.
    """
    D, N = Input.shape
    ls = list(length_scales) if length_scales is not None else [1.0] * D
    return DiscreteBO(
        Input=Input,
        length_scales=ls,
        sigma_f2=sigma_f2,
        sigma_n2=sigma_n2,
        ei_acq=ei_acq,
        xi=xi,
        kappa=kappa,
        normalize_y=normalize_y,
        random_state=random_state,
        allow_hyperparam_optimization=allow_hyperparam_optimization,
        length_scale_bounds=length_scale_bounds,
        hopt_every=1,
        hopt_warmup=1,
    )


def initialize_bo(
    bo: DiscreteBO,
    init_indices: Sequence[int],
    lut: Dict[int, float],
    available: Set[int],
) -> None:
    """Seed the GP with initial observations from the LUT."""
    for i in init_indices:
        if i not in available:
            raise RuntimeError(f"Init index {i} not found in evaluation table.")
    bo.initialize(init_indices, [lut[i] for i in init_indices])


# ---------------------------------------------------------------------------
# Workflow 1: fixed-iteration BO loop
# ---------------------------------------------------------------------------

def run_fixed_bo(
    bo: DiscreteBO,
    evaluate_fn,
    *,
    n_iters: int,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    eps: float,
    out_dir: Optional[str] = None,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
) -> pd.DataFrame:
    """
    Classic fixed-iteration BO loop.

    At each step: suggest -> evaluate -> tell -> log.
    """
    hist_rows = []
    for it in range(1, n_iters + 1):
        idxs, acqs = bo.suggest_next(top_k=1)
        idx = int(idxs[0])
        x = bo.Xc[idx - 1, :]
        y = float(evaluate_fn(idx, x))
        bo.tell(idx, y)

        Xp, _, _ = bo.compute_Xp_candidates(eps=eps, use_ucb=True, kappa=1.96)
        row = log_step(
            bo=bo,
            step=it,
            idx_next=idx,
            y_measured=y,
            Z_index=Z_matrix.T,
            var_names=var_names,
            eps=eps,
            Xp_size=len(Xp),
            save_plots_dir=out_dir,
            plot_secondary=plot_secondary,
            n_init=n_init,
        )
        row["acq"] = float(acqs[0])
        hist_rows.append(row)
        print(f"[BO] iter {it:03d}: pick={idx} y={y:.6g} acq={acqs[0]:.6g}")

    return pd.DataFrame(hist_rows)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarize_near_optimal(
    bo: DiscreteBO,
    eps: float,
    kappa_screen: float = 3.0,
    available: Optional[Set[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two DataFrames:
      - confirmed: observed points within eps of best
      - predicted: unobserved candidates the GP predicts within eps of best
    """
    y_best = float(np.max(bo._y_obs))
    target = y_best - eps

    confirmed = pd.DataFrame({
        "index_1based": np.array(bo._chosen_idx) + 1,
        "y": bo._y_obs,
    })
    confirmed = (
        confirmed[confirmed["y"] >= target]
        .sort_values("y", ascending=False)
        .reset_index(drop=True)
    )

    Xp, mu, std = bo.compute_Xp_candidates(
        eps=eps, use_ucb=True, kappa=kappa_screen, available_1based=available
    )
    predicted = pd.DataFrame({
        "index_1based": Xp,
        "mu": mu[np.array(Xp) - 1],
        "std": std[np.array(Xp) - 1],
        "target": target,
    }).sort_values("mu", ascending=False).reset_index(drop=True)

    return confirmed, predicted
