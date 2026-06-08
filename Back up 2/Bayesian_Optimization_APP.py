"""
LUT-based Bayesian Optimization application (pre-computed efficiency table).

Evaluation uses a lookup-table (Excel file with all efficiency results pre-computed).
For live-simulation evaluation, use Bayesian_Optimization_APP_Paper01.py.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from BO_evaluation import load_eval_lut, make_evaluate_fn_from_lut
from core.bo_app_common import (
    build_bo,
    initialize_bo,
    load_fixed_indices,
    load_input_and_index,
    read_index_list,
    run_fixed_bo,
    summarize_near_optimal,
)
from core.paths import FILE_PATH, OUT_DIR, RESULT_PATH

# =========================================================
# CONFIG  — paths are defined in core/paths.py
# =========================================================

INPUT_SHEET = "Mid_Input_encoding"
INDEX_SHEET = "Mid_Input_Indexing"
EVAL_SHEET  = "Test Case 01"
INIT_SHEET  = "Mid_Z_Testing"

EPS = 0.005


# =========================================================
# Near-optimal screening (LUT mode: availability-filtered)
# =========================================================

def run_near_optimal_screening(
    bo,
    evaluate_fn,
    *,
    eps: float,
    available: set,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    out_dir: Optional[str] = None,
    max_steps: int = 500,
    kappa_screen_init: float = 3.0,
    policy_kappa_init: float = 3.0,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
) -> tuple:
    """
    Screening loop with exponentially decaying kappa (LUT mode).

    In LUT mode, only indices present in `available` are eligible for Xp.
    Stops when Xp is empty (no candidate can plausibly match the best).
    """
    from BO_reporting import log_step

    history_rows = []
    kappa_screen = kappa_screen_init

    for step in range(1, max_steps + 1):
        kappa_screen = kappa_screen_init * np.exp(-3.0 * step / max_steps)
        policy_kappa = policy_kappa_init * np.exp(-3.0 * step / max_steps)

        Xp, _, _ = bo.compute_Xp_candidates(
            eps=eps,
            use_ucb=True,
            kappa=kappa_screen,
            available_1based=available,
        )

        if len(Xp) == 0:
            print(f"[STOP] Xp empty at step {step}")
            break

        idx_next = bo.suggest_next_within_indices(Xp, policy="ucb", kappa=policy_kappa)
        x = bo.Xc[idx_next - 1, :]
        y = float(evaluate_fn(idx_next, x))
        bo.tell(idx_next, y)

        row = log_step(
            bo=bo,
            step=step,
            idx_next=idx_next,
            y_measured=y,
            Z_index=Z_matrix.T,
            var_names=var_names,
            eps=eps,
            Xp_size=len(Xp),
            save_plots_dir=out_dir,
            plot_secondary=plot_secondary,
            n_init=n_init,
        )
        history_rows.append(row)
        print(
            f"[STEP {step:03d}] pick={idx_next} y={y:.6g} "
            f"best={row['best']:.6g} target={row['target']:.6g} "
            f"|Xp|={len(Xp)} kappa={kappa_screen:.4g}"
        )

    return pd.DataFrame(history_rows), kappa_screen


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    Input, Z_matrix = load_input_and_index(
        FILE_PATH,
        input_sheet=INPUT_SHEET,
        index_sheet=INDEX_SHEET,
    )
    lut, available, _real_df = load_eval_lut(RESULT_PATH, EVAL_SHEET)
    evaluate_fn = make_evaluate_fn_from_lut(lut)

    bo = build_bo(Input)
    init_indices = read_index_list(FILE_PATH, sheet_name=INIT_SHEET)
    initialize_bo(bo, init_indices, lut, available)

    var_names = ["Index", "MOS", "CORE", "DIO", "Freq", "Ind"]

    hist_screen, kappa_screen = run_near_optimal_screening(
        bo, evaluate_fn,
        eps=EPS,
        available=available,
        Z_matrix=Z_matrix,
        var_names=var_names,
        out_dir=OUT_DIR,
        max_steps=500,
        kappa_screen_init=3.0,
        policy_kappa_init=3.0,
        plot_secondary="ucb",
        n_init=len(init_indices),
    )
    hist_screen.to_csv(os.path.join(OUT_DIR, "screening_history.csv"), index=False)

    confirmed, predicted = summarize_near_optimal(
        bo, eps=EPS, kappa_screen=kappa_screen, available=available
    )
    print("\nConfirmed near-optimal (observed):")
    print(confirmed)
    print("\nPredicted near-optimal (unobserved):")
    print(predicted)


if __name__ == "__main__":
    main()
