"""
Live-simulation Bayesian Optimization application for Paper 01.

Evaluation calls efficiency_one_desin_option (SIMBA live simulation) for
design options not already in the observed cache.

For pre-computed LUT evaluation, use Bayesian_Optimization_APP.py.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from bo_lut_evaluator import load_eval_lut, load_index_eff, make_evaluate_fn_from_lut
from bo_gp_reporter import log_step
from discrete_bo_engine import DiscreteBO
from simba_efficiency_evaluator import efficiency_one_desin_option
from core.bo_app_common import (
    build_bo,
    initialize_bo,
    load_fixed_indices,
    load_input_and_index,
    read_index_list,
    run_fixed_bo,
    summarize_near_optimal,
)
from core.paths import FILE_PATH, OUT_DIR_PAPER01 as OUT_DIR, OUT_MU_STD

# =========================================================
# CONFIG — paths are defined in core/paths.py
# =========================================================

INPUT_SHEET = "Mid_Input_encoding"
INDEX_SHEET = "Mid_Input_Indexing"
INIT_SHEET  = "Mid_Z_Init_T1"

EPS = 0.000000001


# =========================================================
# Device lookup helpers
# =========================================================

def get_devices_from_x(
    x1: int,
    x2: int,
    x3: int,
    *,
    q1_map: Dict[int, str],
    q2_map: Dict[int, str],
    q3_map: Dict[int, str],
) -> Dict[str, str]:
    """Return device part-numbers from x1/x2/x3 1-based indices."""
    try:
        return {"MOS": q1_map[x1], "CORE": q2_map[x2], "DIO": q3_map[x3]}
    except KeyError as e:
        raise ValueError(f"Device index not found: {e}") from e


# =========================================================
# Near-optimal screening — UCB acquisition (standard)
# =========================================================

def run_near_optimal_screening(
    bo: DiscreteBO,
    *,
    eps: float,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    eff_obs: np.ndarray,
    indices_obs: np.ndarray,
    out_dir: Optional[str] = None,
    max_steps: int = 4000,
    kappa_screen: float = 2.0,
    policy_kappa: float = 2.0,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
    q1_map: Dict[int, str] = None,
    q2_map: Dict[int, str] = None,
    q3_map: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Screening loop using constant kappa (Paper 01 mode).

    Cache hit  : look up efficiency in eff_obs / indices_obs.
    Cache miss : call efficiency_one_desin_option (live SIMBA simulation).

    Stops when Xp is empty (no unobserved candidate can plausibly match the best).
    On stop, saves GP posterior (mu, std) to OUT_MU_STD.
    """
    history_rows = []

    for step in range(1, max_steps + 1):
        Xp, mu_all, std_all = bo.compute_Xp_candidates(
            eps=eps,
            use_ucb=True,
            kappa=kappa_screen,
            available_1based=None,
        )

        if len(Xp) == 0:
            print(f"[STOP] Xp empty at step {step}")
            df_mu_std = pd.DataFrame({"Mu": mu_all, "Std": std_all})
            df_mu_std.to_excel(OUT_MU_STD, index=False, engine="openpyxl")
            print(f"[saved] {OUT_MU_STD}")
            break

        idx_next = bo.suggest_next_within_indices(Xp, policy="ucb", kappa=policy_kappa)

        # --- evaluate: cache first, then live simulation ---
        matches = (indices_obs == idx_next)
        if matches.any():
            y = float(eff_obs[matches][0])
        else:
            d = get_devices_from_x(
                x1=int(Z_matrix[idx_next - 1, 1]),
                x2=int(Z_matrix[idx_next - 1, 2]),
                x3=int(Z_matrix[idx_next - 1, 3]),
                q1_map=q1_map,
                q2_map=q2_map,
                q3_map=q3_map,
            )
            y_list = efficiency_one_desin_option(
                MOS_PN=d["MOS"],
                DIO_PN=d["DIO"],
                Core_PN=d["CORE"],
                fsw_index=int(Z_matrix[idx_next - 1, 4]),
                ind_index=int(Z_matrix[idx_next - 1, 5]),
            )
            y = float(y_list[6])

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

    return pd.DataFrame(history_rows)


# =========================================================
# Near-optimal screening — EI acquisition (optional mode)
# =========================================================

def run_near_optimal_screening_with_EI(
    bo: DiscreteBO,
    *,
    eps: float,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    eff_obs: np.ndarray,
    indices_obs: np.ndarray,
    out_dir: Optional[str] = None,
    max_steps: int = 4000,
    kappa_screen: float = 2.0,
    xi: float = 0.01,
    plot_secondary: Optional[str] = "ei",
    n_init: int = 2,
    q1_map: Dict[int, str] = None,
    q2_map: Dict[int, str] = None,
    q3_map: Dict[int, str] = None,
) -> pd.DataFrame:
    """
    Screening loop using EI-based suggestion within the Xp candidate set.

    Xp is still computed with UCB (kappa_screen) to define the screening region.
    The point chosen within Xp uses the GP's EI acquisition.

    Bug-fix vs the original _EI.py: kappa_screen and policy_kappa are now
    explicit parameters instead of being used before assignment.
    """
    history_rows = []
    # temporarily switch bo to EI mode for suggestions
    original_ei_acq = bo.ei_acq
    original_xi = bo.xi
    bo.ei_acq = True
    bo.xi = float(xi)

    try:
        for step in range(1, max_steps + 1):
            Xp, mu_all, std_all = bo.compute_Xp_candidates(
                eps=eps,
                use_ucb=True,
                kappa=kappa_screen,
                available_1based=None,
            )

            if len(Xp) == 0:
                print(f"[STOP-EI] Xp empty at step {step}")
                df_mu_std = pd.DataFrame({"Mu": mu_all, "Std": std_all})
                df_mu_std.to_excel(OUT_MU_STD, index=False, engine="openpyxl")
                print(f"[saved] {OUT_MU_STD}")
                break

            # suggest within Xp using EI
            idx_next, _ = bo.suggest_next(top_k=1)
            idx_next = int(idx_next[0])
            # if suggest_next picks something outside Xp, fall back to UCB within Xp
            if idx_next not in set(Xp):
                idx_next = bo.suggest_next_within_indices(Xp, policy="ucb", kappa=kappa_screen)

            matches = (indices_obs == idx_next)
            if matches.any():
                y = float(eff_obs[matches][0])
            else:
                d = get_devices_from_x(
                    x1=int(Z_matrix[idx_next - 1, 1]),
                    x2=int(Z_matrix[idx_next - 1, 2]),
                    x3=int(Z_matrix[idx_next - 1, 3]),
                    q1_map=q1_map,
                    q2_map=q2_map,
                    q3_map=q3_map,
                )
                y_list = efficiency_one_desin_option(
                    MOS_PN=d["MOS"],
                    DIO_PN=d["DIO"],
                    Core_PN=d["CORE"],
                    fsw_index=int(Z_matrix[idx_next - 1, 4]),
                    ind_index=int(Z_matrix[idx_next - 1, 5]),
                )
                y = float(y_list[6])

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
                f"[STEP-EI {step:03d}] pick={idx_next} y={y:.6g} "
                f"best={row['best']:.6g} target={row['target']:.6g} |Xp|={len(Xp)}"
            )
    finally:
        # restore bo acquisition settings
        bo.ei_acq = original_ei_acq
        bo.xi = original_xi

    return pd.DataFrame(history_rows)


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

    # Load initial observations (from one of the Mid_Z_Init_T* sheets)
    init_indices = read_index_list(FILE_PATH, sheet_name=INIT_SHEET, column_name="indices_1based")
    indices_obs, eff_obs, _df = load_index_eff(FILE_PATH, sheet_name=INIT_SHEET)

    # Build LUT for seeding the GP (initial observations must be in observed cache)
    lut = dict(zip(indices_obs, eff_obs))
    available = set(lut.keys())

    bo = build_bo(Input, kappa=2.0)
    initialize_bo(bo, init_indices, lut, available)

    var_names = ["Index", "MOS", "CORE", "DIO", "Freq", "Ind"]

    # --- Device index maps (edit to match your Input sheet values) ---
    # These map 1-based x1/x2/x3 indices to part-number strings for SIMBA
    # Example maps — replace with actual values from your Input sheet:
    q1_map: Dict[int, str] = {}   # MOSFET: {1: "C3M0040120K1", 2: ..., ...}
    q2_map: Dict[int, str] = {}   # Core:   {1: "KAM184-075A", ...}
    q3_map: Dict[int, str] = {}   # Diode:  {1: "C4D20120D", ...}

    hist = run_near_optimal_screening(
        bo,
        eps=EPS,
        Z_matrix=Z_matrix,
        var_names=var_names,
        eff_obs=eff_obs,
        indices_obs=indices_obs,
        out_dir=OUT_DIR,
        max_steps=4000,
        kappa_screen=2.0,
        policy_kappa=2.0,
        plot_secondary="ucb",
        n_init=len(init_indices),
        q1_map=q1_map,
        q2_map=q2_map,
        q3_map=q3_map,
    )
    hist.to_csv(os.path.join(OUT_DIR, "screening_history.csv"), index=False)

    confirmed, predicted = summarize_near_optimal(bo, eps=EPS, kappa_screen=2.0)
    print("\nConfirmed near-optimal (observed):")
    print(confirmed)
    print("\nPredicted near-optimal (unobserved):")
    print(predicted)


if __name__ == "__main__":
    main()
