"""
Live-simulation Bayesian Optimization application for Paper 01.

Evaluation calls efficiency_one_desin_option (SIMBA live simulation) for
design options not already in the observed cache.

For pre-computed LUT evaluation, use Bayesian_Optimization_APP.py.
"""
from __future__ import annotations

import os
import pickle
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
INIT_SHEET  = "Mid_Z_Init_T5"

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


def append_new_observations_to_excel(file_path: str, sheet_name: str, new_rows: list[tuple[int, float]]):
    """Append new simulation results to the Excel sheet to save them as cache."""
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows, columns=["indices_1based", "eff_7"])
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        workbook = writer.book
        if sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            start_row = sheet.max_row
        else:
            start_row = 0
        df_new.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            header=(start_row == 0),
            startrow=start_row
        )
    print(f"[saved] Appended {len(new_rows)} new observations to sheet '{sheet_name}' in {file_path}")


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
    kappa_screen: float = 1.96,
    policy_kappa: float = 1.96,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
    q1_map: Dict[int, str] = None,
    q2_map: Dict[int, str] = None,
    q3_map: Dict[int, str] = None,
    history_rows: Optional[list] = None,
    pkl_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Screening loop using constant kappa (Paper 01 mode).

    Cache hit  : look up efficiency in eff_obs / indices_obs.
    Cache miss : call efficiency_one_desin_option (live SIMBA simulation).

    Stops when Xp is empty (no unobserved candidate can plausibly match the best).
    On stop, saves GP posterior (mu, std) to OUT_MU_STD.
    """
    if history_rows is None:
        history_rows = []
    new_obs = []
    start_step = len(history_rows) + 1

    try:
        for step in range(start_step, max_steps + 1):
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

            idx_next = bo.suggest_next_within_indices(
                Xp, 
                policy="ucb", 
                kappa=policy_kappa, 
                precomputed_mu_std=(mu_all, std_all)
            )

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
                new_obs.append((int(idx_next), y))

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

            # Save checkpoint after each step
            if pkl_path:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump((bo, history_rows), f)
                except Exception as e:
                    print(f"[WARNING] Could not save BO checkpoint: {e}")
    finally:
        if new_obs:
            append_new_observations_to_excel(FILE_PATH, "Observed", new_obs)

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
    kappa_screen: float = 1.96,
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
    new_obs = []
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
                new_obs.append((int(idx_next), y))

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
        if new_obs:
            append_new_observations_to_excel(FILE_PATH, "Observed", new_obs)

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
    indices_obs, eff_obs, _df = load_index_eff(FILE_PATH, sheet_name="Observed")

    # Build LUT for seeding the GP (initial observations must be in observed cache)
    lut = dict(zip(indices_obs, eff_obs))
    available = set(lut.keys())

    # Check for existing checkpoint to resume
    pkl_path = os.path.join(OUT_DIR, "bo_state.pkl")
    bo_loaded = False
    history_rows = None

    if os.path.exists(pkl_path):
        print(f"[RESUME] Found existing BO checkpoint at {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                bo, history_rows = pickle.load(f)
            print(f"[RESUME] Successfully loaded checkpoint. Resuming from step {len(history_rows) + 1}.")
            bo_loaded = True
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}. Starting fresh.")

    if not bo_loaded:
        bo = build_bo(Input, kappa=1.96)
        initialize_bo(bo, init_indices, lut, available)
        history_rows = []

    var_names = ["Index", "MOS", "CORE", "DIO", "Freq", "Ind"]

    # Load device index maps from the Input sheet
    df_input = pd.read_excel(FILE_PATH, sheet_name="Input")
    df_input = df_input.dropna(subset=["Order"])
    q1_map = dict(zip(df_input["Order"], df_input["x1"]))
    q2_map = dict(zip(df_input["Order"], df_input["x2"]))
    q3_map = dict(zip(df_input["Order"], df_input["x3"]))

    hist = run_near_optimal_screening(
        bo,
        eps=EPS,
        Z_matrix=Z_matrix,
        var_names=var_names,
        eff_obs=eff_obs,
        indices_obs=indices_obs,
        out_dir=None,
        max_steps=4000,
        kappa_screen=1.96,
        policy_kappa=1.96,
        plot_secondary="ucb",
        n_init=len(init_indices),
        q1_map=q1_map,
        q2_map=q2_map,
        q3_map=q3_map,
        history_rows=history_rows,
        pkl_path=pkl_path,
    )
    hist.to_csv(os.path.join(OUT_DIR, "screening_history.csv"), index=False)

    # Clean up checkpoint since it finished successfully
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
            print(f"[CLEANUP] Deleted checkpoint {pkl_path} because the optimization finished successfully.")
        except Exception as e:
            print(f"[WARNING] Could not delete checkpoint: {e}")

    confirmed, predicted = summarize_near_optimal(bo, eps=EPS, kappa_screen=1.96)
    print("\nConfirmed near-optimal (observed):")
    print(confirmed)
    print("\nPredicted near-optimal (unobserved):")
    print(predicted)


if __name__ == "__main__":
    main()
