"""
Dictionary-encoding BO runner for Paper 01 (DIC variant).

Identical screening loop to run_bo_simba.py, but uses the
dictionary-encoded input matrix (Mid_Input_encoding_DIC.npy) built by
build_input_matrix_dic.py instead of the raw normalized-feature matrix.

Key differences vs run_bo_simba.py:
  - Input matrix  : loaded from CACHE_INPUT_DIC (.npy, D=K1+K2+K3+2)
  - Z_matrix      : loaded from CACHE_INDEX_DIC (.npy)
  - Observed sheet: read/written to UserProvidedDataFile_DIC.xlsx only
  - Output dir    : Results/Paper01_DIC/  (separate from raw-features run)
  - Checkpoint    : bo_state_dic.pkl

The original UserProvidedDataFile.xlsx is never touched.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from bo_lut_evaluator import load_index_eff
from bo_gp_reporter import log_step
from discrete_bo_engine import DiscreteBO
from simba_efficiency_evaluator import efficiency_one_desin_option
from core.bo_app_common import (
    build_bo,
    initialize_bo,
    read_index_list,
    summarize_near_optimal,
)
from core.paths import (
    FILE_PATH_DIC,
    CACHE_INPUT_DIC,
    CACHE_INDEX_DIC,
    OUT_DIR_PAPER01_DIC as OUT_DIR,
    OUT_MU_STD_DIC as OUT_MU_STD,
)

# =========================================================
# CONFIG
# =========================================================

INIT_SHEET = "Mid_Z_Init_T5"   # change to T1-T10 as needed

EPS = 0.000000001


# =========================================================
# Device lookup helpers  (identical to run_bo_simba.py)
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
    try:
        return {"MOS": q1_map[x1], "CORE": q2_map[x2], "DIO": q3_map[x3]}
    except KeyError as e:
        raise ValueError(f"Device index not found: {e}") from e


def append_new_observations_to_excel(new_rows: list[tuple[int, float]]) -> None:
    """Append new simulation results to Observed in the DIC file."""
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows, columns=["indices_1based", "eff_7"])
    with pd.ExcelWriter(FILE_PATH_DIC, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        workbook = writer.book
        sheet = workbook["Observed"]
        start_row = sheet.max_row
        df_new.to_excel(
            writer,
            sheet_name="Observed",
            index=False,
            header=(start_row == 0),
            startrow=start_row,
        )
    print(f"[saved] Appended {len(new_rows)} new observations to DIC Observed sheet.")


# =========================================================
# Near-optimal screening loop  (identical logic to run_bo_simba.py)
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
    Near-optimal screening with UCB acquisition.

    Cache hit  : look up efficiency in eff_obs / indices_obs.
    Cache miss : call efficiency_one_desin_option (live SIMBA simulation),
                 then append result to DIC Observed sheet.
    """
    if history_rows is None:
        history_rows = []
    new_obs: list[tuple[int, float]] = []
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
                precomputed_mu_std=(mu_all, std_all),
            )

            # Evaluate: cache first, then live simulation
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

            if pkl_path:
                try:
                    with open(pkl_path, "wb") as f:
                        pickle.dump((bo, history_rows), f)
                except Exception as e:
                    print(f"[WARNING] Could not save BO checkpoint: {e}")
    finally:
        append_new_observations_to_excel(new_obs)

    return pd.DataFrame(history_rows)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Input matrix from pre-built DIC .npy cache (D x N)
    if not os.path.exists(CACHE_INPUT_DIC):
        raise FileNotFoundError(
            f"DIC Input cache not found: {CACHE_INPUT_DIC}\n"
            "Run build_input_matrix_dic.py first."
        )
    if not os.path.exists(CACHE_INDEX_DIC):
        raise FileNotFoundError(
            f"DIC Z_index cache not found: {CACHE_INDEX_DIC}\n"
            "Run build_input_matrix_dic.py first."
        )

    print(f"[INFO] Loading DIC Input matrix from {CACHE_INPUT_DIC} ...")
    Input = np.load(CACHE_INPUT_DIC)       # (D, N)
    print(f"[INFO] Input shape: {Input.shape}  (D={Input.shape[0]}, N={Input.shape[1]})")

    print(f"[INFO] Loading Z_matrix from {CACHE_INDEX_DIC} ...")
    Z_matrix = np.load(CACHE_INDEX_DIC)    # (N, 6): Order, x1..x5
    print(f"[INFO] Z_matrix shape: {Z_matrix.shape}")

    # Initial seed indices and observed cache — both from DIC file
    init_indices = read_index_list(FILE_PATH_DIC, sheet_name=INIT_SHEET, column_name="indices_1based")
    indices_obs, eff_obs, _df = load_index_eff(FILE_PATH_DIC, sheet_name="Observed")

    lut = dict(zip(indices_obs, eff_obs))
    available = set(lut.keys())

    # Checkpoint resume
    pkl_path = os.path.join(OUT_DIR, "bo_state_dic.pkl")
    bo_loaded = False
    history_rows = None

    if os.path.exists(pkl_path):
        print(f"[RESUME] Found checkpoint at {pkl_path}")
        try:
            with open(pkl_path, "rb") as f:
                bo, history_rows = pickle.load(f)
            print(f"[RESUME] Resuming from step {len(history_rows) + 1}.")
            bo_loaded = True
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}. Starting fresh.")

    if not bo_loaded:
        bo = build_bo(Input, kappa=1.96)
        initialize_bo(bo, init_indices, lut, available)
        history_rows = []

    var_names = ["Index", "MOS", "CORE", "DIO", "Freq", "Ind"]

    # Device maps for SIMBA simulation (read from DIC file's Input sheet)
    df_input = pd.read_excel(FILE_PATH_DIC, sheet_name="Input")
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
    hist.to_csv(os.path.join(OUT_DIR, "screening_history_dic.csv"), index=False)

    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
            print(f"[CLEANUP] Deleted checkpoint {pkl_path}.")
        except Exception as e:
            print(f"[WARNING] Could not delete checkpoint: {e}")

    confirmed, predicted = summarize_near_optimal(bo, eps=EPS, kappa_screen=1.96)
    print("\nConfirmed near-optimal (observed):")
    print(confirmed)
    print("\nPredicted near-optimal (unobserved):")
    print(predicted)


if __name__ == "__main__":
    main()
