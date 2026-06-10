"""
Generate initial seed sheets T6–T10 and evaluate efficiencies.

Strategy per index:
  1. Already in an existing T1-T5 sheet → copy all columns (eff_1..7 + device info)
  2. Only in the Observed sheet → copy eff_7; leave eff_1..6 as NaN
  3. Not in any cache → run live SIMBA simulation; save eff_7 to Observed sheet
"""
from __future__ import annotations

import sys
import os

# Allow importing project modules when run from AutomationDesign dir
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from core.clustering_utils import (
    cluster_design,
    load_design_sheet,
    sample_group_combinations_n,
    tuple_samples_to_flat_indices_n,
)
from core.paths import FILE_PATH
from simba_efficiency_evaluator import efficiency_one_desin_option

# ── Constants ─────────────────────────────────────────────────────────────────
NEW_SEED   = 100       # RNG seed for T6-T10  (T1-T5 used seed=2)
N_SAMPLES  = 5        # number of new seeds to generate
FIRST_NEW  = 6        # first new T sheet number
DIMS       = [12, 8, 6, 10, 10]   # must match initial_seed_generator.py

T_COLUMNS = [
    "indices_1based",
    "MOS", "CORE", "DIO",
    "fsw_index", "ind_index",
    "eff_1", "eff_2", "eff_3", "eff_4", "eff_5", "eff_6", "eff_7",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_existing_T_cache(excel_path: str) -> dict[int, dict]:
    """
    Load all T1-T5 rows into a dict keyed by indices_1based.
    Each value is a dict with all T_COLUMNS (except indices_1based).
    """
    cache: dict[int, dict] = {}
    for t in range(1, 6):
        sheet = f"Mid_Z_Init_T{t}"
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet)
        except Exception:
            continue
        for _, row in df.iterrows():
            idx = int(row["indices_1based"])
            if idx not in cache:
                cache[idx] = {c: row.get(c, float("nan")) for c in T_COLUMNS if c != "indices_1based"}
    print(f"[CACHE] Loaded {len(cache)} unique entries from T1-T5 sheets.")
    return cache


def _build_observed_cache(excel_path: str) -> dict[int, float]:
    """Load Observed sheet → {index: eff_7}."""
    df = pd.read_excel(excel_path, sheet_name="Observed")
    return dict(zip(df["indices_1based"].astype(int), df["eff_7"].astype(float)))


def _load_device_maps(excel_path: str):
    """Return q1_map, q2_map, q3_map and the Z_matrix."""
    df_z = pd.read_excel(excel_path, sheet_name="Mid_Input_Indexing")
    Z_matrix = df_z.to_numpy(dtype=int)  # (N, 6): Order, x1..x5

    df_input = pd.read_excel(excel_path, sheet_name="Input")
    df_input = df_input.dropna(subset=["Order"])
    df_input = df_input[~df_input["Order"].isin(["Input_Type", "Value_Name"])]
    df_input["Order"] = pd.to_numeric(df_input["Order"], errors="coerce")
    df_input = df_input.dropna(subset=["Order"])

    q1_map = dict(zip(df_input["Order"].astype(int), df_input["x1"]))
    q2_map = dict(zip(df_input["Order"].astype(int), df_input["x2"]))
    q3_map = dict(zip(df_input["Order"].astype(int), df_input["x3"]))
    return q1_map, q2_map, q3_map, Z_matrix


def _index_to_device_row(idx_1based: int, Z_matrix: np.ndarray, q1_map, q2_map, q3_map) -> dict:
    """Return device info dict for a flat 1-based index."""
    row = Z_matrix[idx_1based - 1]   # row = [Order, x1, x2, x3, x4, x5]
    x1, x2, x3, x4, x5 = int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
    return {
        "MOS":       q1_map[x1],
        "CORE":      q2_map[x2],
        "DIO":       q3_map[x3],
        "fsw_index": x4,
        "ind_index": x5,
    }


def _append_to_observed(excel_path: str, new_rows: list[tuple[int, float]]) -> None:
    """Append (index, eff_7) rows to the Observed sheet."""
    if not new_rows:
        return
    df_new = pd.DataFrame(new_rows, columns=["indices_1based", "eff_7"])
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        wb = writer.book
        ws = wb["Observed"]
        start_row = ws.max_row
        df_new.to_excel(writer, sheet_name="Observed", index=False, header=False, startrow=start_row)
    print(f"[OBSERVED] Appended {len(new_rows)} new rows to Observed sheet.")


# ── Index generation ──────────────────────────────────────────────────────────

def generate_new_indices(excel_path: str) -> np.ndarray:
    """Return shape-(n_combos, N_SAMPLES) array of 1-based indices for T6-T10."""
    df_device = pd.read_excel(excel_path, sheet_name="Mid_Normalization")
    groups_mos,    _ = cluster_design(df_device, feature_cols="Rds(on)", n_groups=2)
    groups_core,   _ = cluster_design(df_device, feature_cols="Ae",      n_groups=2)
    groups_diode,  _ = cluster_design(df_device, feature_cols="IF",      n_groups=2)

    df_input = load_design_sheet(excel_path, sheet_name="Input",
                                 value_name_row_excel=3, data_start_row_excel=4)
    df_input.columns = df_input.columns.astype(str).str.strip()

    groups_freq, _ = cluster_design(
        df_input, feature_cols="SwitchingFrequency", n_groups=2,
        index_col="Value_Name", method="kmeans", standardize=True,
    )
    groups_inductance, _ = cluster_design(
        df_input, feature_cols="Inductance_value", n_groups=2,
        index_col="Value_Name", method="kmeans", standardize=True,
    )

    samples = sample_group_combinations_n(
        [groups_mos, groups_core, groups_diode, groups_freq, groups_inductance],
        s=N_SAMPLES,
        mode="B",
        require_full=True,
        id_base=0,
        seed=NEW_SEED,
    )
    indices_dic = tuple_samples_to_flat_indices_n(samples, dims=DIMS)
    return np.asarray([v for v in indices_dic.values()])   # (n_combos, N_SAMPLES)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    excel_path = FILE_PATH
    print(f"[INFO] Excel file: {excel_path}")

    # Load caches
    t_cache  = _build_existing_T_cache(excel_path)
    obs_cache = _build_observed_cache(excel_path)
    q1_map, q2_map, q3_map, Z_matrix = _load_device_maps(excel_path)

    # Generate indices for T6-T10
    print("[INFO] Generating indices for T6-T10 ...")
    indices = generate_new_indices(excel_path)   # (32, 5)
    n_combos, n_seeds = indices.shape
    print(f"[INFO] Got {n_combos} cluster-combinations × {n_seeds} seeds")

    # Build T6-T10 sheets
    new_obs_rows: list[tuple[int, float]] = []

    for seed_col in range(n_seeds):
        t_num   = FIRST_NEW + seed_col
        sheet   = f"Mid_Z_Init_T{t_num}"
        col_idx = indices[:, seed_col]
        print(f"\n{'='*60}")
        print(f"[T{t_num}] Building {sheet} ({len(col_idx)} rows) ...")

        rows = []
        for idx in col_idx:
            idx = int(idx)
            dev = _index_to_device_row(idx, Z_matrix, q1_map, q2_map, q3_map)

            # --- Priority 1: in existing T1-T5 cache ---
            if idx in t_cache:
                src = t_cache[idx]
                row = {"indices_1based": idx, **dev}
                for k in ["eff_1","eff_2","eff_3","eff_4","eff_5","eff_6","eff_7"]:
                    row[k] = src.get(k, float("nan"))
                print(f"  [{idx}] copied from T1-T5 cache (eff_7={row['eff_7']:.5f})")

            # --- Priority 2: in Observed cache ---
            elif idx in obs_cache:
                eff7 = obs_cache[idx]
                row = {
                    "indices_1based": idx, **dev,
                    "eff_1": float("nan"), "eff_2": float("nan"), "eff_3": float("nan"),
                    "eff_4": float("nan"), "eff_5": float("nan"), "eff_6": float("nan"),
                    "eff_7": eff7,
                }
                print(f"  [{idx}] copied from Observed (eff_7={eff7:.5f})")

            # --- Priority 3: run live simulation ---
            else:
                print(f"  [{idx}] running simulation: MOS={dev['MOS']}, CORE={dev['CORE']}, DIO={dev['DIO']}, fsw={dev['fsw_index']}, ind={dev['ind_index']}")
                eff_list = efficiency_one_desin_option(
                    MOS_PN=dev["MOS"],
                    DIO_PN=dev["DIO"],
                    Core_PN=dev["CORE"],
                    fsw_index=dev["fsw_index"],
                    ind_index=dev["ind_index"],
                )
                if len(eff_list) != 7:
                    raise ValueError(f"Expected 7 efficiency values, got {len(eff_list)} for idx={idx}")
                row = {
                    "indices_1based": idx, **dev,
                    "eff_1": eff_list[0], "eff_2": eff_list[1], "eff_3": eff_list[2],
                    "eff_4": eff_list[3], "eff_5": eff_list[4], "eff_6": eff_list[5],
                    "eff_7": eff_list[6],
                }
                # Queue for Observed update
                new_obs_rows.append((idx, float(eff_list[6])))
                obs_cache[idx] = float(eff_list[6])   # keep local cache updated
                print(f"  [{idx}] simulated eff_7={eff_list[6]:.5f}")

            rows.append({c: row[c] for c in T_COLUMNS})

        df_sheet = pd.DataFrame(rows, columns=T_COLUMNS)

        # Write sheet to Excel
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_sheet.to_excel(writer, sheet_name=sheet, index=False)
        print(f"[SAVED] Sheet '{sheet}' written to Excel.")

    # Save all new observations to Observed at the end
    if new_obs_rows:
        _append_to_observed(excel_path, new_obs_rows)
        print(f"\n[DONE] {len(new_obs_rows)} new simulations saved to Observed sheet.")
    else:
        print("\n[DONE] No new simulations needed — all indices were in cache.")

    print("\n[SUMMARY]")
    print(f"  Created sheets: Mid_Z_Init_T6 through Mid_Z_Init_T{FIRST_NEW + N_SAMPLES - 1}")
    print(f"  Total new simulations run: {len(new_obs_rows)}")


if __name__ == "__main__":
    main()
