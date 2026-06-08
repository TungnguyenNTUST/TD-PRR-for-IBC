"""
Initial condition generation for Paper 01 BO runs.

Clusters the MOSFET/Core/Diode/Frequency/Inductance design space and
samples diverse initial seeds across cluster combinations.

Shared utilities (load_design_sheet, cluster_design, sample_group_combinations_n,
tuple_samples_to_flat_indices_n) have been extracted to core/clustering_utils.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.clustering_utils import (
    cluster_design,
    load_design_sheet,
    sample_group_combinations_n,
    tuple_samples_to_flat_indices_n,
)


if __name__ == "__main__":
    from core.paths import FILE_PATH as Excel_File

    df_device = pd.read_excel(Excel_File, sheet_name="Mid_Normalization")

    groups_mos, _ = cluster_design(df_device, feature_cols="Rds(on)", n_groups=2)
    for cid, idx_list in groups_mos.items():
        print(f"Group of MOS {cid} (size={len(idx_list)}): {idx_list}")

    groups_core, _ = cluster_design(df_device, feature_cols="Ae", n_groups=2)
    for cid, idx_list in groups_core.items():
        print(f"Group of core {cid} (size={len(idx_list)}): {idx_list}")

    groups_diode, _ = cluster_design(df_device, feature_cols="IF", n_groups=2)
    for cid, idx_list in groups_diode.items():
        print(f"Group of diode {cid} (size={len(idx_list)}): {idx_list}")

    df_input = load_design_sheet(
        Excel_File,
        sheet_name="Input",
        value_name_row_excel=3,
        data_start_row_excel=4,
    )
    df_input.columns = df_input.columns.astype(str).str.strip()

    groups_freq, _ = cluster_design(
        df_input,
        feature_cols="SwitchingFrequency",
        n_groups=2,
        index_col="Value_Name",
        method="kmeans",
        standardize=True,
    )
    for cid, idx_list in groups_freq.items():
        print(f"Group of frequency {cid} (size={len(idx_list)}): {idx_list}")

    groups_inductance, _ = cluster_design(
        df_input,
        feature_cols="Inductance_value",
        n_groups=2,
        index_col="Value_Name",
        method="kmeans",
        standardize=True,
    )
    for cid, idx_list in groups_inductance.items():
        print(f"Group of inductance {cid} (size={len(idx_list)}): {idx_list}")

    samples = sample_group_combinations_n(
        [groups_mos, groups_core, groups_diode, groups_freq, groups_inductance],
        s=5,
        mode="B",
        require_full=True,
        id_base=0,
        seed=2,
    )
    print(samples)

    indices_dic = tuple_samples_to_flat_indices_n(samples, dims=[12, 8, 6, 10, 10])
    indices_1based = [values for values in indices_dic.values()]
    indices = np.asarray(indices_1based)   # shape: (n_combos, s)

    with pd.ExcelWriter(Excel_File, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        for i in range(indices.shape[1]):
            test_index = indices[:, i]
            df = pd.DataFrame({"indices_1based": test_index})
            df.to_excel(writer, sheet_name=f"Mid_Z_Init_T{i + 1}", index=False)
