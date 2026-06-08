from __future__ import annotations
from typing import List, Dict, Union, Literal, Tuple, Optional,Set,Any,Sequence
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel

def load_design_sheet(
    xlsx_path: str,
    sheet_name: str,
    *,
    value_name_row_excel: int = 3,  # row containing final column names (Value_Name row)
    data_start_row_excel: int = 4,  # first data row (Order=1 row)
) -> pd.DataFrame:
    """
    Robust loader for your layout:
      Row 1: Order / x1 / x2 / x3 ...
      Row 2: Input_Type ...
      Row 3: Value_Name row (actual desired column names)
      Row 4+: data
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    value_row = value_name_row_excel - 1
    data_row  = data_start_row_excel - 1

    # Use Value_Name row as column names
    cols = raw.iloc[value_row].tolist()
    df = raw.iloc[data_row:].copy()
    df.columns = cols

    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Clean: ensure Order is numeric if present
    if "Order" in df.columns:
        df["Order"] = pd.to_numeric(df["Order"], errors="coerce")

    return df

#ClusterMethodDefine
ClusterMethod = Literal["kmeans", "agglomerative"]

def cluster_design(
    df: pd.DataFrame,
    feature_cols: Union[str, List[str]],
    n_groups: int,
    *,
    index_col: Optional[str] = "Index",
    method: ClusterMethod = "kmeans",
    standardize: bool = True,
    random_state: int = 0,
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Cluster Design parameters into N groups based on selected feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing Design Parameter rows.
    feature_cols : str or list[str]
        Feature column name(s) used for clustering. Example: "Rds(on)" or ["Rds(on)", "Coss"].
    n_groups : int
        Number of clusters (groups).
    index_col : str | None
        Column containing MOSFET IDs/indices to return (e.g., Excel 'Index').
        If None or missing, uses df.index.
    method : {"kmeans", "agglomerative"}
        Clustering algorithm.
    standardize : bool
        If True, standardize features before clustering (recommended).
    random_state : int
        Random seed (used by kmeans).

    Returns
    -------
    groups : dict[int, list[int]]
        Mapping cluster_id -> list of MOSFET indices (from index_col or df.index).
    labels : np.ndarray
        Cluster label for each row in the filtered dataframe order.

    Notes
    -----
    - Rows with missing values in selected features are dropped.
    - If you use many features with different units (Rds, Coss, etc.), standardize=True is strongly recommended.
    """

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    if n_groups < 1:
        raise ValueError("n_groups must be >= 1.")
    if not all(col in df.columns for col in feature_cols):
        missing = [c for c in feature_cols if c not in df.columns]
        raise KeyError(f"Missing feature columns in df: {missing}")

    # Extract feature matrix, drop rows with NaNs in features
    work = df.copy()
    work = work.dropna(subset=feature_cols)

    if len(work) < n_groups:
        raise ValueError(f"Not enough rows ({len(work)}) to form {n_groups} groups after dropping NaNs.")

    X = work[feature_cols].to_numpy(dtype=float)

    # Standardize if requested (recommended when mixing units)
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    # Fit clustering
    if method == "kmeans":
        model = KMeans(n_clusters=n_groups, n_init=20, random_state=random_state)
        labels = model.fit_predict(X_fit)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_groups, linkage="ward" if standardize else "ward")
        labels = model.fit_predict(X_fit)
    else:
        raise ValueError("method must be 'kmeans' or 'agglomerative'.")

    # Decide which indices to return
    # if index_col is not None and index_col in work.columns:
    #     ids = work[index_col].to_numpy()
    # else:
    #     ids = work.index.to_numpy()
    if index_col not in work.columns:
        raise KeyError(f"index_col='{index_col}' not found. Columns: {list(work.columns)}")
    ids = work[index_col].to_numpy()

    # Build groups: cluster_id -> list of ids
    groups: Dict[int, List[int]] = {}
    for cid in range(n_groups):
        groups[cid] = [int(v) for v in ids[labels == cid]]

    return groups, labels

UniqMode = Literal["A", "B"]

def sample_group_combinations_n(
    groups_list: Sequence[Dict[int, List[Any]]],
    *,
    s: int = 2,
    mode: UniqMode = "A",
    seed: int = 0,
    require_full: bool = True,
    id_base: Literal[0, 1] = 1,
) -> Dict[Tuple[int, ...], List[Tuple[Any, ...]]]:
    """
    General sampler for N variables.

    Parameters
    ----------
    groups_list:
        A sequence of dicts. Each dict maps cluster_id -> list of candidate IDs/values
        for ONE variable.
        Example (5 vars): [groups_mos, groups_freq, groups_ind, groups_cap, groups_temp]

    s:
        Number of samples to draw per cluster-combination.

    mode:
        "A": no repeated full tuple within a cluster-combination.
        "B": additionally enforce that ALL dimensions differ from the previous tuple.

    seed:
        RNG seed.

    require_full:
        If True, raise if can't draw all s samples for some cluster-combination.
        If False, keep what was possible.

    id_base:
        If 1, shift all int / numpy-int IDs by +1 (convert 0-based -> 1-based).
        If 0, keep as-is.

    Returns
    -------
    out:
        Dict keyed by cluster-combo tuple (c1, c2, ..., cN),
        values are lists of sampled value tuples (v1, v2, ..., vN).
    """
    if s < 1:
        raise ValueError("s must be >= 1")
    if not groups_list:
        raise ValueError("groups_list must contain at least 1 variable group dict")

    rng = np.random.default_rng(seed)

    # Sorted cluster keys per variable
    cluster_keys_per_var: List[List[int]] = [sorted(g.keys()) for g in groups_list]
    n_vars = len(groups_list)

    def to_base(v: Any) -> Any:
        if isinstance(v, (int, np.integer)):
            iv = int(v)
            return iv if id_base == 0 else iv + 1
        return v

    def draw_one(cluster_combo: Tuple[int, ...]) -> Tuple[Any, ...]:
        # pick one candidate per variable according to its cluster id
        picks = []
        for j, cj in enumerate(cluster_combo):
            cand = rng.choice(groups_list[j][cj])
            picks.append(to_base(cand))
        return tuple(picks)

    # Cartesian product over cluster ids across variables (no itertools.product needed, but it's clean)
    import itertools
    out: Dict[Tuple[int, ...], List[Tuple[Any, ...]]] = {}

    for cluster_combo in itertools.product(*cluster_keys_per_var):
        picked: List[Tuple[Any, ...]] = []
        max_tries = 5000

        for _ in range(s):
            if not picked:
                picked.append(draw_one(cluster_combo))
                continue

            prev = picked[-1]
            ok = False
            tries = 0

            while tries < max_tries and not ok:
                cand = draw_one(cluster_combo)

                if mode == "A":
                    ok = (cand not in picked)

                elif mode == "B":
                    # all dims differ vs previous + not duplicate overall
                    all_dims_differ = all(cand[j] != prev[j] for j in range(n_vars))
                    ok = all_dims_differ and (cand not in picked)

                else:
                    raise ValueError("mode must be 'A' or 'B'")

                tries += 1

            if not ok:
                msg = f"Cannot draw sample #{len(picked)+1} for combo {cluster_combo} under mode={mode}."
                if require_full:
                    raise RuntimeError(msg)
                break

            picked.append(cand)

        out[tuple(cluster_combo)] = picked

    return out

from typing import Dict, Tuple, List, Sequence


def tuple_samples_to_flat_indices_n(
    samples: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    *,
    dims: Sequence[int],
) -> Dict[Tuple[int, ...], List[int]]:
    """
    Convert tuple-based samples (i1, i2, ..., iN) into flattened indices.

    Assumptions
    -----------
    - All indices are 1-based
    - dims = [D1, D2, ..., DN] are sizes of each dimension

    Mapping (row-major)
    -------------------
    idx = sum((ik-1)*prod(dims[k+1:])) + 1
    """

    n = len(dims)

    # Precompute strides
    # Example: dims=[D1,D2,D3] → strides=[D2*D3, D3, 1]
    strides = [1] * n
    acc = 1

    for i in range(n - 1, -1, -1):
        strides[i] = acc
        acc *= dims[i]

    flat_indices: Dict[Tuple[int, ...], List[int]] = {}

    for combo_key, tuples in samples.items():

        idx_list: List[int] = []

        for t in tuples:

            if len(t) != n:
                raise ValueError(f"Tuple {t} does not match dims length {n}")

            idx = 1  # 1-based

            for k in range(n):
                idx += (t[k] - 1) * strides[k]

            idx_list.append(idx)

        flat_indices[combo_key] = idx_list

    return flat_indices

# ------------------- Example usage -------------------
if __name__ == "__main__":

    # --- Example Usage ---
    # Replace 'your_file.xlsx' with the actual name of your file
    Excel_File = r'C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx'
    # results_file = r'C:\Users\Tungtan\PycharmProjects\AutomationDesign\Data\20260108_32 efficiency_results.xlsx'

    # Suppose you already loaded your Excel sheet into df
    df_device = pd.read_excel(Excel_File, sheet_name="Mid_Normalization")

    groups_mos, labels_mos = cluster_design(df_device, feature_cols="Rds(on)", n_groups=2)
    for cid, idx_list in groups_mos.items():
        print(f"Group of MOS {cid} (size={len(idx_list)}): {idx_list}")
    pass

    groups_core, labels_core = cluster_design(df_device, feature_cols="Ae", n_groups=2)
    for cid, idx_list in groups_core.items():
        print(f"Group of core {cid} (size={len(idx_list)}): {idx_list}")
    pass

    groups_diode, labels_diode = cluster_design(df_device, feature_cols="IF", n_groups=2)
    for cid, idx_list in groups_diode.items():
        print(f"Group of diode {cid} (size={len(idx_list)}): {idx_list}")
    pass

    df_freq_inductance = load_design_sheet(
        Excel_File,
        sheet_name="Input",
        value_name_row_excel=3,
        data_start_row_excel=4
    )

    df_freq_inductance.columns = df_freq_inductance.columns.astype(str).str.strip()
    # print(df_freq_turn.columns)

    groups_freq, labels_freq = cluster_design(
        df_freq_inductance,
        feature_cols="SwitchingFrequency",
        n_groups=2,
        index_col="Value_Name",  # IMPORTANT: your sheet uses Order, not Index
        method="kmeans",
        standardize=True,
    )
    for cid, idx_list in groups_freq.items():
        print(f"Group of frequency {cid} (size={len(idx_list)}): {idx_list}")

    groups_inductance, labels_inductance = cluster_design(
        df_freq_inductance,
        feature_cols="Inductance_value",
        n_groups=2,
        index_col="Value_Name",  # IMPORTANT: your sheet uses Order, not Index
        method="kmeans",
        standardize=True,
    )
    for cid, idx_list in groups_inductance.items():
        print(f"Group of inductance {cid} (size={len(idx_list)}): {idx_list}")

    samples = sample_group_combinations_n(
        [groups_mos,groups_core,groups_diode, groups_freq, groups_inductance],
        s=3,
        mode="B",  # or "A"
        require_full=True,
        id_base =0,
        seed = 2,
    )
    print(samples)

    indices_Dic = tuple_samples_to_flat_indices_n(samples,dims = [12,8,6,10,10])
    indices_1based = [values for values in indices_Dic.values()]

    # Ensure indices_1based is a NumPy array for fast column slicing
    indices = np.asarray(indices_1based)  # shape: (n_rows, n_cols

    # Open the writer ONCE
    with pd.ExcelWriter(Excel_File, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:

        # If you truly need to replace these sheets, first delete them (optional example below)
        # Otherwise just write new names.

        for i in range(indices.shape[1]):
            test_index = indices[:, i]  # fast column extract
            df = pd.DataFrame({"indices_1based": test_index})
            df.to_excel(writer, sheet_name=f"Mid_Z_Init_T{i + 1}", index=False)

