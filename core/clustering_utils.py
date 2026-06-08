"""
Shared clustering utilities for initial-condition generation and length-scale learning.

Functions extracted here were previously duplicated across:
  - Initial_Condition_Generation_Paper01.py
  - Length_Scale_Learn.py
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

ClusterMethod = Literal["kmeans", "agglomerative"]
UniqMode = Literal["A", "B"]


def load_design_sheet(
    xlsx_path: str,
    sheet_name: str,
    *,
    value_name_row_excel: int = 3,
    data_start_row_excel: int = 4,
) -> pd.DataFrame:
    """
    Robust loader for the IBC project Excel layout:
      Row 1 : Order / x1 / x2 / ...
      Row 2 : Input_Type
      Row 3 : Value_Name  (used as column headers)
      Row 4+: data
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    value_row = value_name_row_excel - 1
    data_row = data_start_row_excel - 1

    cols = raw.iloc[value_row].tolist()
    df = raw.iloc[data_row:].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)

    if "Order" in df.columns:
        df["Order"] = pd.to_numeric(df["Order"], errors="coerce")

    return df


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
    Cluster design parameters into N groups using the selected feature columns.

    Returns
    -------
    groups : dict[int, list[int]]
        cluster_id -> list of row indices (from index_col).
    labels : np.ndarray
        Cluster label per row (in the filtered dataframe order).
    """
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1.")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    work = df.dropna(subset=feature_cols).copy()
    if len(work) < n_groups:
        raise ValueError(
            f"Only {len(work)} rows after dropping NaNs — cannot form {n_groups} groups."
        )

    X = work[feature_cols].to_numpy(dtype=float)
    if standardize:
        X = StandardScaler().fit_transform(X)

    if method == "kmeans":
        labels = KMeans(n_clusters=n_groups, n_init=20, random_state=random_state).fit_predict(X)
    elif method == "agglomerative":
        labels = AgglomerativeClustering(n_clusters=n_groups, linkage="ward").fit_predict(X)
    else:
        raise ValueError("method must be 'kmeans' or 'agglomerative'.")

    if index_col not in work.columns:
        raise KeyError(f"index_col='{index_col}' not found. Columns: {list(work.columns)}")
    ids = work[index_col].to_numpy()

    groups: Dict[int, List[int]] = {
        cid: [int(v) for v in ids[labels == cid]] for cid in range(n_groups)
    }
    return groups, labels


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
    Sample s candidates per cluster-combination across N variables.

    Parameters
    ----------
    groups_list : list of dicts, one per variable.
                  Each dict maps cluster_id -> list of candidate IDs.
    s           : samples per cluster-combination.
    mode        : "A" no duplicate tuples; "B" additionally all dims differ from previous.
    seed        : RNG seed.
    require_full: raise if cannot draw all s samples for a combo.
    id_base     : 1 -> shift int IDs by +1 (0-based -> 1-based); 0 -> keep as-is.
    """
    if s < 1:
        raise ValueError("s must be >= 1")
    if not groups_list:
        raise ValueError("groups_list must contain at least 1 variable group dict")

    rng = np.random.default_rng(seed)
    cluster_keys_per_var = [sorted(g.keys()) for g in groups_list]
    n_vars = len(groups_list)

    def to_base(v: Any) -> Any:
        if isinstance(v, (int, np.integer)):
            return int(v) if id_base == 0 else int(v) + 1
        return v

    def draw_one(combo: Tuple[int, ...]) -> Tuple[Any, ...]:
        return tuple(to_base(rng.choice(groups_list[j][cj])) for j, cj in enumerate(combo))

    out: Dict[Tuple[int, ...], List[Tuple[Any, ...]]] = {}
    for combo in itertools.product(*cluster_keys_per_var):
        picked: List[Tuple[Any, ...]] = []
        for _ in range(s):
            if not picked:
                picked.append(draw_one(combo))
                continue
            prev = picked[-1]
            ok = False
            for _ in range(5000):
                cand = draw_one(combo)
                if mode == "A":
                    ok = cand not in picked
                elif mode == "B":
                    ok = all(cand[j] != prev[j] for j in range(n_vars)) and cand not in picked
                else:
                    raise ValueError("mode must be 'A' or 'B'")
                if ok:
                    break
            if not ok:
                msg = f"Cannot draw sample #{len(picked)+1} for combo {combo} under mode={mode}."
                if require_full:
                    raise RuntimeError(msg)
                break
            picked.append(cand)
        out[combo] = picked
    return out


def tuple_samples_to_flat_indices_n(
    samples: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    *,
    dims: Sequence[int],
) -> Dict[Tuple[int, ...], List[int]]:
    """
    Convert N-dimensional 1-based index tuples to flattened row-major indices.

    dims = [D1, D2, ..., DN]  (sizes of each dimension, 1-based)
    flat_index = 1 + sum((ik - 1) * stride_k)
    """
    n = len(dims)
    strides = [1] * n
    acc = 1
    for i in range(n - 1, -1, -1):
        strides[i] = acc
        acc *= dims[i]

    flat_indices: Dict[Tuple[int, ...], List[int]] = {}
    for key, tuples in samples.items():
        idx_list: List[int] = []
        for t in tuples:
            if len(t) != n:
                raise ValueError(f"Tuple {t} has length {len(t)}, expected {n}")
            idx = 1 + sum((t[k] - 1) * strides[k] for k in range(n))
            idx_list.append(idx)
        flat_indices[key] = idx_list
    return flat_indices
