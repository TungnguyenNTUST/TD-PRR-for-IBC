"""
LUT-based evaluation utilities for Bayesian Optimization.

Single source of truth for load_eval_lut and make_evaluate_fn_from_lut.
Previously duplicated in:
  - Bayesian_Optimization_APP.py
  - Bayesian_Optimization_APP_Paper01.py
  - Bayesian_Optimization_APP_Paper01_EI.py
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd


def load_eval_lut(
    result_xlsx_path: str,
    sheet_name: str | int = 0,
) -> Tuple[Dict[int, float], Set[int], pd.DataFrame]:
    """
    Load an efficiency results sheet into a look-up table.

    Convention
    ----------
    - Column 0 (first column)  : 1-based design-option index.
    - Column -1 (last column)  : objective value (e.g. efficiency).

    Returns
    -------
    lut       : dict  index -> objective value
    available : set of all available indices
    real_df   : two-column DataFrame [index_col, obj_col] for metrics
    """
    df = pd.read_excel(result_xlsx_path, sheet_name=sheet_name)

    index_col = df.columns[0]   # first column is always the index
    obj_col = df.columns[-1]    # last column is always the objective

    df[index_col] = df[index_col].astype(int)
    df[obj_col] = pd.to_numeric(df[obj_col], errors="coerce")
    df = df.dropna(subset=[obj_col])

    lut: Dict[int, float] = dict(zip(df[index_col].to_numpy(), df[obj_col].to_numpy()))
    available: Set[int] = set(lut.keys())
    real_df = df[[index_col, obj_col]].copy()

    return lut, available, real_df


def make_evaluate_fn_from_lut(lut: Dict[int, float]):
    """
    Wrap a LUT dict as an evaluate_fn(index_1based, x) -> float callable.

    Raises KeyError if an index is not in the LUT (fail fast).
    """
    def evaluate_fn(index_1based: int, x: np.ndarray) -> float:
        idx = int(index_1based)
        if idx not in lut:
            raise KeyError(f"Index {idx} not found in LUT.")
        return float(lut[idx])
    return evaluate_fn


def load_index_eff(
    file_path: str,
    sheet_name: str | int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load index and efficiency arrays from a results sheet.

    Returns
    -------
    indices : int array of 1-based indices
    eff     : float array of efficiency values
    df      : clean DataFrame (rows with valid efficiency only)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    index_col = df.columns[0]
    eff_col = df.columns[-1]
    df[index_col] = df[index_col].astype(int)
    df[eff_col] = pd.to_numeric(df[eff_col], errors="coerce")
    df = df.dropna(subset=[eff_col])
    return df[index_col].to_numpy(dtype=int), df[eff_col].to_numpy(dtype=float), df
