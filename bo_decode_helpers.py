
from typing import Sequence, Union, Optional, Dict, Tuple, Any
import numpy as np

def decode_any(
    col1_based: int,
    Z_index: np.ndarray,
    *,
    var_names: Optional[Sequence[str]] = None,
    select: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    base: int = 1,
) -> Union[Tuple[int, ...], Dict[str, int], int, Tuple[int, ...], Dict[str, int]]:
    """
    Decode a 1-based (or base-based) column index from Z_index shaped (n_vars, N).

    Parameters
    ----------
    col1_based : int
        Column index in the encoded space (1-based by default).
    Z_index : np.ndarray
        Shape (n_vars, N). Each column is a code vector [x1, x2, ..., xM]^T.
    var_names : optional list[str]
        Names for variables, length must equal n_vars. If provided, returns dict by default.
    select : int|str|list[int|str]|None
        - None: return all variables (tuple if no names, dict if names)
        - int: return only that variable index (0-based)
        - str: return only that variable by name
        - list: return subset
    base : int
        Indexing base for col (default 1).

    Returns
    -------
    If select is None:
        tuple[int,...] or dict[str,int]
    If select is specified:
        int or tuple/dict subset
    """
    if Z_index is None or Z_index.ndim != 2:
        raise ValueError("Z_index must be a 2D array of shape (n_vars, N).")

    n_vars, N = Z_index.shape

    if var_names is not None and len(var_names) != n_vars:
        raise ValueError(f"var_names length ({len(var_names)}) must equal n_vars ({n_vars}).")

    c0 = int(col1_based) - base
    if not (0 <= c0 < N):
        raise IndexError(f"Column out of range: got {col1_based} with base={base}, valid=[{base}..{N+base-1}].")

    vec = Z_index[:, c0].astype(int)  # (n_vars,)

    # Helper: map name -> index
    name_to_idx = {name: i for i, name in enumerate(var_names)} if var_names else {}

    def _pick_one(key: Union[int, str]) -> int:
        if isinstance(key, int):
            if not (0 <= key < n_vars):
                raise IndexError(f"Variable index {key} out of range [0..{n_vars-1}].")
            return int(vec[key])
        if isinstance(key, str):
            if not var_names:
                raise ValueError("select by name requires var_names.")
            if key not in name_to_idx:
                raise KeyError(f"Unknown variable name '{key}'. Valid: {list(name_to_idx.keys())}")
            return int(vec[name_to_idx[key]])
        raise TypeError("select entries must be int or str.")

    # Return logic
    if select is None:
        return {var_names[i]: int(vec[i]) for i in range(n_vars)} if var_names else tuple(int(x) for x in vec)

    if isinstance(select, (int, str)):
        return _pick_one(select)

    # select is a sequence
    out = [_pick_one(k) for k in select]
    if var_names and all(isinstance(k, str) for k in select):
        return {k: v for k, v in zip(select, out)}
    return tuple(out)
