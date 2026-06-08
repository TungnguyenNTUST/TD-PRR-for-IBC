from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Tuple, Optional
import re

from core.constants import Frequency_list, Inductance_list
from core.paths import FILE_PATH as Excel_File

Core_PN = "KPH200-060A"

# KAM184-075A
# KAM185-060A
# KAM200-060A
# KAM226-060A
# KPH200-060A
# KPH226-060A
# KH184-060A-H
# KH200-060A-H


def get_number_before_A(pn: str) -> int:
    """
    Extract the number immediately before 'A' in PN.

    Example:
      'KH184-060A-H' -> 60
      'KAM184-075A'  -> 75
    """
    m = re.search(r'(\d+)A', pn)

    if not m:
        raise ValueError(f"No number before 'A' found in: {pn}")

    return int(m.group(1))

def load_core_params_dict(
    xlsx_path: Union[str, Path],
    *,
    sheet_name: str = 0,
    pn_col: str = "Device",
    param_cols: Sequence[str] = ("Ae", "le", "k1", "k2", "anpha", "beta", "b", "c"),
    drop_if_any_missing: bool = False,
) -> Dict[str, List[float]]:
    """
    Read an Excel sheet like your screenshot and return:
        { part_number: [Ae, le, k1, k2, anpha, beta, b, c] }

    Logic:
    - Normalize column names (strip spaces)
    - Keep rows where Device exists AND at least one param is non-empty
      (this naturally isolates the "core" block)
    - Convert params to numeric when possible
    """

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    if pn_col not in df.columns:
        raise KeyError(f"'{pn_col}' not found. Available columns: {list(df.columns)}")

    missing = [c for c in param_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing param columns: {missing}. Available columns: {list(df.columns)}")

    # Ensure PN strings
    pn_series = df[pn_col].astype(str).str.strip()
    pn_ok = pn_series.notna() & (pn_series != "") & (pn_series.str.lower() != "nan")

    # Convert parameter columns to numeric if possible
    df_params = df.loc[:, list(param_cols)].apply(pd.to_numeric, errors="coerce")

    # Select rows that actually have core params filled
    if drop_if_any_missing:
        row_ok = pn_ok & df_params.notna().all(axis=1)
    else:
        row_ok = pn_ok & df_params.notna().any(axis=1)

    df_core = df.loc[row_ok, [pn_col] + list(param_cols)].copy()
    df_core[pn_col] = df_core[pn_col].astype(str).str.strip()

    # Build dict
    out: Dict[str, List[float]] = {}
    for _, r in df_core.iterrows():
        pn = r[pn_col]
        vals = []
        for c in param_cols:
            v = pd.to_numeric(r[c], errors="coerce")
            vals.append(float(v) if pd.notna(v) else float("nan"))
        out[pn] = vals

    return out


MU0 = 4.0 * math.pi * 1e-7  # H/m

def solve_N_from_L(
    *,
    L: float,          # target inductance [H]
    Ae: float,         # effective core area [m^2]
    le: float,         # effective magnetic path length [m]
    I: float,          # current used for incremental/permeability model [A]
    mu_r0: float = 60.0,  # low-field relative permeability (your "100")
    beta: float = 0.0,     # beta parameter
    c: float = 1.0,        # exponent
    N_min: float = 5,
    N_max: float = 500.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[float, int]:
    """
    Solve for N in:
      L = MU0*(Ae/le) * N^2 * mu_r(H)
      mu_r(H) = mu_r0 / (1 + beta*H^c)
      H = N*I/le

    Returns
    -------
    (N_float, N_int_nearest)

    Notes
    -----
    - Ae, le must be in meters^2 and meters.
    - If beta == 0, closed-form sqrt solution is used.
    - Uses bisection with automatic bracketing (very robust).
    """

    if L <= 0:
        raise ValueError("L must be > 0")
    if Ae <= 0 or le <= 0:
        raise ValueError("Ae and le must be > 0")
    if mu_r0 <= 0:
        raise ValueError("mu_r0 must be > 0")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if c <= 0:
        raise ValueError("c must be > 0")
    if N_min <= 0 or N_max <= N_min:
        raise ValueError("Invalid N_min/N_max")

    K = MU0 * (Ae / le)  # [H] scaling

    # ---- Closed form if permeability is constant ----
    if beta == 0.0 or I == 0.0:
        N = math.sqrt(L / (K * mu_r0))
        N_int = max(1, int(round(N)))
        return N, N_int

    def mu_r_of_N(N: float) -> float:
        H = (N * I) / (le*79.58)
        return mu_r0 / (1.0 + beta * (H ** c))

    def f(N: float) -> float:
        # f(N) = predicted_L(N) - target_L
        return K * (N ** 2) * mu_r_of_N(N) - L +  50e-6

    # ---- Bracket the root ----
    a, b = N_min, N_max
    fa, fb = f(a), f(b)
    print(fa)
    print(fb)
    # Expand b if needed (up to a reasonable cap) to ensure sign change
    expand = 0
    while fa * fb > 0 and expand < 30:
        b *= 2.0
        fb = f(b)
        expand += 1

    if fa * fb > 0:
        raise RuntimeError(
            "Failed to bracket a root for N. "
            "Try increasing N_max (or check units/parameters)."
        )

    # ---- Bisection ----
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        if abs(fm) < tol:
            N = m
            N_int = max(1, int(round(N)))
            return N, N_int

        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

        if abs(b - a) < 1e-12 * max(1.0, abs(m)):
            break

    N = 0.5 * (a + b)
    N_int = max(1, int(round(N)))
    return N, N_int


def compute_L_from_NI(
    *,
    N: float,
    I: float,
    Ae: float,
    le: float,
    mu_r0: float,
    beta: float,
    c: float,
) -> float:
    """
    Compute inductance L for given N and I using nonlinear mu(H) model.

    Returns:
        L in Henry
    """

    if N <= 0:
        raise ValueError("N must be > 0")
    if Ae <= 0 or le <= 0:
        raise ValueError("Ae and le must be > 0")

    # Base factor
    K = MU0 * (Ae / le)

    # Field (convert to Oe like your solver)
    H = (N * I) / (le * 79.58)

    # Relative permeability
    mu_r = mu_r0 / (1.0 + beta * (H ** c))

    # Inductance
    L = K * (N ** 2) * mu_r

    return L


if __name__ == "__main__":
    core_dict = load_core_params_dict(Excel_File, sheet_name="Mid_Normalization")
    print(core_dict[Core_PN])
    # Params (ensure numeric)
    b = float(core_dict[Core_PN][6])
    c = float(core_dict[Core_PN][7])
    Ae = 3*float(core_dict[Core_PN][0])*1e-4
    le = float(core_dict[Core_PN][1])*1e-2

    N_float, N_int = solve_N_from_L(
        L=Inductance_list[10],
        Ae=Ae,
        le=le,
        I=16.67,       # current at which you want incremental L
        mu_r0=get_number_before_A(Core_PN),
        beta=b,
        c=c,
    )

    print(N_int)

    L0 = compute_L_from_NI(N=N_int,I = 1,Ae=Ae,le=le,mu_r0=get_number_before_A(Core_PN),beta=b,c=c)
    print(L0)
