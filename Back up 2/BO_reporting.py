"""
GP reporting, metrics, and plotting utilities for Bayesian Optimization.

Merged from:
  - BO_reporting.py   (canonical metrics using pre-loaded DataFrames)
  - Bayesian_Optimization_Plotting.py  (superseded — legacy Excel-re-reading versions removed)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Bayesian_Optimziation_Helper import decode_any


# ---------------------------------------------------------------------------
# GP posterior plot
# ---------------------------------------------------------------------------

def plot_gp_over_candidates(
    bo,
    secondary: Optional[str] = "ucb",
    mu_std: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_next: bool = True,
    title: Optional[str] = None,
    save_as: Optional[str] = None,
    n_init: Optional[int] = None,
) -> None:
    """
    Plot GP posterior mean ±2σ over all candidate indices.

    Parameters
    ----------
    secondary : None | "ei" | "ucb"  — overlay acquisition on a right axis.
    mu_std    : pre-computed (mu, std) arrays; if None, bo.predict_all() is called.
    show_next : draw a vertical dashed line at the next suggested index.
    n_init    : number of initial seed observations (coloured differently).
    save_as   : file path to save the figure; if None, plt.show() is called.
    """
    if secondary not in (None, "ei", "ucb"):
        raise ValueError("secondary must be None, 'ei', or 'ucb'.")

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    LEGEND_KW = dict(
        fontsize=8, markerscale=0.8, handlelength=1.0,
        handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
        frameon=True, framealpha=0.8,
    )

    with plt.rc_context(rc):
        mu, std = mu_std if mu_std is not None else bo.predict_all()
        idx = np.arange(1, bo.N + 1)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
        ax.fill_between(idx, mu - 2 * std, mu + 2 * std, alpha=0.15, label="±2σ")
        ax.plot(idx, mu, label="GP mean")

        if bo._chosen_idx:
            obs_1b = np.array(bo._chosen_idx, dtype=int) + 1
            y_obs = np.array(bo._y_obs, dtype=float)
            n0 = int(n_init) if n_init is not None else int(getattr(bo, "_n_init", 0))

            if n0 > 0:
                ax.scatter(obs_1b[:n0], y_obs[:n0], s=40, marker="o",
                           color="tab:orange", label="initial observation", zorder=4)
            if len(obs_1b) - n0 >= 2:
                ax.scatter(obs_1b[n0:-1], y_obs[n0:-1], s=40, marker="x",
                           color="tab:green", label="observed (prev)", zorder=4)
            if len(obs_1b) > n0:
                ax.scatter(obs_1b[-1], y_obs[-1], s=90, marker="X",
                           color="tab:red", linewidths=1.5, label="observed (latest)", zorder=5)

        ax2 = None
        if secondary == "ei":
            ax2 = ax.twinx()
            ax2.plot(idx, bo.ei_all(), linestyle=":", label="EI")
            ax2.set_ylabel("EI")
        elif secondary == "ucb":
            ax2 = ax.twinx()
            ax2.plot(idx, bo.ucb_all(), linestyle=":", label="UCB")
            ax2.set_ylabel("UCB")

        if show_next:
            next_idxs, _ = bo.suggest_next(top_k=1)
            ax.axvline(int(next_idxs[0]), linestyle="--", label="next suggestion")

        ax.set_xlabel("Candidate index (Input column)")
        ax.set_ylabel("Predicted Efficiency (GP mean)")
        ax.set_title(title or "GP posterior over candidates")

        if ax2 is not None:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", **LEGEND_KW)
        else:
            ax.legend(loc="best", **LEGEND_KW)

        fig.tight_layout()
        if save_as:
            plt.savefig(save_as, dpi=600, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Metrics (pre-loaded DataFrame versions — superior to the legacy Plotting.py versions)
# ---------------------------------------------------------------------------

def compute_rmse_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, pd.DataFrame]:
    """Compute RMSE of GP mean vs real values at fixed indices."""
    N = len(mu)
    pred_df = pd.DataFrame({index_col: np.arange(1, N + 1), "y_pred": mu})
    sample_df = pd.DataFrame({index_col: np.array(index_list, dtype=int)})
    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    ).dropna(subset=["y_pred", real_col])
    y_true = merged[real_col].to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))), merged


def compute_r2_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, float, pd.DataFrame]:
    """Compute raw R² and min-max normalised R² at fixed indices."""
    N = len(mu)
    pred_df = pd.DataFrame({index_col: np.arange(1, N + 1), "y_pred": mu})
    sample_df = pd.DataFrame({index_col: np.array(index_list, dtype=int)})
    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    ).dropna(subset=["y_pred", real_col])
    y_true = merged[real_col].to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_raw = 1.0 - ss_res / ss_tot if ss_tot != 0 else 1.0

    denom = y_true.max() - y_true.min()
    if denom == 0:
        return float(r2_raw), 1.0, merged

    y_true_n = (y_true - y_true.min()) / denom
    y_pred_n = (y_pred - y_true.min()) / denom
    ss_res_n = np.sum((y_true_n - y_pred_n) ** 2)
    ss_tot_n = np.sum((y_true_n - np.mean(y_true_n)) ** 2)
    r2_norm = 1.0 - ss_res_n / ss_tot_n if ss_tot_n != 0 else 1.0

    return float(r2_raw), float(r2_norm), merged


def relative_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scale-invariant R² — works well when efficiency variance is tiny (0.97–0.99)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    safe = np.where(np.abs(y_true) < eps, eps, y_true)
    rel_res = np.sum(((y_true - y_pred) / safe) ** 2)
    rel_tot = np.sum(((y_true - np.mean(y_true)) / safe) ** 2)
    return 1.0 if rel_tot == 0 else float(1.0 - rel_res / rel_tot)


def compute_relative_r2_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, pd.DataFrame]:
    """Compute relative R² at fixed indices."""
    N = len(mu)
    pred_df = pd.DataFrame({index_col: np.arange(1, N + 1), "y_pred": mu})
    sample_df = pd.DataFrame({index_col: np.array(index_list, dtype=int)})
    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    ).dropna(subset=["y_pred", real_col])
    y_true = merged[real_col].to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)
    return relative_r2(y_true, y_pred), merged


# ---------------------------------------------------------------------------
# Step logger
# ---------------------------------------------------------------------------

def log_step(
    *,
    bo,
    step: int,
    idx_next: int,
    y_measured: float,
    Z_index: np.ndarray,
    var_names: Sequence[str],
    eps: float,
    Xp_size: int,
    save_plots_dir: Optional[str] = None,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
) -> Dict[str, Any]:
    """
    Log one BO step: decode the chosen index, compute posterior stats, optionally plot.

    Returns a dict suitable for appending to a history list → pd.DataFrame.
    """
    decoded = decode_any(idx_next, Z_index, var_names=var_names)
    mu, std = bo.predict_all()
    y_best = float(np.max(bo._y_obs))
    target = y_best - float(eps)

    if save_plots_dir:
        decoded_str = ", ".join(f"{k}={v}" for k, v in decoded.items())
        title = f"After step {step} — pick={idx_next} ({decoded_str})"
        save_as = os.path.join(save_plots_dir, f"gp_step_{step:03d}.png")
        plot_gp_over_candidates(
            bo,
            secondary=plot_secondary,
            mu_std=(mu, std),
            show_next=True,
            title=title,
            save_as=save_as,
            n_init=n_init,
        )

    return {
        "step": step,
        "picked": int(idx_next),
        "y_measured": float(y_measured),
        "best": float(y_best),
        "target": float(target),
        "Xp_size": int(Xp_size),
        "avg_std": float(np.mean(std)),
        **decoded,
    }
