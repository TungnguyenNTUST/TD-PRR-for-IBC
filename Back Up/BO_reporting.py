from __future__ import annotations

import os
from typing import Sequence, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bayesian_Optimziation_Helper import decode_any



def plot_gp_over_candidates(bo, secondary: Optional[str] = "ucb", mu_std: Optional[Tuple[np.ndarray, np.ndarray]] = None, show_next=True, title=None, save_as=None, n_init=None):
    """
    Plot GP mean ±2σ over candidate index, with optional EI/UCB on a secondary axis.
    """
    if secondary not in (None, "ei", "ucb"):
        raise ValueError("secondary must be one of: None, 'ei', 'ucb'")

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        if mu_std is None:
            mu, std = bo.predict_all()
        else:
            mu, std = mu_std

        # mu, std = bo.predict_all()
        idx = np.arange(1, bo.N + 1)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
        ax.fill_between(idx, mu - 2*std, mu + 2*std, alpha=0.15, label="±2σ")
        ax.plot(idx, mu, label="GP mean")

        # ----- observed points: seeds vs previous vs latest -----
        # ----- observed points: seeds vs previous vs latest -----
        if bo._chosen_idx:
            obs_1b = np.array(bo._chosen_idx, dtype=int) + 1
            y_obs = np.array(bo._y_obs, dtype=float)
            n0 = int(n_init) if n_init is not None else int(getattr(bo, "_n_init", 0))

            # pick fixed colors
            c_seeds = "tab:orange"  # initial seeds
            c_prev = "tab:green"  # all previous (non-latest) observations
            c_latest = "tab:red"  # latest observation (always red)

            # seeds
            if n0 > 0:
                ax.scatter(obs_1b[:n0], y_obs[:n0], s=40, marker="o",
                           color=c_seeds, label="initial observation", zorder=4)

            # previously observed (excluding latest)
            if len(obs_1b) - n0 >= 2:
                ax.scatter(obs_1b[n0:-1], y_obs[n0:-1], s=40, marker="x",
                           color=c_prev, label="observed (prev)", zorder=4)

            # latest observed (the most recent point) — ALWAYS RED
            if len(obs_1b) > n0:
                ax.scatter(obs_1b[-1], y_obs[-1], s=90, marker="X",
                           color=c_latest, linewidths=1.5, label="observed (latest)", zorder=5)
        # put this near the top of your function (inside the rc_context is fine)
        LEGEND_KW = dict(
            fontsize=8,  # smaller legend font
            markerscale=0.8,  # scale down markers in legend
            handlelength=1.0,  # shorter line segments
            handletextpad=0.4,  # tighter text spacing
            borderpad=0.3,  # tighter box padding
            labelspacing=0.3,  # less vertical space between items
            frameon=True,
            framealpha=0.8,  # faint box
        )
        # EI or UCB on right axis
        ax2 = None
        if secondary == "ei":
            ei = bo.ei_all()
            ax2 = ax.twinx()
            ax2.plot(idx, ei, linestyle=":", label="EI")
            ax2.set_ylabel("EI")

        elif secondary == "ucb":
            ucb = bo.ucb_all()
            ax2 = ax.twinx()
            ax2.plot(idx, ucb, linestyle=":", label="UCB")
            ax2.set_ylabel("UCB")
        # next suggestion


        if show_next:
            next_idxs, _ = bo.suggest_next(top_k=1)
            ax.axvline(int(next_idxs[0]), linestyle="--", label="next suggestion")

        ax.set_xlabel("Candidate index (Input column)")
        ax.set_ylabel("Predicted Efficiency (GP mean)")
        ax.set_title(title or "GP posterior over candidates")

        # merge legend
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

def compute_rmse_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, pd.DataFrame]:
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
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return rmse_val, merged


def compute_r2_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, float, pd.DataFrame]:

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
    r2_raw = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    y_min, y_max = y_true.min(), y_true.max()
    denom = y_max - y_min
    if denom == 0:
        return r2_raw, 1.0, merged

    y_true_n = (y_true - y_min) / denom
    y_pred_n = (y_pred - y_min) / denom

    ss_res_n = np.sum((y_true_n - y_pred_n) ** 2)
    ss_tot_n = np.sum((y_true_n - np.mean(y_true_n)) ** 2)
    r2_norm = 1 - ss_res_n / ss_tot_n if ss_tot_n != 0 else 1.0

    return float(r2_raw), float(r2_norm), merged


def relative_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    eps = 1e-12
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)

    rel_res = np.sum(((y_true - y_pred) / y_true_safe) ** 2)

    y_mean = np.mean(y_true)
    rel_tot = np.sum(((y_true - y_mean) / y_true_safe) ** 2)

    if rel_tot == 0:
        return 1.0
    return float(1 - rel_res / rel_tot)


def compute_relative_r2_fixed_indices_from_df_mu(
    mu: np.ndarray,
    real_df: pd.DataFrame,
    index_list: Sequence[int],
    *,
    index_col: str = "index_1based",
    real_col: str = "Efficiency",
) -> Tuple[float, pd.DataFrame]:
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

    r2_rel = relative_r2(y_true, y_pred)
    return float(r2_rel), merged

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
    plot_secondary: Optional[str] = "ucb",   # None / "ucb" / "ei"
    n_init: int = 2,
) -> Dict[str, Any]:
    decoded = decode_any(idx_next, Z_index, var_names=var_names)
    mu, std = bo.predict_all()
    stats = float(np.mean(std))
    y_best = float(np.max(bo._y_obs))
    target = y_best - float(eps)
    if save_plots_dir:
        decoded_str = ", ".join(f"{k}={v}" for k, v in decoded.items())
        title = f"After step {step} — pick={idx_next} ({decoded_str})"
        save_as = os.path.join(save_plots_dir, f"gp_step_{step:03d}.png")
        plot_gp_over_candidates(
            bo,
            secondary=plot_secondary,
            show_next=True,
            mu_std=(mu, std),
            title=title,
            save_as=save_as,
            n_init=n_init,
        )

    row = {
        "step": step,
        "picked": int(idx_next),
        "y_measured": float(y_measured),
        "best": float(y_best),
        "target": float(target),
        "Xp_size": int(Xp_size),
        "avg_std": stats,
        # "R2": float(r2_raw),
        # "R2_norm": float(r2_norm),
        # "R2_rel": float(r2_rel),
        **decoded,
    }
    return row
