import numpy as np
import matplotlib.pyplot as plt
import os
from Bayesian_Optimziation_Helper import (
    decode_any,
)

def plot_gp_over_candidates(bo, show_ei=False, show_ucb = True, show_next=True, title=None, save_as=None, n_init=None):
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
    with plt.rc_context(rc):
        mu, std = bo.predict_all()
        idx = np.arange(1, bo.N + 1)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
        ax.fill_between(idx, mu - 2*std, mu + 2*std, alpha=0.15, label="±2σ")
        ax.plot(idx, mu, label="GP mean")

        # ----- observed points: seeds vs previous vs latest -----
        # ----- observed points: seeds vs previous vs latest -----
        if bo._chosen_idx:
            obs_1b = np.array(bo._chosen_idx) + 1
            y_obs = bo._y_obs
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
        # EI on right axis
        if show_ei:
            ei = bo.ei_all()
            ax2 = ax.twinx()
            ax2.plot(idx, ei, linestyle=":", label="EI")
            ax2.set_ylabel("EI")

            # merge legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", **LEGEND_KW)
        else:
            ax.legend(loc="best", **LEGEND_KW)
        # next suggestion

        if show_ucb:
            ucb = bo.ucb_all()
            ax2 = ax.twinx()
            ax2.plot(idx, ucb, linestyle=":", label="UCB")
            ax2.set_ylabel("UCB")

            # merge legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="best", **LEGEND_KW)
        else:
            ax.legend(loc="best", **LEGEND_KW)
        # next suggestion


        if show_next:
            next_idxs, _ = bo.suggest_next(top_k=1)
            ax.axvline(int(next_idxs[0]), linestyle="--", label="next suggestion")

        ax.set_xlabel("Candidate index (Input column)")
        ax.set_ylabel("Predicted Efficiency (GP mean)")
        ax.set_title(title or "GP posterior over candidates")
        fig.tight_layout()

        if save_as:
            plt.savefig(save_as, dpi=600, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def get_simulated_efficiency_from_excel(index_1based, excel_path,
                              index_col="index_1based",
                              eff_col="Efficiency",
                              sheet_name=0):
    """
    Look up Efficiency in an Excel file using a 1-based index.

    Parameters
    ----------
    index_1based : int
        The candidate index (same as your BO idx).
    excel_path : str
        Path to the Excel file.
    index_col : str, optional
        Column name that stores the 1-based index.
    eff_col : str, optional
        Column name that stores the efficiency value.
    sheet_name : int or str, optional
        Excel sheet name or index (default: first sheet).

    Returns
    -------
    float
        Efficiency value for the given index_1based.

    Raises
    ------
    ValueError
        If no row is found for the given index_1based.
    """
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Find the row where index_col == index_1based
    row = df.loc[df[index_col] == index_1based]

    if row.empty:
        raise ValueError(f"No row found in Excel for {index_col} = {index_1based}")

    # Take the Efficiency value
    return float(row[eff_col].iloc[0])

def compute_rmse_fixed_indices(
    bo,
    excel_path,
    index_list,
    index_col="index_1based",
    real_col="Efficiency",
    sheet_name=0,
):
    """
    Compute RMSE using fixed 1-based indices.

    Parameters
    ----------
    bo : BO object with predict_all()
    excel_path : str
        Path to Excel file containing real data.
    index_list : array-like
        Predefined fixed index array (e.g., 50 random points).
    index_col : str
        Name of index column in Excel.
    real_col : str
        Name of real values (Efficiency) column.

    Returns
    -------
    rmse_val : float
    df : pd.DataFrame
        Combined table of index, real value, and predicted value.
    """

    # 1) GP predictions for all candidates
    mu, std = bo.predict_all()
    N = bo.N

    # Build prediction table
    pred_df = pd.DataFrame({
        index_col: np.arange(1, N + 1),
        "y_est": mu,
    })

    # 2) Read real data
    real_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 3) Convert index_list to DataFrame
    sample_df = pd.DataFrame({index_col: index_list})

    # 4) Merge predicted + real
    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    )

    # 5) Drop missing rows
    merged = merged.dropna(subset=["y_est", real_col])

    # 6) Compute RMSE
    y_true = merged[real_col].to_numpy(dtype=float)
    y_pred = merged["y_est"].to_numpy(dtype=float)
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred)**2)))

    return rmse_val, merged

import numpy as np
import pandas as pd

def compute_r2_fixed_indices(
    bo,
    excel_path,
    index_list,
    index_col="index_1based",
    real_col="Efficiency",
    sheet_name=0,
):
    # ---------------------------------------------------------
    # 1. GP predictions
    # ---------------------------------------------------------
    mu, std = bo.predict_all()
    N = bo.N

    pred_df = pd.DataFrame({
        index_col: np.arange(1, N + 1),
        "y_pred": mu,
    })

    real_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    sample_df = pd.DataFrame({index_col: index_list})

    # ---------------------------------------------------------
    # 2. Merge predicted + true values for fixed indices
    # ---------------------------------------------------------
    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    ).dropna(subset=["y_pred", real_col])

    y_true = merged[real_col].to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)

    # ---------------------------------------------------------
    # 3. R² on RAW (unnormalized) efficiency
    # ---------------------------------------------------------
    ss_res_raw = np.sum((y_true - y_pred)**2)
    ss_tot_raw = np.sum((y_true - np.mean(y_true))**2)
    r2_raw = 1 - ss_res_raw / ss_tot_raw if ss_tot_raw != 0 else 1.0

    # ---------------------------------------------------------
    # 4. Normalize true & predicted values (min–max scaling)
    # ---------------------------------------------------------
    y_min = y_true.min()
    y_max = y_true.max()
    denom = (y_max - y_min)

    if denom == 0:
        # all true values identical → normalization impossible → perfect R²
        return r2_raw, 1.0, merged

    y_true_n = (y_true - y_min) / denom
    y_pred_n = (y_pred - y_min) / denom

    # ---------------------------------------------------------
    # 5. R² on NORMALIZED values
    # ---------------------------------------------------------
    ss_res_norm = np.sum((y_true_n - y_pred_n)**2)
    ss_tot_norm = np.sum((y_true_n - np.mean(y_true_n))**2)
    r2_norm = 1 - ss_res_norm / ss_tot_norm if ss_tot_norm != 0 else 1.0

    return r2_raw, r2_norm, merged


def relative_r2(y_true, y_pred):
    """
    Compute relative R² (scale-invariant R²).
    Works much better when y_true values have very small variance,
    like efficiency values in the range (0.97–0.99).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Avoid division by zero
    eps = 1e-12
    y_true_safe = np.where(np.abs(y_true) < eps, eps, y_true)

    # Relative squared residuals
    rel_res = np.sum(((y_true - y_pred) / y_true_safe) ** 2)

    # Relative total variance
    y_mean = np.mean(y_true)
    rel_tot = np.sum(((y_true - y_mean) / y_true_safe) ** 2)

    if rel_tot == 0:
        return 1.0

    return 1 - rel_res / rel_tot

def compute_relative_r2_fixed_indices(
    bo,
    excel_path,
    index_list,
    index_col="index_1based",
    real_col="Efficiency",
    sheet_name=0,
):
    mu, std = bo.predict_all()
    N = bo.N

    pred_df = pd.DataFrame({
        index_col: np.arange(1, N + 1),
        "y_pred": mu,
    })

    real_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    sample_df = pd.DataFrame({index_col: index_list})

    merged = (
        sample_df
        .merge(pred_df, on=index_col, how="left")
        .merge(real_df[[index_col, real_col]], on=index_col, how="left")
    ).dropna(subset=["y_pred", real_col])

    y_true = merged[real_col].to_numpy(float)
    y_pred = merged["y_pred"].to_numpy(float)

    # relative R²
    r2_rel = relative_r2(y_true, y_pred)

    return r2_rel, merged

var_names=["Index","Device", "Freq", "Turn"]
# ---------- main manual BO loop (uses Z_index) ----------
def manual_bo_with_decode_and_plot(bo, Z_index, steps=5,indices = None,
                                   show_ei=False, show_ucb=True,save_plots_dir=None, save_history_csv=None,result_path=None):
    history = []
    if save_plots_dir: os.makedirs(save_plots_dir, exist_ok=True)

    for step in range(1, steps + 1):
        idxs, acqs = bo.suggest_next(1)

        # ranked = bo.rank_candidates(top_k=100)
        idx = int(idxs[0]); acq = float(acqs[0])
        decoded = decode_any(idx, Z_index, var_names=var_names)

        decoded_str = ", ".join(f"{k}={v}" for k, v in decoded.items())
        print(f"[SUGGEST] step {step}: col={idx} ({decoded_str}), ACQ={acq:.6g}")

        y_measured = get_simulated_efficiency_from_excel(idx,result_path)
        bo.tell(idx, y_measured)  # maximizing
        rmse_val, df50 = compute_rmse_fixed_indices(
            bo,
            result_path,
            index_list=indices,
            index_col="index_1based",
            real_col="Efficiency",
        )

        print("Fixed-index RMSE =", rmse_val)
        print(df50.head())  # show first few rows

        r2_raw, r2_norm, df50_2 = compute_r2_fixed_indices(
            bo,
            result_path,
            index_list=indices,
            index_col="index_1based",
            real_col="Efficiency"
        )

        r2_rel,df50_3 = compute_relative_r2_fixed_indices(
        bo,
        result_path,
        index_list=indices,
        index_col="index_1based",
        real_col="Efficiency",
        sheet_name=0,
        )

        title = f"After step {step} — col={idx} ({decoded_str})"
        save_as = f"{save_plots_dir}/gp_step_{step:03d}.png" if save_plots_dir else None
        plot_gp_over_candidates(bo, show_ei=show_ei,show_ucb=show_ucb, show_next=True, title=title, save_as=save_as,n_init =2)

        row = {
            "step": step,
            "index_1based": idx,
            "y_measured": y_measured,
            "ACQ_at_pick": acq,
            "RMSE": rmse_val,
            "R2": r2_raw,
            "R2-norm": r2_norm,
            "R2-rel": r2_rel,
        }
        row.update(decoded)  # add all variables dynamically
        history.append(row)

        print(f"    Updated. Best-so-far (max) = {float(np.max(bo._y_obs)):.6g}")

    hist_df = pd.DataFrame(history)
    if save_history_csv:
        hist_df.to_csv(save_history_csv, index=False)
        print(f"History saved to {save_history_csv}")
    return hist_df