import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Set, Sequence, List
from Bayesian_Optimization_Source import DiscreteBO
# from BO_evaluation import load_eval_lut, make_evaluate_fn_from_lut
from BO_reporting import log_step

# =========================================================
# 0) CONFIG
# =========================================================
FILE_PATH = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx"
# RESULT_PATH = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Data\final_V2.xlsx"
RESULT_PATH = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Data\20260108_41_efficiency_results.xlsx"
OUT_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Results"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_SHEET = "Mid_Input_encoding"
INDEX_SHEET = "Mid_Input_Indexing"
EVAL_SHEET = "Test Case 01"
INIT_SHEET = "Mid_Z_Testing"

INIT1, INIT2, INIT3 = 1355, 1920, 1
INIT_LIST = [INIT1,INIT2]
EPS = 0.005

# =========================================================
# 1) DATA LOADING
# =========================================================

def read_index_list(
    excel_path: str,
    sheet_name: str = "Mid_Z_Testing",
    column_name: str = "index_1based",
) -> List[int]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, usecols=[column_name])
    return df[column_name].dropna().astype(int).tolist()

def load_input_and_index(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df_enc = pd.read_excel(file_path, sheet_name=INPUT_SHEET, header=0)
    df_idx = pd.read_excel(file_path, sheet_name=INDEX_SHEET, header=0)

    Input = df_enc.to_numpy(dtype=float).T          # (D, N)
    Z_matrix = df_idx.to_numpy(dtype=int)           # whatever your indexing shape is
    return Input, Z_matrix

# =========================================================
# 2) EVALUATION LUT
# =========================================================

def load_eval_lut(result_xlsx_path: str, sheet_name: str) -> Tuple[Dict[int, float], Set[int], pd.DataFrame]:
    df = pd.read_excel(result_xlsx_path, sheet_name=sheet_name)
    df["index_1based"] = df["index_1based"].astype(int)
    df["Efficiency"] = pd.to_numeric(df["Efficiency"], errors="coerce")
    df = df.dropna(subset=["Efficiency"])

    lut = dict(zip(df["index_1based"].to_numpy(), df["Efficiency"].to_numpy()))
    available = set(lut.keys())

    real_df = df[["index_1based", "Efficiency"]].copy()
    return lut, available, real_df

def make_evaluate_fn_from_lut(lut: Dict[int, float]):
    def evaluate_fn(index_1based: int, x: np.ndarray) -> float:
        idx = int(index_1based)
        return float(lut[idx])   # let KeyError raise if missing
    return evaluate_fn

def load_fixed_indices(file_path: str, sheet_name: str = "Mid_Z_Testing") -> np.ndarray:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    return df.iloc[:, 0].to_numpy(dtype=int)

# =========================================================
# 3) BO FACTORY
# =========================================================
def build_bo(Input: np.ndarray) -> DiscreteBO:
    D, N = Input.shape
    length_scales = [1.0] * D

    #7.5% training data
    # length_scales = [4.48485009e+04,1.00000000e+06,2.32611108e-01,5.96753725e-01,
    #  2.58847255e+04,5.29422841e+00,1.08791850e+05]

    # 2.5% training data
    # length_scales =[5.90400726e-01,1.00000000e+06,2.22711691e+00,1.05948795e+00,
    #  1.00000000e+06,3.75589536e+00,5.38944742e+00]

    #Ver1
    # length_scales = [5.31690932e+02, 1.92227490e-03, 3.17702206e+05, 6.08036621e+02,
    #  5.53837852e+00,2.85820328e+00,1.31154351e+01]

    # Ver2
    # length_scales = [2.24406549e-01, 5.43769467e+05, 6.34938061e+03, 1.78061805e+00,
    #  2.46152093e+05, 1.44958157e+00, 3.47740834e+01]
    bo = DiscreteBO(
        Input=Input,
        length_scales=length_scales,
        sigma_f2=1.0,
        sigma_n2=0.05**2,
        ei_acq=False,
        xi=0.01,
        kappa=0.5,
        normalize_y=True,
        random_state=0,
        allow_hyperparam_optimization=False,
        length_scale_bounds=(1e-9, 1e10),
        hopt_every=1,
        hopt_warmup=1,
    )
    return bo

def initialize_bo(bo: DiscreteBO, init_indices, lut: Dict[int, float], available: Set[int]) -> None:
    for i in init_indices:
        if i not in available:
            raise RuntimeError(f"Init index {i} not in evaluation table.")
    bo.initialize(init_indices, [lut[i] for i in init_indices])

# =========================================================
# 4A) WORKFLOW 1: FIXED-ITERATION BO (classic BO loop)
# =========================================================
# def run_fixed_bo(bo: DiscreteBO, evaluate_fn, n_iters: int = 150) -> pd.DataFrame:
#     hist = []
#     for it in range(1, n_iters + 1):
#         idxs, acqs = bo.suggest_next(top_k=1)
#         idx = int(idxs[0])
#         x = bo.Xc[idx - 1, :]
#         y = float(evaluate_fn(idx, x))
#         bo.tell(idx, y)
#         hist.append({"iter": it, "picked": idx, "y": y, "acq": float(acqs[0])})
#         print(f"[BO] iter {it:03d}: pick={idx} y={y:.6g} acq={acqs[0]:.6g}")
#     return pd.DataFrame(hist)

def run_fixed_bo(
    bo: DiscreteBO,
    evaluate_fn,
    *,
    n_iters: int,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    real_df: pd.DataFrame,
    fixed_indices: np.ndarray,
    eps: float,
    out_dir: Optional[str] = None,
    plot_secondary: Optional[str] = "ucb",   # "ucb" / "ei" / None
    n_init: int = 2,
) -> pd.DataFrame:
    hist_rows = []

    for it in range(1, n_iters + 1):
        idxs, acqs = bo.suggest_next(top_k=1)
        idx = int(idxs[0])
        x = bo.Xc[idx - 1, :]
        y = float(evaluate_fn(idx, x))
        bo.tell(idx, y)

        # If you want S_size in fixed BO, define it as -1 or compute separately
        row = log_step(
            bo=bo,
            step=it,
            idx_next=idx,
            y_measured=y,
            Z_index=Z_matrix.T,          # IMPORTANT: transpose for decode_any (n_vars, N)
            var_names=var_names,
            fixed_indices=fixed_indices,
            real_df=real_df,
            eps=eps,
            S_size=-1,
            save_plots_dir=out_dir,      # will save gp_step_XXX.png if not None
            plot_secondary=plot_secondary,
            n_init=n_init,
        )

        # Also store acquisition value
        row["acq"] = float(acqs[0])
        hist_rows.append(row)

        print(f"[BO] iter {it:03d}: pick={idx} y={y:.6g} acq={acqs[0]:.6g} RMSE={row['RMSE']:.6g} R2={row['R2']:.6g}")

    return pd.DataFrame(hist_rows)

# =========================================================
# 4B) WORKFLOW 2: NEAR-OPTIMAL SCREENING (until S empty)
# =========================================================

def run_near_optimal_screening(
    bo: DiscreteBO,
    evaluate_fn,
    *,
    eps: float,
    available: Set[int],
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    real_df: pd.DataFrame,
    fixed_indices: np.ndarray,
    out_dir: Optional[str] = None,
    max_steps: int = 300,
    kappa_screen_init: float = 3,
    policy_kappa_init: float = 3,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
) -> pd.DataFrame:
    history_rows = []

    for step in range(1, max_steps + 1):

        kappa_screen = kappa_screen_init * np.exp(-3.0 * step / max_steps)
        policy_kappa = policy_kappa_init * np.exp(-3.0 * step / max_steps)
        Xp = bo.compute_Xp_candidates(
            eps=eps,
            use_ucb=True,
            kappa=kappa_screen,
            available_1based=available
        )

        if len(Xp) == 0:
            print(f"[STOP] Xp empty at step {step}")
            break

        idx_next = bo.suggest_next_within_indices(Xp, policy="ucb", kappa=policy_kappa)
        x = bo.Xc[idx_next - 1, :]
        y = float(evaluate_fn(idx_next, x))
        bo.tell(idx_next, y)

        row = log_step(
            bo=bo,
            step=step,
            idx_next=idx_next,
            y_measured=y,
            Z_index=Z_matrix.T,
            var_names=var_names,
            fixed_indices=fixed_indices,
            real_df=real_df,
            eps=eps,
            Xp_size=len(Xp),
            save_plots_dir=out_dir,
            plot_secondary=plot_secondary,
            n_init=n_init,
        )
        history_rows.append(row)

        print(f"[STEP {step:03d}] pick={idx_next} y={y:.6g} best={row['best']:.6g} target={row['target']:.6g} |Xp|={len(Xp)} RMSE={row['RMSE']:.6g} Kappa = {kappa_screen:.6g}" )

    return pd.DataFrame(history_rows), kappa_screen

def summarize_near_optimal(bo: DiscreteBO, eps: float, available: Set[int], kappa_screen: float = 3.0):
    y_best = float(np.max(bo._y_obs))
    target = y_best - eps

    final_obs_df = pd.DataFrame({
        "index_1based": np.array(bo._chosen_idx) + 1,
        "y": bo._y_obs,
    })
    final_obs_df = final_obs_df[final_obs_df["y"] >= target].sort_values("y", ascending=False).reset_index(drop=True)

    Xp_pred = bo.compute_Xp_candidates(eps=eps, use_ucb=True, kappa=kappa_screen, available_1based=available)
    mu, std = bo.predict_all()
    final_unobs_df = pd.DataFrame({
        "index_1based": Xp_pred,
        "mu": mu[np.array(Xp_pred) - 1],
        "std": std[np.array(Xp_pred) - 1],
        "target": target,
    }).sort_values("mu", ascending=False).reset_index(drop=True)

    return final_obs_df, final_unobs_df

# =========================================================
# 5) MAIN
# =========================================================
def main():
    Input, Z_matrix = load_input_and_index(FILE_PATH)
    fixed_indices = load_fixed_indices(FILE_PATH, "Mid_Z_Testing")

    lut, available, real_df = load_eval_lut(RESULT_PATH, EVAL_SHEET)
    evaluate_fn = make_evaluate_fn_from_lut(lut)

    bo = build_bo(Input)
    initialize_bo(bo, read_index_list(FILE_PATH), lut, available)
    # initialize_bo(bo, INIT_LIST, lut, available)

    # print(read_index_list(FILE_PATH))
    # print(lut)
    var_names = ["Index", "Device", "Freq", "Turn"]



    # ---- choose one mode ----
    # hist_bo = run_fixed_bo(
    #     bo, evaluate_fn,
    #     n_iters=150,
    #     Z_matrix=Z_matrix,
    #     var_names=var_names,
    #     real_df=real_df,
    #     fixed_indices=fixed_indices,
    #     eps=EPS,
    #     out_dir=OUT_DIR,
    #     plot_secondary="ucb",
    #     n_init=2,
    # )
    # hist_bo.to_csv(os.path.join(OUT_DIR, "fixed_bo_history.csv"), index=False)

    # Or screening mode:
    hist_screen, kappa_screen = run_near_optimal_screening(
        bo, evaluate_fn,
        eps=EPS,
        available=available,
        Z_matrix=Z_matrix,
        var_names=var_names,
        real_df=real_df,
        fixed_indices=fixed_indices,
        out_dir=OUT_DIR,
        max_steps=500,
        kappa_screen_init=3,
        policy_kappa_init=3,
        plot_secondary="ucb",
        n_init=len(INIT_LIST),
        # n_init = len(read_index_list(FILE_PATH)),
    )
    hist_screen.to_csv(os.path.join(OUT_DIR, "screening_history.csv"), index=False)


    final_obs_df, final_unobs_df = summarize_near_optimal(bo, eps=EPS, available=available, kappa_screen=kappa_screen)
    print("\nConfirmed near-optimal (observed):")
    print(final_obs_df)
    print("\nPredicted near-optimal (unobserved):")
    print(final_unobs_df)

if __name__ == "__main__":
    main()