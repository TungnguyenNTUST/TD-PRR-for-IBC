import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Set, Sequence, List
from Bayesian_Optimization_Source import DiscreteBO
# from BO_evaluation import load_eval_lut, make_evaluate_fn_from_lut
from BO_reporting import log_step
from Loss import efficiency_one_desin_option
# =========================================================
# 0) CONFIG
# =========================================================
FILE_PATH = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx"
RESULT_PATH = FILE_PATH
OUT_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Results\Paper01"
OUT_MU_STD = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Results\Paper01\Mu_and_Std.xlsx"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_SHEET = "Mid_Input_encoding"
INDEX_SHEET = "Mid_Input_Indexing"
EVAL_SHEET = "Test Case 03"
INIT_SHEET = "Mid_Z_Init_T2"

INIT1, INIT2, INIT3 = 1355, 1920, 1
INIT_LIST = [INIT1,INIT2]
# EPS = 0.0000000001
EPS = 0.001

# =========================================================
# 1) DATA LOADING
# =========================================================

def read_index_list(
    excel_path_int: str,
    sheet_name_int: str = "Mid_Z_Init_T2",
    column_name_int: str = "indices_1based",
) -> List[int]:
    """
    Read the index (a column) in an Excel sheet.
    Provide the sheet name and column header name

    Returns:
        the index value stored in the specific column
        type of returned data is a list
    """

    df = pd.read_excel(excel_path_int, sheet_name= sheet_name_int, usecols=[column_name_int])
    return df[column_name_int].dropna().astype(int).tolist()

def load_input_and_index(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the data (all column) in two  Excel sheets.
    Returns:
        1. The input matrix which already normalize
        2. The index matrix of all design options
    """
    df_enc = pd.read_excel(file_path, sheet_name=INPUT_SHEET, header=0)
    df_idx = pd.read_excel(file_path, sheet_name=INDEX_SHEET, header=0)
    Input_int = df_enc.to_numpy(dtype=float).T          # (D, N)
    Z_matrix_int = df_idx.to_numpy(dtype=int)           # whatever your indexing shape is
    return Input_int, Z_matrix_int

# =========================================================
# 2) EVALUATION LUT
# =========================================================

def load_eval_lut(result_xlsx_path: str, sheet_name: str):
    # Read file Excel
    df = pd.read_excel(result_xlsx_path, sheet_name=sheet_name)

    # fist column = index
    index_col = df.columns[0]          # "indices_1based"
    # final column is objective
    obj_col = df.columns[-1]           # "eff_7"

    # Standardization
    df[index_col] = df[index_col].astype(int)
    df[obj_col] = pd.to_numeric(df[obj_col], errors="coerce")

    # Remove the row invalid objective
    df = df.dropna(subset=[obj_col])

    # Create LUT: index → objective
    lut = dict(zip(df[index_col].to_numpy(), df[obj_col].to_numpy()))

    # index set without issue
    available = set(lut.keys())

    # DataFrame clean for debug/log
    real_df = df[[index_col, obj_col]].copy()

    return lut, available, real_df


def make_evaluate_fn_from_lut(lut: Dict[int, float]):
    def evaluate_fn(index_1based: int, x: np.ndarray) -> float:
        idx = int(index_1based)
        return float(lut[idx])   # let KeyError raise if missing
    return evaluate_fn

def load_fixed_indices(file_path: str, sheet_name: str = "Mid_Z_Testing") -> np.ndarray:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
    return df.iloc[:, 0].to_numpy(dtype=int)

import pandas as pd

def load_index_eff(file_path: str, sheet_name: str = 0):
    # Đọc file Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Lấy cột đầu tiên (index) và cột cuối (objective)
    index_col = df.columns[0]      # "indices_1based"
    eff_col = df.columns[-1]       # "eff_7"

    # Chuyển kiểu dữ liệu
    df[index_col] = df[index_col].astype(int)
    df[eff_col] = pd.to_numeric(df[eff_col], errors="coerce")

    # Loại bỏ dòng lỗi
    df = df.dropna(subset=[eff_col])

    # Trả về numpy array
    indices = df[index_col].to_numpy(dtype=int)
    eff = df[eff_col].to_numpy(dtype=float)

    return indices, eff, df

def get_devices_from_x(
    x1: int,
    x2: int,
    x3: int,
    *,
    q1_map: dict,
    q2_map: dict,
    q3_map: dict,
):
    """
    Return device PN from x1,x2,x3 indices
    """

    try:
        q1 = q1_map[x1]
        q2 = q2_map[x2]
        q3 = q3_map[x3]
    except KeyError as e:
        raise ValueError(f"Index not found: {e}")

    return {
        "MOS": q1,
        "CORE": q2,
        "DIO": q3,
    }


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
        kappa=2.0,
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
    *,
    eps: float,
    Z_matrix: np.ndarray,
    var_names: Sequence[str],
    eff_obs: np.ndarray,
    indices_obs: np.ndarray,
    out_dir: Optional[str] = None,
    max_steps: int = 4000,
    kappa_screen_init: float = 2,
    policy_kappa_init: float = 2,
    plot_secondary: Optional[str] = "ucb",
    n_init: int = 2,
    q1_table_in_fn:Dict,
    q2_table_in_fn:Dict,
    q3_table_in_fn:Dict,
)-> pd.DataFrame:
    history_rows = []

    for step in range(1, max_steps + 1):

        kappa_screen = kappa_screen_init #* np.exp(-3.0 * step / max_steps)
        policy_kappa = policy_kappa_init #* np.exp(-3.0 * step / max_steps)

        Xp, eff_avg, eff_std = bo.compute_Xp_candidates(
            eps=eps,
            use_ucb=True,
            kappa=kappa_screen,
            available_1based=None,
        )

        if len(Xp) == 0:
        # if len(Xp) == 0 and float(np.mean(eff_std)) < 5e-7:
            print(f"[STOP] Xp empty at step {step}")

            # (Tùy chọn) Kiểm tra cùng độ dài
            if len(eff_avg) != len(eff_std):
                raise ValueError(f"Độ dài 2 mảng khác nhau: {len(eff_avg)} vs {len(eff_std)}")

            # 2) Tạo DataFrame 2 cột
            df = pd.DataFrame({"Mu": eff_avg, "Std": eff_std})

            # 3) Ghi ra Excel (tạo file mới)
            df.to_excel(OUT_MU_STD, index=False, engine="openpyxl")
            print(f"Đã lưu: {OUT_MU_STD}")
            break

        idx_next  = bo.suggest_next_within_indices(Xp, policy="ucb", kappa=policy_kappa)


        # idx_next, acq_val = bo.suggest_next(top_k=1)
        # idx_next =int(idx_next[0])
        matches = (indices_obs == idx_next)
        if matches.any():
            y= eff_obs[matches][0]
        else:
            devices_in_fn = get_devices_from_x(
                x1=int(Z_matrix[idx_next-1, 1]),
                x2=int(Z_matrix[idx_next-1, 2]),
                x3=int(Z_matrix[idx_next-1, 3]),
                q1_map=q1_table_in_fn,
                q2_map=q2_table_in_fn,
                q3_map=q3_table_in_fn,
            )
            # print("Next index of design option:",idx_next)
            # print("Efficiency in case of: MOS=", devices_in_fn["MOS"], "CORE =", devices_in_fn["CORE"], ",and DIO=", devices_in_fn["DIO"])
            # print("Fsw_index:", int(Z_matrix[idx_next-1, 4]), "and Ind_index:", int(Z_matrix[idx_next-1, 5]))
            y_list  = efficiency_one_desin_option(MOS_PN=devices_in_fn["MOS"], DIO_PN=devices_in_fn["DIO"],
                                                   Core_PN=devices_in_fn["CORE"], fsw_index=int(Z_matrix[idx_next-1, 4])
                                                   , ind_index=int(Z_matrix[idx_next-1, 5]))
            y =float(y_list[6])

        bo.tell(idx_next, y)
        # print("Space Xp:",len(Xp))
        row = log_step(
            bo=bo,
            step=step,
            idx_next=idx_next,
            y_measured=y,
            Z_index=Z_matrix.T,
            var_names=var_names,
            eps=eps,
            Xp_size=len(Xp),
            save_plots_dir=out_dir,
            plot_secondary=plot_secondary,
            n_init=n_init,
        )
        history_rows.append(row)

        print(f"[STEP {step:03d}] pick={idx_next} y={y:.6g} best={row['best']:.6g} target={row['target']:.6g} |Xp|={len(Xp)} Kappa = {kappa_screen:.6g}" )

    return pd.DataFrame(history_rows)


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
    # print("\nConfirmed near-optimal (observed):")
    print(final_obs_df)
    # print("\nPredicted near-optimal (unobserved):")
    # print(final_unobs_df)

if __name__ == "__main__":
    # main()

    df_input = pd.read_excel(FILE_PATH, sheet_name="Input")
    df_input = df_input.dropna(subset=["Order"])
    q1_table = dict(zip(df_input["Order"], df_input["x1"]))
    q2_table = dict(zip(df_input["Order"], df_input["x2"]))
    q3_table = dict(zip(df_input["Order"], df_input["x3"]))

    # debug = read_index_list(FILE_PATH)
    # print(debug)

    Input, Z_matrix = load_input_and_index(FILE_PATH)
    # print("The size of input of BO matrix:",Input.shape)
    # print("The size of Index matrix of BO matrix:", Z_matrix.shape)

    lut_inital, available_inital, real_df_inital = load_eval_lut(FILE_PATH, INIT_SHEET)
    # print("Available:",available_inital)
    bo = build_bo(Input)
    initialize_bo(bo, read_index_list(FILE_PATH), lut_inital, available_inital)
    var_names = ["Index", "Mos", "Core","Dio", "Freq", "Inductance"]

    indices_observed, eff_observed, df_observed = load_index_eff(FILE_PATH,sheet_name="Observed")

    hist_screen = run_near_optimal_screening(
        bo,
        eps=EPS,
        Z_matrix=Z_matrix,
        var_names=var_names,
        eff_obs=eff_observed,
        indices_obs=indices_observed,
        out_dir=OUT_DIR,
        max_steps=6000,
        kappa_screen_init=3,
        policy_kappa_init=3,
        plot_secondary="ucb",
        n_init = len(read_index_list(FILE_PATH)),
        q1_table_in_fn=q1_table,
        q2_table_in_fn=q2_table,
        q3_table_in_fn=q3_table,
    )
    hist_screen.to_csv(os.path.join(OUT_DIR, "screening_history.csv"), index=False)


