import pandas as pd

def load_eval_lut(result_xlsx_path: str, sheet_name_final: str = "data"):
    df = pd.read_excel(result_xlsx_path, sheet_name=sheet_name_final)
    df["index_1based"] = df["index_1based"].astype(int)
    df["Efficiency"] = pd.to_numeric(df["Efficiency"], errors="coerce")
    df = df.dropna(subset=["Efficiency"])

    lut = dict(zip(df["index_1based"].to_numpy(), df["Efficiency"].to_numpy()))
    available = set(lut.keys())

    # keep a real_df for metrics
    real_df = df[["index_1based", "Efficiency"]].copy()
    return lut, available, real_df


def make_evaluate_fn_from_lut(lut: dict):
    def evaluate_fn(index_1based: int, x):
        idx = int(index_1based)
        if idx not in lut:
            raise KeyError(f"Index {idx} not found in LUT.")
        return float(lut[idx])
    return evaluate_fn
