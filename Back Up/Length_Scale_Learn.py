from __future__ import annotations

from typing import List, Dict, Union, Literal, Tuple, Optional,Set,Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel

def learn_lengthscales_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel_type: str = "matern52",   # "matern52" or "rbf"
    n_restarts: int = 10,
    normalize_y: bool = True,
    standardize_X: bool = True,
    noise_level: float = None,  # set None to estimate noise
):
    """
    Learn ARD length-scale vector via GP marginal likelihood maximization.

    Parameters
    ----------
    X : (n_samples, n_features) array
    y : (n_samples,) or (n_samples, 1) array
    kernel_type : {"matern52","rbf"}
    n_restarts : number of optimizer restarts
    normalize_y : whether sklearn normalizes y internally
    standardize_X : recommended for ARD (puts all features on comparable scale)
    noise_level : if provided, fixes noise variance; if None, estimates noise

    Returns
    -------
    model : fitted GaussianProcessRegressor
    lengthscales : learned ARD length-scale vector (n_features,)
    scaler : fitted StandardScaler for X (or None)
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D: (n_samples, n_features).")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    scaler = None
    if standardize_X:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    n_features = X_fit.shape[1]

    # ARD: length_scale is a vector of size n_features
    if kernel_type.lower() == "matern52":
        base = Matern(length_scale=np.ones(n_features),
                      length_scale_bounds=(1e-3, 1e6),
                      nu=2.5)
    elif kernel_type.lower() == "rbf":
        base = RBF(length_scale=np.ones(n_features),
                   length_scale_bounds=(1e-3, 1e3))
    else:
        raise ValueError("kernel_type must be 'matern52' or 'rbf'.")

    # Signal variance + (optional) noise estimation
    k = C(1.0, (1e-6, 1e6)) * base
    if noise_level is None:
        # Estimate noise from data
        k += WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
    else:
        # Fix noise variance (no optimization for noise)
        k += WhiteKernel(noise_level=float(noise_level), noise_level_bounds="fixed")

    gp = GaussianProcessRegressor(
        kernel=k,
        alpha=0.0,              # keep 0 when using WhiteKernel for noise
        normalize_y=normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=0,
    )

    gp.fit(X_fit, y)

    # Extract learned length-scale vector
    fitted_kernel = gp.kernel_
    # kernel structure: (ConstantKernel * BaseKernel) + WhiteKernel
    # So we can access product k1 * k2 via .k1/.k2 and then base kernel under .k2
    # Depending on sklearn version, the sum/product nesting is stable as below.
    prod = fitted_kernel.k1      # ConstantKernel * base
    base_fitted = prod.k2        # Matern or RBF
    lengthscales = np.asarray(base_fitted.length_scale, dtype=float)

    return gp, lengthscales, scaler

def load_design_sheet(
    xlsx_path: str,
    sheet_name: str,
    *,
    value_name_row_excel: int = 3,  # row containing final column names (Value_Name row)
    data_start_row_excel: int = 4,  # first data row (Order=1 row)
) -> pd.DataFrame:
    """
    Robust loader for your layout:
      Row 1: Order / x1 / x2 / x3 ...
      Row 2: Input_Type ...
      Row 3: Value_Name row (actual desired column names)
      Row 4+: data
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    value_row = value_name_row_excel - 1
    data_row  = data_start_row_excel - 1

    # Use Value_Name row as column names
    cols = raw.iloc[value_row].tolist()

    df = raw.iloc[data_row:].copy()
    df.columns = cols

    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Clean: ensure Order is numeric if present
    if "Order" in df.columns:
        df["Order"] = pd.to_numeric(df["Order"], errors="coerce")

    return df

ClusterMethod = Literal["kmeans", "agglomerative"]
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
    Cluster Design parameters into N groups based on selected feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table containing Design Parameter rows.
    feature_cols : str or list[str]
        Feature column name(s) used for clustering. Example: "Rds(on)" or ["Rds(on)", "Coss"].
    n_groups : int
        Number of clusters (groups).
    index_col : str | None
        Column containing MOSFET IDs/indices to return (e.g., Excel 'Index').
        If None or missing, uses df.index.
    method : {"kmeans", "agglomerative"}
        Clustering algorithm.
    standardize : bool
        If True, standardize features before clustering (recommended).
    random_state : int
        Random seed (used by kmeans).

    Returns
    -------
    groups : dict[int, list[int]]
        Mapping cluster_id -> list of MOSFET indices (from index_col or df.index).
    labels : np.ndarray
        Cluster label for each row in the filtered dataframe order.

    Notes
    -----
    - Rows with missing values in selected features are dropped.
    - If you use many features with different units (Rds, Coss, etc.), standardize=True is strongly recommended.
    """

    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]

    if n_groups < 1:
        raise ValueError("n_groups must be >= 1.")
    if not all(col in df.columns for col in feature_cols):
        missing = [c for c in feature_cols if c not in df.columns]
        raise KeyError(f"Missing feature columns in df: {missing}")

    # Extract feature matrix, drop rows with NaNs in features
    work = df.copy()
    work = work.dropna(subset=feature_cols)

    if len(work) < n_groups:
        raise ValueError(f"Not enough rows ({len(work)}) to form {n_groups} groups after dropping NaNs.")

    X = work[feature_cols].to_numpy(dtype=float)

    # Standardize if requested (recommended when mixing units)
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X

    # Fit clustering
    if method == "kmeans":
        model = KMeans(n_clusters=n_groups, n_init=20, random_state=random_state)
        labels = model.fit_predict(X_fit)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_groups, linkage="ward" if standardize else "ward")
        labels = model.fit_predict(X_fit)
    else:
        raise ValueError("method must be 'kmeans' or 'agglomerative'.")

    # Decide which indices to return
    # if index_col is not None and index_col in work.columns:
    #     ids = work[index_col].to_numpy()
    # else:
    #     ids = work.index.to_numpy()
    if index_col not in work.columns:
        raise KeyError(f"index_col='{index_col}' not found. Columns: {list(work.columns)}")
    ids = work[index_col].to_numpy()

    # Build groups: cluster_id -> list of ids
    groups: Dict[int, List[int]] = {}
    for cid in range(n_groups):
        groups[cid] = [int(v) for v in ids[labels == cid]]

    return groups, labels
#
#

UniqMode = Literal["A", "B"]

def sample_group_combinations(
    groups_mos: Dict[int, List[Any]],
    groups_freq: Dict[int, List[Any]],
    groups_turn: Dict[int, List[Any]],
    *,
    s: int = 2,
    mode: UniqMode = "A",
    seed: int = 0,
    require_full: bool = True,
    id_base: Literal[0, 1] = 1,   # <-- NEW: expected base for returned IDs
) -> Dict[Tuple[int, int, int], List[Tuple[Any, Any, Any]]]:
    """
    Sample `s` tuples (mos_id, freq_id, turn_id) for each (cm, cf, ct).
    `id_base=1` will convert integer IDs that look 0-based to 1-based (add +1).
    """

    if s < 1:
        raise ValueError("s must be >= 1")

    rng = np.random.default_rng(seed)

    mos_keys = sorted(groups_mos.keys())
    freq_keys = sorted(groups_freq.keys())
    turn_keys = sorted(groups_turn.keys())

    def to_base(v: Any) -> Any:
        # Only shift ints (or numpy ints). Leave strings/device names unchanged.
        if isinstance(v, (int, np.integer)):
            return int(v) if id_base == 0 else int(v) + 1
        return v

    def draw_one(cm: int, cf: int, ct: int) -> Tuple[Any, Any, Any]:
        m = rng.choice(groups_mos[cm])
        f = rng.choice(groups_freq[cf])
        t = rng.choice(groups_turn[ct])
        return (to_base(m), to_base(f), to_base(t))

    out: Dict[Tuple[int, int, int], List[Tuple[Any, Any, Any]]] = {}

    for cm in mos_keys:
        for cf in freq_keys:
            for ct in turn_keys:
                combo = (cm, cf, ct)
                picked: List[Tuple[Any, Any, Any]] = []
                max_tries = 5000

                for _ in range(s):
                    if not picked:
                        picked.append(draw_one(cm, cf, ct))
                        continue

                    prev = picked[-1]
                    ok = False
                    tries = 0

                    while tries < max_tries and not ok:
                        cand = draw_one(cm, cf, ct)

                        if mode == "A":
                            ok = (cand not in picked)
                        elif mode == "B":
                            ok = (cand[0] != prev[0] and cand[1] != prev[1] and cand[2] != prev[2])
                            ok = ok and (cand not in picked)
                        else:
                            raise ValueError("mode must be 'A' or 'B'")

                        tries += 1

                    if not ok:
                        msg = f"Cannot draw sample #{len(picked)+1} for combo {combo} under mode={mode}."
                        if require_full:
                            raise RuntimeError(msg)
                        break

                    picked.append(cand)

                out[combo] = picked

    return out


def tuple_samples_to_flat_indices(
    samples: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    *,
    dim2: int,
    dim3: int,
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Convert tuple-based samples (i, j, k) into flattened global indices.

    Assumptions
    -----------
    - i, j, k are 1-based indices
    - dim2 = size of 2nd variable
    - dim3 = size of 3rd variable

    Mapping
    -------
    index = (i-1)*(dim2*dim3) + (j-1)*dim3 + k
    """

    flat_indices: Dict[Tuple[int, int, int], List[int]] = {}

    for combo_key, tuples in samples.items():
        idx_list: List[int] = []

        for (i, j, k) in tuples:
            idx = (i - 1) * (dim2 * dim3) + (j - 1) * dim3 + k
            idx_list.append(idx)

        flat_indices[combo_key] = idx_list

    return flat_indices



def extract_mid_z_index(file_path):
    """
    Extracts the 'index_1based' column from the 'Mid_Z_Testing' sheet
    in the specified Excel file.

    Args:
        file_path (str): The path to your .xlsx file.

    Returns:
        list: A list of values from the column, or None if error.
    """
    try:
        # Load the specific sheet 'Mid_Z_Testing'
        df = pd.read_excel(file_path, sheet_name='Mid_Z_Testing')

        # Check if the target column exists
        target_col = 'index_1based'
        if target_col in df.columns:
            # Convert the column data to a Python list
            data_list = df[target_col].tolist()
            return data_list
        else:
            print(f"Error: Column '{target_col}' not found in sheet 'Mid_Z_Testing'.")
            return None

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def load_feature_matrix(file_path):
    """
    Reads the 'Mid_Input_encoding' sheet and converts all data
    into a NumPy matrix (array) suitable for ML.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        numpy.ndarray: A 2D matrix where each row is a feature vector.
                       (Row 0 corresponds to Excel Row 2, aka Index 1)
    """
    try:
        # Load the specific sheet
        df = pd.read_excel(file_path, sheet_name='Mid_Input_encoding')

        # Ensure all data is numeric (coercing errors just in case)
        # This prevents headers or text artifacts from breaking the matrix
        df = df.apply(pd.to_numeric, errors='coerce')

        # Convert the DataFrame to a NumPy matrix
        # This strips the column names and leaves just the raw vectors
        feature_matrix = df.to_numpy()

        return feature_matrix

    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None


def build_training_X(file_path,indices_1based):
    """
    Combines the index extraction and matrix loading to create
    the final training input X.
    """
    # 1. Get the full feature matrix (The Library)
    full_matrix = load_feature_matrix(file_path)

    if indices_1based is None or full_matrix is None:
        print("Failed to build training data due to previous errors.")
        return None

    # 2. Convert 1-based indices to 0-based Python indices
    # We use np.array to make subtraction easy
    indices_0based = np.array(indices_1based) - 1

    # Validation: Ensure we don't try to access a row that doesn't exist
    max_idx = np.max(indices_0based)
    if max_idx >= full_matrix.shape[0]:
        print(f"Error: Index {max_idx + 1} exceeds matrix size {full_matrix.shape[0]}")
        return None

    # 3. Extract the specific rows to create X
    # This selects the rows from 'full_matrix' that match 'indices_0based'
    X_train = full_matrix[indices_0based]

    return X_train


def load_y_target(file_path, target_indices):
    """
    Extracts the 'Efficiency' values for the specific indices provided.

    Args:
        file_path (str): Path to the Excel file containing the 'data' sheet.
        target_indices (list): The list of integer indices (IDs) extracted
                               from the previous 'extract_mid_z_index' function.

    Returns:
        numpy.ndarray: A vector of Efficiency values matching the order of target_indices.
                       Shape: (n_samples, 1)
    """
    try:
        # 1. Load the 'data' sheet
        df = pd.read_excel(file_path, sheet_name='data')

        # 2. Identify the Index Column and Target Column
        # We assume Column A (index 0) is the ID, and we look for 'Efficiency'
        index_col_name = df.columns[0]  # Gets name of Column A (e.g., 'index_1ba...')
        target_col_name = 'Efficiency'

        if target_col_name not in df.columns:
            print(f"Error: Column '{target_col_name}' not found in 'data' sheet.")
            return None

        # 3. Set the Index of the DataFrame to match your IDs
        # This allows us to look up data by ID number (e.g., ID 25) instantly
        df_indexed = df.set_index(index_col_name)

        # 4. Extract only the rows that match your target_indices
        # .loc[list] preserves the order of the list you provide!
        # If your list is [10, 1, 5], the result will be [Eff_10, Eff_1, Eff_5]
        try:
            selected_rows = df_indexed.loc[target_indices]
        except KeyError as e:
            print(f"Error: One or more indices from your list were not found in the data file: {e}")
            return None

        # 5. Extract Efficiency and convert to (N, 1) NumPy array
        y_vector = selected_rows[target_col_name].to_numpy()
        y_vector = y_vector.reshape(-1, 1)  # Force shape to be (N, 1)

        return y_vector
    except Exception as e:
        print(f"Error loading y_target: {e}")
        return None

# ------------------- Example usage -------------------
if __name__ == "__main__":

    # --- Example Usage ---
    # Replace 'your_file.xlsx' with the actual name of your file
    features_file = r'C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx'
    results_file = r'C:\Users\Tungtan\PycharmProjects\AutomationDesign\Data\20260108_32 efficiency_results.xlsx'

    # Suppose you already loaded your Excel sheet into df
    df_mos = pd.read_excel(features_file, sheet_name="Mid_Normalization")

    groups_mos, labels_mos = cluster_design(df_mos, feature_cols="Rds(on)", n_groups=4)

    for cid, idx_list in groups_mos.items():
        print(f"Group {cid} (size={len(idx_list)}): {idx_list}")
    pass
    df_freq_turn = load_design_sheet(
        features_file,
        sheet_name="Input",
        value_name_row_excel=3,
        data_start_row_excel=4
    )

    df_freq_turn.columns = df_freq_turn.columns.astype(str).str.strip()


    # Now df_design columns are: ["Order", "Q1_value", "SwitchingFrequency", "PriTurn_value", ...]
    groups_freq, labels_freq = cluster_design(
        df_freq_turn,
        feature_cols="SwitchingFrequency",
        n_groups=4,
        index_col="Value_Name",  # IMPORTANT: your sheet uses Order, not Index
        method="kmeans",
        standardize=True,
    )
    print(groups_freq)
    print(df_freq_turn.head(5))
    print(df_freq_turn.dtypes)

    groups_turn, labels_turn = cluster_design(
        df_freq_turn,
        feature_cols="PriTurn_value",
        n_groups=4,
        index_col="Value_Name",  # IMPORTANT: your sheet uses Order, not Index
        method="kmeans",
        standardize=True,
    )
    print(groups_turn)

    # Example: take 2 samples per group-combination, strict component-wise difference
    samples = sample_group_combinations(
        groups_mos,
        groups_freq,
        groups_turn,
        s=1,
        mode="A",  # or "A"
        require_full=True,
        id_base =0,
        seed = 2,
    )

    print("Total combinations:", len(samples))  # should be 64 if groups have keys 0..3
    print(samples)  # e.g., [(mos_id, freq_id, turn_id), (mos_id, freq_id, turn_id)]

    #indices_1based = extract_mid_z_index(features_file)
    indices_Dic = tuple_samples_to_flat_indices(samples, dim2=8, dim3=12)
    indices_1based = [v for values in indices_Dic.values() for v in values]

    print(indices_1based)

    df = pd.DataFrame({"index_1based": indices_1based})
    with pd.ExcelWriter(
            features_file,
            engine="openpyxl",
            mode="a",  # append mode
            if_sheet_exists="replace"  # replace ONLY this sheet
    ) as writer:
        df.to_excel(writer, sheet_name="Mid_Z_Testing", index=False)

    if indices_1based is not None:
        # 2. Get X (from your previous code)
        # Note: build_training_X internally handles the -1 conversion for matrix rows
        X_train = build_training_X(features_file,indices_1based)

        # 3. Get y (NEW function)
        # We pass the original 1-based indices because the Excel file uses 1-based IDs in Col A
        y_train = load_y_target(results_file, indices_1based)

        if X_train is not None and y_train is not None:
            print(f"X Shape: {X_train.shape}")  # Should be (N, 7)
            print(f"y Shape: {y_train.shape}")  # Should be (N, 1)

            # 4. Run GP Learning
            gp_model, lengthscales, scaler = learn_lengthscales_gp(
                X=X_train,
                y=y_train,
                kernel_type="matern52",
                standardize_X = True,
                normalize_y = True,
                noise_level = 0.0005**2
            )

            print("\nSuccess! Learned Lengthscales:")
            print(lengthscales)


