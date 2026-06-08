"""
Length-scale learning for the IBC BO GP kernel.
Duplicate clustering utilities removed — now imported from core.clustering_utils.
"""
from __future__ import annotations

from typing import List, Dict, Union, Literal, Tuple, Optional, Set, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel

from core.clustering_utils import (
    load_design_sheet,
    cluster_design,
    sample_group_combinations_n,
    tuple_samples_to_flat_indices_n,
)


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


def build_training_X(file_path, indices_1based):
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
    from core.paths import FILE_PATH, RESULT_PATH
    features_file = FILE_PATH
    results_file = RESULT_PATH  # update in core/paths.py if a different results file is needed

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

    # Example: take 1 sample per group-combination using N-var sampler
    samples = sample_group_combinations_n(
        [groups_mos, groups_freq, groups_turn],
        s=1,
        mode="A",
        require_full=True,
        id_base=0,
        seed=2,
    )

    print("Total combinations:", len(samples))  # should be 64 if groups have keys 0..3
    print(samples)  # e.g., [(mos_id, freq_id, turn_id), (mos_id, freq_id, turn_id)]

    indices_dic = tuple_samples_to_flat_indices_n(samples, dims=[12, 8, 8])
    indices_1based = [v for values in indices_dic.values() for v in values]

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
        X_train = build_training_X(features_file, indices_1based)

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
                standardize_X=True,
                normalize_y=True,
                noise_level=0.0005**2
            )

            print("\nSuccess! Learned Lengthscales:")
            print(lengthscales)
