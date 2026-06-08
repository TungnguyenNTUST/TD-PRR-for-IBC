"""
Build the BO input feature matrix from the UserProvidedDataFile Excel.

Edit the CONFIG block below, then run:
    python bo_build_input_matrix.py

Encoding methods
----------------
pra  (Parameter / Raw feature encoding)  ← recommended
    Each device type contributes its raw normalised feature columns directly.
    Calls: build_feature_matrix_and_write_excel

dic  (Dictionary distance encoding)
    Each device type contributes distance-to-dictionary columns.
    Calls: build_z_encoding_matrix_and_write_excel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from excel_data_handler import (
    add_block_id_column,
    build_feature_matrix_and_write_excel,
    build_normalized_features_by_q,
    build_normalized_input_block_and_write_excel,
    build_requested_params_per_device_vector,
    build_z_encoding_matrix_and_write_excel,
    capture_excel_variables,
    capture_requested_params_and_weights_from_device_sheet,
    clear_destination_sheet,
    extract_params_from_chara_library,
    normalize_params_blockwise,
    parse_excel_config,
    write_design_option_index_matrix,
    _sorted_vector_names_by_x,
)


def build_input_matrix(
    file_path: str,
    lib_root: str,
    *,
    encoding: str = "dic",
    sample_ratio: float = 0.075,
) -> np.ndarray:
    """
    Full pipeline: read Excel config → extract device params → normalise →
    build index matrix → encode into GP feature matrix → save Mid_Z_Testing.

    Returns
    -------
    Input : np.ndarray  shape (D, N) — ready for DiscreteBO(Input=...)
    """
    encoding = encoding.lower()
    if encoding not in ("dic", "pra"):
        raise ValueError("encoding must be 'dic' or 'pra'.")

    # --- Step 1: capture variable vectors from Input sheet ---
    res = capture_excel_variables(
        xlsx_path=file_path,
        sheet_name="Input",
        include_blanks=False,
    )

    # --- Step 2: clear intermediate sheets ---
    clear_destination_sheet(dest_xlsx_path=file_path, dest_sheet_name="Mid_Normalization")
    clear_destination_sheet(dest_xlsx_path=file_path, dest_sheet_name="Mid_Input_encoding")

    # --- Step 3: read device parameter selections ---
    q_to = capture_requested_params_and_weights_from_device_sheet(
        root_xlsx_path=file_path,
        device_sheet_name="Device",
    )
    q_to_params = {q: v["params"] for q, v in q_to.items()}
    vector_to_params = build_requested_params_per_device_vector(res, q_to_params)
    features_by_q = build_normalized_features_by_q(q_to_params)

    # --- Step 4: extract params from component library ---
    for vec in res.vectors.values():
        if str(vec.input_type).strip().lower() != "device":
            continue
        device_list = [str(v).strip() for v in vec.values if v]
        requested_params = vector_to_params.get(vec.name, [])
        if not requested_params:
            print(f"  Skip {vec.name}: no params selected in Device sheet.")
            continue
        extract_params_from_chara_library(
            available_devices=device_list,
            requested_devices=None,
            requested_params=requested_params,
            library_root_dir=lib_root,
            dest_xlsx_path=file_path,
            dest_sheet_name="Mid_Normalization",
        )

    # --- Step 5: normalise blocks ---
    normalize_params_blockwise(
        file_path=file_path,
        sheet_name="Mid_Normalization",
        round_digits=6,
        zero_mean_atol=0.0,
    )
    add_block_id_column(
        file_path=file_path,
        sheet_name="Mid_Normalization",
        header_row=1,
        index_header="Index",
        block_header="BlockID",
        reset_value=1,
    )

    # --- Step 6: build index matrix ---
    cfg = parse_excel_config(file_path=file_path, sheet_name="Optimization")
    summary = write_design_option_index_matrix(
        capture_result=res,
        output_xlsx_path=file_path,
        sheet_name="Mid_Input_Indexing",
        create_new_workbook=False,
        include_order_col=True,
        max_rows=2_000_000,
        z_index_mode="numpy",
        write_excel=True,
    )
    Z_index = summary["Z_index"]

    # --- Step 7: encode feature matrix ---
    Z_enc_blocks = []
    start_col_index = 1
    vector_names = _sorted_vector_names_by_x(res)
    x_order = [res.vectors[vn].x_name for vn in vector_names]

    for _v, vec in res.vectors.items():
        input_type = vec.input_type.strip().lower()

        if input_type == "device":
            q = vec.value_name.replace("_value", "")
            if q not in features_by_q:
                continue
            try:
                device_var_col = x_order.index(vec.x_name)
            except ValueError:
                raise RuntimeError(
                    f"x_name '{vec.x_name}' not found in x_order {x_order}"
                )
            FEATURES = features_by_q[q]

            if encoding == "dic":
                DIC_LIST_NO = cfg.dictionary_q_values[q]
                thetas = tuple(q_to[q]["weights"])
                device_block = int(q[1:])
                Z_enc = build_z_encoding_matrix_and_write_excel(
                    start_col_index=start_col_index,
                    Z_Index=Z_index,
                    file_path=file_path,
                    features_sheet="Mid_Normalization",
                    feature_cols=FEATURES,
                    dic_list_no=DIC_LIST_NO,
                    weights=thetas,
                    device_var_col=device_var_col,
                    dict_block=device_block,
                    dest_xlsx_path=file_path,
                    dest_sheet="Mid_Input_encoding",
                    include_header=True,
                    row_offset=1,
                )
            else:  # pra
                Z_enc = build_feature_matrix_and_write_excel(
                    start_col_index=start_col_index,
                    Z_Index=Z_index,
                    file_path=file_path,
                    features_sheet="Mid_Normalization",
                    feature_cols=FEATURES,
                    device_var_col=device_var_col,
                    no_col="Index",
                    block_col="BlockID",
                    dest_xlsx_path=file_path,
                    dest_sheet="Mid_Input_encoding",
                    include_header=True,
                    row_offset=1,
                )

            start_col_index += Z_enc.shape[1]
            Z_enc_blocks.append(Z_enc)

        elif input_type in {"parameter", "magnetic"}:
            device_var_col = x_order.index(vec.x_name)
            Z_norm = build_normalized_input_block_and_write_excel(
                start_col_index=start_col_index,
                Z_Index=Z_index,
                vec=vec,
                device_var_col=device_var_col,
                dest_xlsx_path=file_path,
                dest_sheet="Mid_Input_encoding",
                include_header=True,
                row_offset=1,
                write_excel=True,
            )
            start_col_index += Z_norm.shape[1]
            Z_enc_blocks.append(Z_norm)

    if not Z_enc_blocks:
        raise ValueError("No encoding blocks were generated — check Input/Device sheet config.")

    nrows = Z_enc_blocks[0].shape[0]
    if any(b.shape[0] != nrows for b in Z_enc_blocks):
        raise ValueError("Row count mismatch across Z_enc blocks.")

    Z_enc_all = np.hstack(Z_enc_blocks)
    print(f"[build_input_matrix] Z_enc_all shape: {Z_enc_all.shape}")
    Input = Z_enc_all.T  # (D, N)

    # --- Step 8: save random sample as Mid_Z_Testing ---
    n_total = Z_enc_all.shape[0]
    n_sample = max(1, int(n_total * sample_ratio))
    fixed_indices = np.random.choice(np.arange(1, n_total + 1), size=n_sample, replace=False)
    df_fixed = pd.DataFrame({"index_1based": fixed_indices})
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df_fixed.to_excel(writer, sheet_name="Mid_Z_Testing", index=False)
    print(f"[build_input_matrix] Saved {n_sample} test indices to Mid_Z_Testing.")

    return Input


def main() -> None:
    # =========================================================
    # CONFIG — paths are defined in core/paths.py
    # =========================================================
    from core.paths import FILE_PATH, LIB_ROOT
    ENCODING     = "pra"    # "pra" = raw parameter features (recommended)
                            # "dic" = dictionary distance encoding
    SAMPLE_RATIO = 0.025    # fraction of all design options saved to Mid_Z_Testing
                            # 0.025 = 2.5%  (~1,440 of 57,600)
    # =========================================================

    Input = build_input_matrix(
        file_path=FILE_PATH,
        lib_root=LIB_ROOT,
        encoding=ENCODING,
        sample_ratio=SAMPLE_RATIO,
    )
    print(f"[done] Input matrix shape: {Input.shape}")



def _main_cli() -> None:
    """Alternative CLI entry-point (optional — use main() above for normal runs)."""
    import argparse
    parser = argparse.ArgumentParser(description="Build BO input feature matrix from Excel.")
    parser.add_argument("--file_path",    required=True, help="Path to UserProvidedDataFile.xlsx")
    parser.add_argument("--lib_root",     required=True, help="Path to component library root folder")
    parser.add_argument("--encoding",     default="pra", choices=["dic", "pra"],
                        help="Encoding method: 'pra' (raw features) or 'dic' (dictionary distance)")
    parser.add_argument("--sample_ratio", default=0.025, type=float,
                        help="Fraction of design options to save as Mid_Z_Testing (default: 0.025)")
    args = parser.parse_args()

    Input = build_input_matrix(
        file_path=args.file_path,
        lib_root=args.lib_root,
        encoding=args.encoding,
        sample_ratio=args.sample_ratio,
    )
    print(f"[done] Input matrix shape: {Input.shape}")


if __name__ == "__main__":
    main()
