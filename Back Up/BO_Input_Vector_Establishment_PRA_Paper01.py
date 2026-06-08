from Data_Handling_Source import (
    capture_excel_variables,
    clear_destination_sheet,
    extract_params_from_chara_library,
    capture_requested_params_and_weights_from_device_sheet,
    build_requested_params_per_device_vector,
    build_normalized_features_by_q,
    normalize_params_blockwise,
    parse_excel_config,
    write_design_option_index_matrix,
    add_block_id_column,
    build_z_encoding_matrix_and_write_excel,
    _sorted_vector_names_by_x,
    build_normalized_input_block_and_write_excel,
    build_feature_matrix_and_write_excel,
)
from Bayesian_Optimization_Source import (
    DiscreteBO,
)

from Bayesian_Optimziation_Helper import (
    decode_any,
)


from Bayesian_Optimization_Plotting import(
    manual_bo_with_decode_and_plot,
)
import pandas as pd
import numpy as np
import os
from typing import Optional

if __name__ == "__main__":
    FILE_PATH = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx"
    LIB_ROOT = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Paper_01_Component_Library_IBC"

    Input_Matrix_Re_Build  = 1
    if Input_Matrix_Re_Build == 1:
        res = capture_excel_variables(
            xlsx_path=FILE_PATH,
            sheet_name="Input",     # or omit to use active sheet
            include_blanks=False     # typically what you want
        )
        ##Test Code For Build the Mid-Normalization Sheet
        clear_destination_sheet(
            dest_xlsx_path=FILE_PATH,
            dest_sheet_name="Mid_Normalization"
        )
        clear_destination_sheet(
            dest_xlsx_path=FILE_PATH,
            dest_sheet_name="Mid_Input_encoding"
        )
        # Read parameter-selection matrix from device sheet
        q_to = capture_requested_params_and_weights_from_device_sheet(
            root_xlsx_path=FILE_PATH,
            device_sheet_name="Device",
        )
        q_to_params = {q: v["params"] for q, v in q_to.items()}
        # print("Print result from Device Sheet:", q_to)
        # print("Print result from Device Sheet original:",q_to_params)
        # For each device vector, get its requested params
        vector_to_params = build_requested_params_per_device_vector(res, q_to_params)
        features_by_q = build_normalized_features_by_q(q_to_params)
        # print("features_by_q:",features_by_q)
        for vec in res.vectors.values():
            if str(vec.input_type).strip().lower() == "device":
                device_list = [str(v).strip() for v in vec.values if v]

                # IMPORTANT: choose requested_params based on this vector name
                requested_params = vector_to_params.get(vec.name, [])  # vec.name like "x1_Q1_value"

                # Skip if no params selected for this Q
                if not requested_params:
                    print(f"Skip {vec.name}: no requested params selected in device sheet.")
                    continue
                extract_params_from_chara_library(
                    available_devices=device_list,
                    requested_devices=None,  # means "use all available_devices"
                    requested_params=requested_params,
                    library_root_dir=LIB_ROOT,
                    dest_xlsx_path=FILE_PATH,
                    dest_sheet_name="Mid_Normalization",
                    # recommended: pass a label to separate blocks (see section 3)
                    # block_title=vec.name
                )
        means_by_block = normalize_params_blockwise(
            file_path=FILE_PATH,
            sheet_name="Mid_Normalization",
            round_digits=6,
            zero_mean_atol=0.0
        )
        #Block ID added and preparation of input encoding
        add_block_id_column(
            file_path=FILE_PATH,
            sheet_name="Mid_Normalization",   # use your actual sheet name
            header_row=1,
            index_header="Index",
            block_header="BlockID",
            reset_value=1,
        )
        cfg = parse_excel_config(
            file_path=FILE_PATH,
            sheet_name = "Optimization")
        # Generate full design-option index matrix and export
        summary = write_design_option_index_matrix(
            capture_result=res,
            output_xlsx_path=FILE_PATH,
            sheet_name="Mid_Input_Indexing",
            create_new_workbook=False,
            include_order_col= True,
            max_rows=2_000_000,  # adjust if needed
            z_index_mode= "numpy",
            write_excel = True,
        )
        Z_index = summary["Z_index"]

        df_feat = pd.read_excel(FILE_PATH, sheet_name="Mid_Normalization")

        Z_enc_blocks = []         # list of (N_options, Kq) blocks
        Z_enc_headers = []        # optional: list of header names per column (for debug/ML)
        start_col_index = 1
        vector_names = _sorted_vector_names_by_x(res)
        x_order = [res.vectors[vn].x_name for vn in vector_names]
        print("Vector name", vector_names)
        print("X_oder:", x_order)
        print("res.vectors.items():",res.vectors.items())
        for v, vec in res.vectors.items():
            input_type = vec.input_type.strip().lower()
            if input_type == "device":
                q = vec.value_name.replace("_value", "")
                if q not in features_by_q:
                    continue
                # ---- identify which Z_Index column to use ----
                try:
                    # Return the order/index of "vex.x_name" in the list of "x_order".
                    device_var_col = x_order.index(vec.x_name)
                    print("device_var_col:", device_var_col)
                except ValueError:
                    raise RuntimeError(
                        f"x_name '{vec.x_name}' not found in Z_Index ordering {x_order}"
                    )
                FEATURES = features_by_q[q]
                print("FEATURES:",FEATURES)
                # DIC_LIST_NO = cfg.dictionary_q_values[q]
                # thetas = tuple(q_to[q]["weights"])
                # print(thetas)
                device_block = int(q[1:])

                print("start_col_index:",start_col_index)
                Z_enc = build_feature_matrix_and_write_excel(
                    start_col_index=start_col_index,  # start at column T (far right to avoid existing content)
                    Z_Index=Z_index,
                    file_path=FILE_PATH,
                    features_sheet="Mid_Normalization",
                    feature_cols=FEATURES,
                    device_var_col=device_var_col,
                    no_col="Index",
                    block_col="BlockID",
                    # dict_block=device_block,  # use same block as device
                    dest_xlsx_path=FILE_PATH,
                    dest_sheet="Mid_Input_encoding",
                    include_header=True,
                    row_offset=1,
                )
                start_col_index += Z_enc.shape[1]
                # ---- accumulate in Python (do NOT overwrite) ----
                Z_enc_blocks.append(Z_enc)

                # optional headers so you know which column is which
                # Z_enc_headers.extend([f"{q}_dist_to_{int(d)}" for d in DIC_LIST_NO])
            elif input_type in {"parameter", "magnetic"}:
                # find the correct Z_Index column for this variable
                device_var_col = x_order.index(vec.x_name)

                Z_norm = build_normalized_input_block_and_write_excel(
                    start_col_index=start_col_index,
                    Z_Index=Z_index,
                    vec=vec,
                    device_var_col=device_var_col,
                    dest_xlsx_path=FILE_PATH,
                    dest_sheet="Mid_Input_encoding",
                    include_header=True,
                    row_offset=1,
                    write_excel=True,
                )
                # append into your Python blocks (same pattern as device encoding)
                Z_enc_blocks.append(Z_norm)
                Z_enc_headers.append(f"{vec.x_name}_norm")

                # advance Excel column pointer by 1 column
                start_col_index += Z_norm.shape[1]  # = 1

        # ---- final combined matrix (N_options, total_cols) ----
        if not Z_enc_blocks:
            raise ValueError("No device blocks were generated.")
        # safety: all blocks must have same number of rows
        nrows = Z_enc_blocks[0].shape[0]
        if any(b.shape[0] != nrows for b in Z_enc_blocks):
            raise ValueError("Row count mismatch across Z_enc blocks; check Z_Index orientation / selection.")
        Z_enc_all = np.hstack(Z_enc_blocks)
        print("Z_enc_all shape:", Z_enc_all.shape)   # (23040, 11) in your example
        # print("First 20 rows of Z_enc_all:")
        # print(Z_enc_all[:20, :])
        Input = Z_enc_all.T

    if Input_Matrix_Re_Build == 1:

        fixed_indices = np.random.choice(np.arange(1, Z_enc_all.shape[0]+1), size=int(Z_enc_all.shape[0]*0.025), replace=False)

        df_fixed_indices = pd.DataFrame({
            "index_1based": fixed_indices
        })
        with pd.ExcelWriter(
                FILE_PATH,
                engine="openpyxl",
                mode="a",  # append to existing file
                if_sheet_exists="replace"  # replace the blank sheet
        ) as writer:
            df_fixed_indices.to_excel(
                writer,
                sheet_name="Mid_Z_Testing",
                index=False
            )
    else:
        df_fixed_indices = pd.read_excel(
            FILE_PATH,
            sheet_name="Mid_Z_Testing",
            header=0
        )
        fixed_indices = df_fixed_indices.iloc[:, 0].to_numpy(dtype=int)
        print("fixed_indices:", fixed_indices)


