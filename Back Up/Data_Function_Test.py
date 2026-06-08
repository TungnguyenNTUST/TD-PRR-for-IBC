from Data_Handling_Source import (
    capture_excel_variables,
    build_variable_report,
    clear_destination_sheet,
    extract_params_from_chara_library,
    capture_requested_params_and_weights_from_device_sheet,
    build_requested_params_per_device_vector,
    build_normalized_features_by_q,
    normalize_params_blockwise,
    parse_excel_config,
    write_design_option_index_matrix,
    device_distance_by_no,
    add_block_id_column,
    build_index_to_block_map,
    build_z_encoding_matrix_and_write_excel,
    _sorted_vector_names_by_x,
    build_normalized_input_block_and_write_excel,
)
from math import prod
if __name__ == "__main__":

    res = capture_excel_variables(
        xlsx_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        sheet_name="Input",     # or omit to use active sheet
        include_blanks=False     # typically what you want
    )
    total_product = prod(res.sizes.values())
    print(build_variable_report(res))

    # Access a specific vector:
    x1_vec = res.vectors["x1_Q1_value"].size
    size_x1 = res.sizes["x1_Q1_value"]
    device_vars = res.by_input_type.get("Parameter", [])
    print(res.input_types.keys())
    print(res)

    ##Test Code For Build the Mid-Normalization Sheet
    clear_destination_sheet(
        dest_xlsx_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        dest_sheet_name="Mid_Normalization"
    )

    clear_destination_sheet(
        dest_xlsx_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        dest_sheet_name="Mid_Input_encoding"
    )

    # 1) Read parameter-selection matrix from device sheet
    q_to = capture_requested_params_and_weights_from_device_sheet(
        root_xlsx_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        device_sheet_name="Device",
    )
    q_to_params = {q: v["params"] for q, v in q_to.items()}
    print(q_to_params)

    # 2) For each device vector, get its requested params
    vector_to_params = build_requested_params_per_device_vector(res, q_to_params)
    print(vector_to_params)

    features_by_q = build_normalized_features_by_q(q_to_params)
    print(features_by_q)


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
                library_root_dir=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Component Library",
                dest_xlsx_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
                dest_sheet_name="Mid_Normalization",
                # recommended: pass a label to separate blocks (see section 3)
                # block_title=vec.name
            )

    # means = normalize_all_params_in_sheet(
    #     file_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
    #     sheet_name="Mid_Normalization",
    #     round_digits=6,
    #     zero_mean_atol=0.0
    # )
    #
    # print("Normalization means used:")
    # for k, v in means.items():
    #     print(k, "->", v)


    means_by_block = normalize_params_blockwise(
        file_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        sheet_name="Mid_Normalization",
        round_digits=6,
        zero_mean_atol=0.0
    )

    #Block ID added and preparation of input encoding
    add_block_id_column(
        file_path=r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx",
        sheet_name="Mid_Normalization",   # use your actual sheet name
        header_row=1,
        index_header="Index",
        block_header="BlockID",
        reset_value=1,
    )
    print("BlockID added.")

    for (r0, r1), means in means_by_block.items():
        print(f"Block {r0}-{r1}:")
        for p, m in means.items():
            print(" ", p, "mean =", m)

    print(means_by_block)

    file_path = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx"
    sheet_name = "Optimization"
    cfg = parse_excel_config(
        file_path=file_path,
        sheet_name = sheet_name)

    print(cfg)
    # print(cfg.init_condition_setting)  # "Manual" / "Automated"
    # print(cfg.xinit_indices)  # {"Xinit1": 1, "Xinit2": 1920, ...}
    #
    # print(cfg.dictionary_setting)  # "Manual" / "Automated"
    # print(cfg.dictionary_q_values.keys())  # which Q columns are actually used
    # print(cfg.dictionary_q_values["Q2_value"])
    # from openpyxl import load_workbook
    #

    # 2) Generate full design-option index matrix and export
    summary = write_design_option_index_matrix(
        capture_result=res,
        output_xlsx_path=file_path,
        sheet_name="Mid_Input_Indexing",
        create_new_workbook=False,
        include_order_col= True,
        max_rows=2_000_000,  # adjust if needed
        z_index_mode= "numpy",
        write_excel = True,
    )

    Z_index = summary["Z_index"]

    print(type(Z_index))  # <class 'list'>
    print(len(Z_index))  # number of design options
    print(Z_index[:10])  # first 10 rows
    print(Z_index[-1])  # last row

    import pandas as pd
    df = pd.read_excel(file_path, sheet_name="Mid_Normalization")
    print(df.columns.tolist())
    print(df.head(3))


    print("Columns:", df.columns.tolist())
    print(df[["Index", "Device", "BlockID"]].head(25))
    print("Unique BlockIDs:", sorted(pd.to_numeric(df["BlockID"], errors="coerce").dropna().unique().tolist()))

    df2 = df.copy()
    df2["Index"] = pd.to_numeric(df2["Index"], errors="coerce")
    df2["BlockID"] = pd.to_numeric(df2["BlockID"], errors="coerce")

    # take block 1, first two indices in that block
    blk1 = df2[df2["BlockID"] == 1].dropna(subset=["Index"])
    idx_list = sorted(blk1["Index"].astype(int).unique().tolist())

    # no_x = idx_list[0]
    # no_y = idx_list[1]
    # block_x = 2
    # block_y = 2
    #
    # print("Testing with:", (block_x, no_x), (block_y, no_y))

    # FEATURES = features_by_q["Q2"]
    #
    # res1 = device_distance_by_no(
    #     file_path=file_path,
    #     sheet_name="Mid_Normalization",
    #     no_x=no_x,
    #     no_y=no_y,
    #     block_x=block_x,
    #     block_y=block_y,
    #     feature_cols=FEATURES,
    #     weights=None,
    # )

    # print("distance:", res1["distance"])
    # print("Device_x:", res1["Device_x"])
    # print("Device_y:", res1["Device_y"])
    # print("block_x, block_y:", res1["block_x"], res1["block_y"])
    # print("deltas:", res1["deltas"])


    import pandas as pd

    file_path = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx"
    sheet_name = "Mid_Normalization"

    df_feat = pd.read_excel(file_path, sheet_name=sheet_name)
    m = build_index_to_block_map(df_feat, no_col="Index", block_col="BlockID")

    # print("Type:", type(m))
    # print("Num unique Index keys:", len(m))

    Z_enc_blocks = []         # list of (N_options, Kq) blocks
    Z_enc_headers = []        # optional: list of header names per column (for debug/ML)
    i = 0
    start_col_index = 1

    vector_names = _sorted_vector_names_by_x(res)
    # Extract x-name order: ["x1", "x2", "x3", "x4", ...]
    x_order = [res.vectors[vn].x_name for vn in vector_names]
    print("Z_Index x-order:", x_order)

    for v, vec in res.vectors.items():
        input_type = vec.input_type.strip().lower()
        if input_type == "device":
            q = vec.value_name.replace("_value", "")

            if q not in features_by_q:
                continue
            # ---- identify which Z_Index column to use ----
            try:
                device_var_col = x_order.index(vec.x_name)
            except ValueError:
                raise RuntimeError(
                    f"x_name '{vec.x_name}' not found in Z_Index ordering {x_order}"
                )
            i = i + 1
            FEATURES = features_by_q[q]
            print(f"Device #{i}: Q={q}")
            print("  All features:", FEATURES)

            DIC_LIST_NO = cfg.dictionary_q_values[q]
            print(DIC_LIST_NO)
            # thetas = (1.0,) * len(q_to_params[q])
            thetas = tuple(q_to[q]["weights"])
            print(thetas)

            device_block = int(q[1:])
            Z_enc = build_z_encoding_matrix_and_write_excel(
                start_col_index=start_col_index,  # start at column T (far right to avoid existing content)
                Z_Index=Z_index,
                file_path=file_path,
                features_sheet=sheet_name,
                feature_cols=FEATURES,
                dic_list_no=DIC_LIST_NO,
                weights=thetas,
                device_var_col=device_var_col,
                dict_block=device_block,  # use same block as device
                dest_xlsx_path=file_path,
                dest_sheet="Mid_Input_encoding",
                include_header=True,
                row_offset=1,
            )
            start_col_index += Z_enc.shape[1]
            # ---- accumulate in Python (do NOT overwrite) ----
            Z_enc_blocks.append(Z_enc)

            # optional headers so you know which column is which
            Z_enc_headers.extend([f"{q}_dist_to_{int(d)}" for d in DIC_LIST_NO])

            print("Z_enc shape:", Z_enc.shape)
            print("First row:", Z_enc[0])


        # elif input_type == "magnetic":
        #     print("I will go to magnetic sheet")
        #
        # elif input_type == "parameter":
        #     print("do normalization")
        elif input_type in {"parameter", "magnetic"}:
            # find the correct Z_Index column for this variable
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
    import numpy as np
    Z_enc_all = np.hstack(Z_enc_blocks)
    print("Z_enc_all shape:", Z_enc_all.shape)   # (23040, 11) in your example
    print("First 20 rows of Z_enc_all:")
    print(Z_enc_all[:20, :])