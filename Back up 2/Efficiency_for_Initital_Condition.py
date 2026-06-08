from Loss import efficiency_one_desin_option
import pandas as pd
from pathlib import Path
from core.paths import FILE_PATH as Excel_File

Test_1_OR_2_OR_3 = "Mid_Z_Init_T1"
Test_02 = "Mid_Z_Init_T2"

def match_indices_to_variables(
    excel_path: str | Path,
    *,
    index_sheet: str,
    mapping_sheet: str,
    index_col: str = "indices_1based",
    order_col: str = "Order",
    var_cols=("x1", "x2", "x3", "x4", "x5"),
):
    """
    Match global indices to variable vectors.

    Returns:
        DataFrame with columns:
        [indices_1based, x1, x2, x3, x4, x5]
    """

    # Read sheets
    df_idx = pd.read_excel(excel_path, sheet_name=index_sheet)
    df_map = pd.read_excel(excel_path, sheet_name=mapping_sheet)

    # Clean column names
    df_idx.columns = df_idx.columns.str.strip()
    df_map.columns = df_map.columns.str.strip()

    # Check columns
    if index_col not in df_idx.columns:
        raise KeyError(f"{index_col} not in {index_sheet}")

    if order_col not in df_map.columns:
        raise KeyError(f"{order_col} not in {mapping_sheet}")

    for c in var_cols:
        if c not in df_map.columns:
            raise KeyError(f"{c} not in {mapping_sheet}")

    # Merge (lookup)
    df_out = df_idx.merge(
        df_map[[order_col, *var_cols]],
        left_on=index_col,
        right_on=order_col,
        how="left",
    )

    # Drop duplicate order column
    df_out = df_out.drop(columns=[order_col])

    return df_out


if __name__ == "__main__":
    df_result  = match_indices_to_variables(excel_path=Excel_File,
                            index_sheet=Test_1_OR_2_OR_3,
                            mapping_sheet="Mid_Input_Indexing",
                            index_col= "indices_1based",
                            order_col = "Order",
                            var_cols = ("x1", "x2", "x3", "x4", "x5")
                            )

    arr = df_result[["x1","x2","x3","x4","x5"]].to_numpy()


    df_input = pd.read_excel(Excel_File, sheet_name="Input")
    df_input = df_input.dropna(subset=["Order"])
    q1_table = dict(zip(df_input["Order"], df_input["x1"]))
    q2_table = dict(zip(df_input["Order"], df_input["x2"]))
    q3_table = dict(zip(df_input["Order"], df_input["x3"]))

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

    rows = []
    for i in range(arr.shape[0]):
        print(i,i,i,i,i,i,i,i,i)
        devices = get_devices_from_x(
            x1=int(arr[i, 0]),
            x2=int(arr[i, 1]),
            x3=int(arr[i, 2]),
            q1_map=q1_table,
            q2_map=q2_table,
            q3_map=q3_table,
        )
        print("Efficiency in case of: MOS=", devices["MOS"],"CORE =",devices["CORE"],",and DIO=",devices["DIO"])
        print("Fsw_index:",int(arr[i, 3]),"and Ind_index:",int(arr[i, 4]))
        print("Switching frequency and inductance as below shown")
        eff_list = efficiency_one_desin_option(MOS_PN=devices["MOS"],DIO_PN=devices["DIO"],
                                             Core_PN=devices["CORE"],fsw_index=int(arr[i, 3])
                                            ,ind_index = int(arr[i, 4]))
        print("RESULT---------------------------------------------------------------------------")
        print(eff_list)
        print("NEXT-----------------------------------------------------------------------------")
        # Ensure it's exactly 7 values
        if len(eff_list) != 7:
            raise ValueError(f"Expected 7 efficiency values, got {len(eff_list)} at i={i}")

        rows.append({
            "MOS": devices["MOS"],
            "CORE": devices["CORE"],
            "DIO": devices["DIO"],
            "fsw_index": int(arr[i, 3]),
            "ind_index": int(arr[i, 4]),

            # 7 columns:
            "eff_1": eff_list[0],
            "eff_2": eff_list[1],
            "eff_3": eff_list[2],
            "eff_4": eff_list[3],
            "eff_5": eff_list[4],
            "eff_6": eff_list[5],
            "eff_7": eff_list[6],})


    df_out = pd.DataFrame(rows)
    with pd.ExcelWriter(Excel_File, engine="openpyxl", mode="a",if_sheet_exists="overlay") as writer:
        df_out.to_excel(writer, sheet_name=Test_1_OR_2_OR_3, index=False,startcol=1)

    print(f"Saved to: {Excel_File} ({Test_1_OR_2_OR_3})")
