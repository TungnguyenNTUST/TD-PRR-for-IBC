
import numpy as np
import os
import pandas as pd
from aesim.simba import ProjectRepository, ThermalDataType,ThermalDataSemiconductorType,ThermalComputationMethodType
from aesim.simba import IV_T, EI_VT
from pathlib import Path
from typing import Dict, List, Sequence, Union, Tuple, Optional
import matplotlib.pyplot as plt
import re


ROOT_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Simulation File"
Project_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Simulation File\Interleaved_Boost_Converter.jsimba"
TARGET_DESIGN = "IBC"
LIB_ROOT_MOS = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Paper_01_Component_Library_IBC"
LIB_ROOT_DIO = LIB_ROOT_MOS
Excel_File = r'C:\Users\Tungtan\PycharmProjects\AutomationDesign\UserProvidedDataFile.xlsx'

# witching cycle =====
n_cycles = 10           # number of switching  cycle
Load_list = {"100%": [36,0.05],"75%":[48,0.53],"50%":[72,0.21],"30%":[120,0.12],"20%":[180,0.05],"10%":[360,0.04]}
Inductance_list = {1:380e-6, 2:430e-6, 3:480e-6, 4:530e-6, 5:580e-6, 6:630e-6, 7:680e-6, 8:730e-6, 9:780e-6,10:830e-6}
Frequency_list = {1:20e3, 2:30e3, 3:40e3, 4:50e3, 5:60e3, 6:70e3, 7:80e3, 8:90e3, 9:100e3, 10:110e3}
MLT = { "KAM184-075A":157.56e-3,
        "KAM185-060A":136.25e-3,
        "KAM200-060A":126.85e-3,
        "KAM226-060A":149.00e-3,
        "KPH200-060A":126.85e-3,
        "KPH226-060A":149.00e-3,
        "KH184-060A-H":157.56e-3,
        "KH200-060A-H":126.85e-3,
        }


# PN_MOS = "IMZC120R078M2HXKSA1"
# MNF_MOS = "Wolfspeed"
# PN_DIO = "C4D30120D"
# MNF_DIO = "Wolfspeed"
# Core_PN = "KAM184-075A"

def efficiency_one_desin_option(MOS_PN:str, DIO_PN:str,Core_PN:str,fsw_index:int, ind_index:int):
    efficiencies = []
    eff_ECE = 0
    fsw = Frequency_list[fsw_index]
    inductance_value = Inductance_list[ind_index]

    PN_MOS = MOS_PN
    MNF_MOS = "Wolfspeed"
    PN_DIO = DIO_PN
    MNF_DIO = "Wolfspeed"
    Core_PN = Core_PN

    # Define the resistor i,v map
    POWER_MAP = {
        "R1": {
            "v": "R1 - Voltage",
            "i": "R1 - Current",
        },
        "R2": {
            "v": "R2 - Voltage",
            "i": "R2 - Current",
        },
        "R3": {
            "v": "R3 - Voltage",
            "i": "R3 - Current",
        },
        "R4": {
            "v": "R4 - Voltage",
            "i": "R4 - Current",
        },
        "R5": {
            "v": "R5 - Voltage",
            "i": "R5 - Current",
        },
        "R6": {
            "v": "R6 - Voltage",
            "i": "R6 - Current",
        },
        "R7": {
            "v": "R7 - Voltage",
            "i": "R7 - Current",
        },
        "R8": {
            "v": "R8 - Voltage",
            "i": "R8 - Current",
        },
        "R9": {
            "v": "R9 - Voltage",
            "i": "R9 - Current",
        },
        "R10": {
            "v": "R10 - Voltage",
            "i": "R10 - Current",
        },
    }
    # *************************************************************************************
    # *************************************************************************************
    # The magnetic handling code
    import math
    MU0 = 4.0 * math.pi * 1e-7  # H/m

    def get_number_before_A(pn: str) -> int:
        """
        Extract the number immediately before 'A' in PN.

        Example:
          'KH184-060A-H' -> 60
          'KAM184-075A'  -> 75
        """
        m = re.search(r'(\d+)A', pn)

        if not m:
            raise ValueError(f"No number before 'A' found in: {pn}")

        return int(m.group(1))

    def load_core_params_dict(
            xlsx_path: Union[str, Path],
            *,
            sheet_name: str = 0,
            pn_col: str = "Device",
            param_cols: Sequence[str] = ("Ae", "le", "k1", "k2", "anpha", "beta", "b", "c"),
            drop_if_any_missing: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Read an Excel sheet like your screenshot and return:
            { part_number: [Ae, le, k1, k2, anpha, beta, b, c] }

        Logic:
        - Normalize column names (strip spaces)
        - Keep rows where Device exists AND at least one param is non-empty
          (this naturally isolates the "core" block)
        - Convert params to numeric when possible
        """

        xlsx_path = Path(xlsx_path)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"Excel not found: {xlsx_path}")

        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]

        if pn_col not in df.columns:
            raise KeyError(f"'{pn_col}' not found. Available columns: {list(df.columns)}")

        missing = [c for c in param_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing param columns: {missing}. Available columns: {list(df.columns)}")

        # Ensure PN strings
        pn_series = df[pn_col].astype(str).str.strip()
        pn_ok = pn_series.notna() & (pn_series != "") & (pn_series.str.lower() != "nan")

        # Convert parameter columns to numeric if possible
        df_params = df.loc[:, list(param_cols)].apply(pd.to_numeric, errors="coerce")

        # Select rows that actually have core params filled
        if drop_if_any_missing:
            row_ok = pn_ok & df_params.notna().all(axis=1)
        else:
            row_ok = pn_ok & df_params.notna().any(axis=1)

        df_core = df.loc[row_ok, [pn_col] + list(param_cols)].copy()
        df_core[pn_col] = df_core[pn_col].astype(str).str.strip()

        # Build dict
        out: Dict[str, List[float]] = {}
        for _, r in df_core.iterrows():
            pn = r[pn_col]
            vals = []
            for c in param_cols:
                v = pd.to_numeric(r[c], errors="coerce")
                vals.append(float(v) if pd.notna(v) else float("nan"))
            out[pn] = vals

        return out

    def solve_N_from_L(
            *,
            L: float,  # target inductance [H]
            Ae: float,  # effective core area [m^2]
            le: float,  # effective magnetic path length [m]
            I: float,  # current used for incremental/permeability model [A]
            mu_r0: float = 60.0,  # low-field relative permeability (your "100")
            beta: float = 0.0,  # beta parameter
            c: float = 1.0,  # exponent
            N_min: float = 5,
            N_max: float = 500.0,
            tol: float = 1e-10,
            max_iter: int = 200,
    ) -> Tuple[float, int]:
        """
        Solve for N in:
          L = MU0*(Ae/le) * N^2 * mu_r(H)
          mu_r(H) = mu_r0 / (1 + beta*H^c)
          H = N*I/le

        Returns
        -------
        (N_float, N_int_nearest)

        Notes
        -----
        - Ae, le must be in meters^2 and meters.
        - If beta == 0, closed-form sqrt solution is used.
        - Uses bisection with automatic bracketing (very robust).
        """

        if L <= 0:
            raise ValueError("L must be > 0")
        if Ae <= 0 or le <= 0:
            raise ValueError("Ae and le must be > 0")
        if mu_r0 <= 0:
            raise ValueError("mu_r0 must be > 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        if c <= 0:
            raise ValueError("c must be > 0")
        if N_min <= 0 or N_max <= N_min:
            raise ValueError("Invalid N_min/N_max")

        K = MU0 * (Ae / le)  # [H] scaling

        # ---- Closed form if permeability is constant ----
        if beta == 0.0 or I == 0.0:
            N = math.sqrt(L / (K * mu_r0))
            N_int = max(1, int(round(N)))
            return N, N_int

        def mu_r_of_N(N: float) -> float:
            H = (N * I) / (le * 79.58)
            return mu_r0 / (1.0 + beta * (H ** c))

        def f(N: float) -> float:
            # f(N) = predicted_L(N) - target_L
            return K * (N ** 2) * mu_r_of_N(N) - L + 30e-6  # We allow the real inductance lower than expected 50uH

        # ---- Bracket the root ----
        a, b = N_min, N_max
        fa, fb = f(a), f(b)

        # Expand b if needed (up to a reasonable cap) to ensure sign change
        expand = 0
        while fa * fb > 0 and expand < 30:
            b *= 2.0
            fb = f(b)
            expand += 1

        if fa * fb > 0:
            raise RuntimeError(
                "Failed to bracket a root for N. "
                "Try increasing N_max (or check units/parameters)."
            )

        # ---- Bisection ----
        for _ in range(max_iter):
            m = 0.5 * (a + b)
            fm = f(m)

            if abs(fm) < tol:
                N = m
                N_int = max(1, int(round(N)))
                return N, N_int

            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm

            if abs(b - a) < 1e-12 * max(1.0, abs(m)):
                break

        N = 0.5 * (a + b)
        N_int = max(1, int(round(N)))
        return N, N_int

    def compute_L_from_NI(
            *,
            N: float,
            I: float,
            Ae: float,
            le: float,
            mu_r0: float,
            beta: float,
            c: float,
    ) -> float:
        """
        Compute inductance L for given N and I using nonlinear mu(H) model.

        Returns:
            L in Henry
        """

        if N <= 0:
            raise ValueError("N must be > 0")
        if Ae <= 0 or le <= 0:
            raise ValueError("Ae and le must be > 0")

        # Base factor
        K = MU0 * (Ae / le)

        # Field (convert to Oe like your solver)
        H = (N * I) / (le * 79.58)

        # Relative permeability
        mu_r = mu_r0 / (1.0 + beta * (H ** c))

        # Inductance
        L = K * (N ** 2) * mu_r

        return L

    import math

    def calc_Rdc(
            *,
            N: int,
            MLT: float,
            d_m: float,
            lead_per_turn_m: float = 0.0,
            rho: float = 1.939e-8,  # Ohm·m (copper ~100°C in your pic)
    ) -> float:
        """
        Calculate DC resistance of toroid winding.

        Parameters
        ----------
        N : int
            Number of turns

        MLT_mm : float
            Mean Length per Turn [mm]

        d_m : float
            Copper diameter [m] (bare conductor)

        lead_per_turn_m : float
            Extra lead length per turn [m]

        rho : float
            Copper resistivity [Ohm·m]

        Returns
        -------
        Rdc : float
            DC resistance [Ohm]
        """

        # Total wire length [m]
        wire_length = N * MLT + N * lead_per_turn_m

        # Copper cross-section [m^2]
        area = math.pi * d_m ** 2 / 4.0

        # DC resistance
        Rdc = rho * wire_length / area

        return Rdc

    def cal_flux_density_in_gauss(
            *,
            N: float,
            I: float,
            Iavg: float,
            le: float,
            mu_r0: float,
            beta: float,
            c: float,
    ) -> float:
        """
        Compute flux_density for given N and I using nonlinear mu(H) model.

        Returns:
            L in Henry
        """

        if N <= 0:
            raise ValueError("N must be > 0")

        # Field (convert to Oe like your solver)
        H_Oe = (N * Iavg) / (le * 79.58)
        H_A_per_m = (N * I) / le

        # Relative permeability
        mu_r = mu_r0 / (1.0 + beta * (H_Oe ** c))

        # Flux density - Gauss
        B_gauss = H_A_per_m * mu_r * MU0 * 1e4

        return B_gauss

    def core_loss_from_B_f(
            *,
            k1: float, k2: float, anpha: float, beta: float,
            B_kG: float,  # flux density [kGauss]
            f_kHz: float,  # frequency [kHz]
            Ae_m2: float,  # effective area [m^2]
            le_m: float,  # effective path length [m]
    ) -> float:
        """
        Core loss using:
            Pv = B^anpha * (k1*f + k2*f^beta)
        Units:
            Pv: mW/cm^3
            B:  kGauss
            f:  kHz
            V = Ae*le in m^3
        Returns:
            (Pv_W_per_m3, Pcore_W)
        """

        if B_kG <= 0:
            raise ValueError("B_kG must be > 0")
        if f_kHz <= 0:
            raise ValueError("f_kHz must be > 0")
        if Ae_m2 <= 0 or le_m <= 0:
            raise ValueError("Ae_m2 and le_m must be > 0")

        # loss density [mW/cm^3]
        Pv_mW_cm3 = (B_kG ** anpha) * (k1 * f_kHz + k2 * (f_kHz ** beta))

        # core volume [m^3]
        V_m3 = Ae_m2 * le_m

        # total core loss [W]
        Pcore_W = Pv_mW_cm3 * 1e3 * V_m3

        # convert to W

        return Pcore_W

    # Using above function to calculate the number of turn
    core_dict = load_core_params_dict(Excel_File, sheet_name="Mid_Normalization")
    # print(core_dict[Core_PN])
    # Params (ensure numeric)
    b = float(core_dict[Core_PN][6])
    c = float(core_dict[Core_PN][7])
    Ae = 3 * float(core_dict[Core_PN][0]) * 1e-4  # m^2, two cores stacked together.
    le = float(core_dict[Core_PN][1]) * 1e-2  # m
    N_float, N_int = solve_N_from_L(
        L=inductance_value,
        Ae=Ae,
        le=le,
        I=16.67,  # current at which you want incremental L
        mu_r0=get_number_before_A(Core_PN),
        beta=b,
        c=c,
    )

    # print("Inductor Number of Turn:", N_int)

    Rdc_wire = calc_Rdc(N=N_int, MLT=MLT[Core_PN], d_m=2e-3, lead_per_turn_m=5e-3) * 1.2
    # print("wire dc resistance:", Rdc_wire)

    # ***********************************************************************************
    # ***********************************************************************************
    def csv_to_iv_points(path: str):
        df = pd.read_csv(path, header=None)
        i = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

        m = np.isfinite(i) & np.isfinite(v)
        i, v = i[m], v[m]

        # sort by current
        idx = np.argsort(i)
        i, v = i[idx], v[idx]

        # drop duplicate currents
        ui, uidx = np.unique(i, return_index=True)
        i, v = ui, v[uidx]

        # # (optional) remove exact origin to avoid UI singularities
        # if len(i) > 0 and i[0] == 0.0 and v[0] == 0.0:
        #     i, v = i[1:], v[1:]

        pts = [[float(ii), float(vv)] for ii, vv in zip(i, v)]
        return pts

    def load_eon_file(base_dir):
        """
        Try to load Eon data from Eon_25.csv first, then Eon_30.csv.
        Return: (points, temperature)
        """
        path_150 = os.path.join(base_dir, "Eon_150.csv")
        path_175 = os.path.join(base_dir, "Eon_175.csv")

        if os.path.isfile(path_150):
            return csv_to_iv_points(path_150), 150.0

        if os.path.isfile(path_175):
            return csv_to_iv_points(path_175), 175.0

        raise FileNotFoundError(
            f"No Eon file found in {base_dir} (Eon_150.csv or Eon_175.csv)"
        )

    def load_eoff_file(base_dir):
        """
        Try to load Eon data from Eon_25.csv first, then Eon_30.csv.
        Return: (points, temperature)
        """
        path_150 = os.path.join(base_dir, "Eoff_150.csv")
        path_175 = os.path.join(base_dir, "Eoff_175.csv")

        if os.path.isfile(path_150):
            return csv_to_iv_points(path_150), 150.0

        if os.path.isfile(path_175):
            return csv_to_iv_points(path_175), 175.0

        raise FileNotFoundError(
            f"No Eoff file found in {base_dir} (Eoff_150.csv or Eoff_175.csv)"
        )

    def load_ifvf_file(base_dir):
        """
        Try to load Eon data from Eon_25.csv first, then Eon_30.csv.
        Return: (points, temperature)
        """
        path_150 = os.path.join(base_dir, "Idiode_150.csv")
        path_175 = os.path.join(base_dir, "Idiode_175.csv")

        if os.path.isfile(path_150):
            return csv_to_iv_points(path_150), 150.0

        if os.path.isfile(path_175):
            return csv_to_iv_points(path_175), 175.0

        raise FileNotFoundError(
            f"No Idiode file found in {base_dir} (Idiode_150.csv or Idiode_175.csv)"
        )

    def load_ecap_file(base_dir):
        """
        Try to load Eon data from Eon_25.csv first, then Eon_30.csv.
        Return: (points, temperature)
        """
        path_150 = os.path.join(base_dir, "Ecap_150.csv")
        path_175 = os.path.join(base_dir, "Ecap_175.csv")

        if os.path.isfile(path_150):
            return csv_to_iv_points(path_150), 150.0

        if os.path.isfile(path_175):
            return csv_to_iv_points(path_175), 175.0

        raise FileNotFoundError(
            f"No Ecap file found in {base_dir} (Ecap_150.csv or Ecap_175.csv)"
        )

    def Thermal_data_assignment():

        project = ProjectRepository(Project_DIR)
        design_main = project.GetDesignByName(TARGET_DESIGN)
        job_main = design_main.TransientAnalysis.NewJob()
        MOSFET_Q1 = design_main.Circuit.GetDeviceByName('Q1')
        MOSFET_Q2 = design_main.Circuit.GetDeviceByName('Q2')
        DIODE_D1A = design_main.Circuit.GetDeviceByName('D1A')
        DIODE_D1B = design_main.Circuit.GetDeviceByName('D1B')
        DIODE_D2A = design_main.Circuit.GetDeviceByName('D2A')
        DIODE_D2B = design_main.Circuit.GetDeviceByName('D2B')

        MOSFET_Q1.InitialTemperature = 30
        MOSFET_Q2.InitialTemperature = 30
        DIODE_D1A.InitialTemperature = 30
        DIODE_D2A.InitialTemperature = 30
        DIODE_D1B.InitialTemperature = 30
        DIODE_D2B.InitialTemperature = 30

        thermal_mos = MOSFET_Q1.ThermalData
        thermal_dio = DIODE_D1A.ThermalData

        if PN_MOS in os.listdir(LIB_ROOT_MOS):
            thermal_mos.Name = PN_MOS
            thermal_mos.PartNumber = PN_MOS
            thermal_mos.Manufacturer = MNF_MOS
            thermal_mos.ThermalImpedanceType = ThermalDataType.FosterThermalNetworkType
            thermal_mos.SemiconductorType = ThermalDataSemiconductorType.MosfetThermalDataSemiconductorType
            thermal_path = os.path.join(LIB_ROOT_MOS, PN_MOS, "Thermal")
            df_mos_thermal_impedance = pd.read_excel(os.path.join(thermal_path, "foster_params.xlsx"))
            tau_mos = df_mos_thermal_impedance["tau_s"].to_numpy()
            Rk_mos = df_mos_thermal_impedance["Rk_C_per_W"].to_numpy()
            thermal_mos.SetThermalImpedanceData([[float(r), float(c)] for r, c in zip(Rk_mos, tau_mos)])

            thermal_mos.ConductionLosses.Clear()
            thermal_mos.TurnOnLosses.Clear()
            thermal_mos.TurnOffLosses.Clear()
            iv_25 = IV_T()
            iv_175 = IV_T()
            iv_25.Temperature = 25.0
            iv_25.IV = csv_to_iv_points(os.path.join(thermal_path, "ID25.csv"))
            iv_25.NumberOfPoints = len(iv_25.IV)
            iv_175.Temperature = 175.0
            iv_175.IV = csv_to_iv_points(os.path.join(thermal_path, "ID175.csv"))
            iv_175.NumberOfPoints = len(iv_175.IV)
            thermal_mos.ConductionLosses.Add(iv_25)
            thermal_mos.ConductionLosses.Add(iv_175)
            thermal_mos.ConductionLossComputationMethod = ThermalComputationMethodType.LookUpTableComputationMethodType
            thermal_mos.ConductionLossesGateDependency = False

            ei_on_800_25 = EI_VT()
            ei_on_800_175or150 = EI_VT()
            ei_on_0_25 = EI_VT()
            ei_on_0_175or150 = EI_VT()
            # print("EI_VT attributes:")
            # for name in dir(ei_on_800_25):
            #     if not name.startswith("_"):
            #         print(name)

            ei_on_800_25.EI = csv_to_iv_points(os.path.join(thermal_path, "Eon_25.csv"))
            ei_on_800_25.Temperature = 25
            ei_on_800_25.Voltage = 800
            ei_on_800_25.NumberOfPoints = len(ei_on_800_25.EI)

            ei_on_0_25.EI = [[x, 0.0] for x, _ in ei_on_800_25.EI]
            ei_on_0_25.Temperature = 25
            ei_on_0_25.Voltage = 0
            ei_on_0_25.NumberOfPoints = len(ei_on_0_25.EI)

            points, temp = load_eon_file(thermal_path)
            ei_on_800_175or150.EI = points
            ei_on_800_175or150.Temperature = temp
            ei_on_800_175or150.Voltage = 800
            ei_on_800_175or150.NumberOfPoints = len(points)

            ei_on_0_175or150.EI = [[x, 0.0] for x, _ in ei_on_800_175or150.EI]
            ei_on_0_175or150.Temperature = temp
            ei_on_0_175or150.Voltage = 0.0
            ei_on_0_175or150.NumberOfPoints = len(points)

            thermal_mos.TurnOnLosses.Add(ei_on_800_25)
            thermal_mos.TurnOnLosses.Add(ei_on_0_25)
            thermal_mos.TurnOnLosses.Add(ei_on_800_175or150)
            thermal_mos.TurnOnLosses.Add(ei_on_0_175or150)

            thermal_mos.TurnOnLossComputationMethod = ThermalComputationMethodType.LookUpTableComputationMethodType

            ei_off_800_25 = EI_VT()
            ei_off_800_175or150 = EI_VT()
            ei_off_0_25 = EI_VT()
            ei_off_0_175or150 = EI_VT()

            ei_off_800_25.EI = csv_to_iv_points(os.path.join(thermal_path, "Eoff_25.csv"))
            ei_off_800_25.Temperature = 25
            ei_off_800_25.Voltage = 800
            ei_off_800_25.NumberOfPoints = len(ei_off_800_25.EI)

            ei_off_0_25.EI = [[x, 0.0] for x, _ in ei_off_800_25.EI]
            ei_off_0_25.Temperature = 25
            ei_off_0_25.Voltage = 0
            ei_off_0_25.NumberOfPoints = len(ei_off_800_25.EI)

            points, temp = load_eoff_file(thermal_path)
            ei_off_800_175or150.EI = points
            ei_off_800_175or150.Temperature = temp
            ei_off_800_175or150.Voltage = 800
            ei_off_800_175or150.NumberOfPoints = len(points)

            ei_off_0_175or150.EI = [[x, 0.0] for x, _ in ei_off_800_175or150.EI]
            ei_off_0_175or150.Temperature = temp
            ei_off_0_175or150.Voltage = 0.0
            ei_off_0_175or150.NumberOfPoints = len(points)

            thermal_mos.TurnOffLosses.Add(ei_off_800_25)
            thermal_mos.TurnOffLosses.Add(ei_off_0_25)
            thermal_mos.TurnOffLosses.Add(ei_off_800_175or150)
            thermal_mos.TurnOffLosses.Add(ei_off_0_175or150)

            thermal_mos.TurnOffLossComputationMethod = ThermalComputationMethodType.LookUpTableComputationMethodType
            project.Save()

        if PN_DIO in os.listdir(LIB_ROOT_DIO):
            # print("yes!yes!")
            thermal_dio.Name = PN_DIO
            thermal_dio.PartNumber = PN_DIO
            thermal_dio.Manufacturer = MNF_DIO
            thermal_dio.ThermalImpedanceType = ThermalDataType.FosterThermalNetworkType
            thermal_dio.SemiconductorType = ThermalDataSemiconductorType.DiodeThermalDataSemiconductorType
            thermal_path = os.path.join(LIB_ROOT_DIO, PN_DIO)
            df_dio_thermal_impedance = pd.read_excel(os.path.join(thermal_path, "foster_params.xlsx"))
            tau_dio = df_dio_thermal_impedance["tau_s"].to_numpy()
            Rk_dio = df_dio_thermal_impedance["Rk_C_per_W"].to_numpy()
            thermal_dio.SetThermalImpedanceData([[float(r), float(c)] for r, c in zip(Rk_dio, tau_dio)])

            thermal_dio.ConductionLosses.Clear()
            thermal_dio.TurnOnLosses.Clear()
            thermal_dio.TurnOffLosses.Clear()

            iv_25 = IV_T()
            iv_150or175 = IV_T()
            iv_25.Temperature = 25.0
            iv_25.IV = csv_to_iv_points(os.path.join(thermal_path, "Idiode_25.csv"))
            iv_25.NumberOfPoints = len(iv_25.IV)

            points, temp = load_ifvf_file(thermal_path)

            iv_150or175.Temperature = temp
            iv_150or175.IV = points
            iv_150or175.NumberOfPoints = len(iv_150or175.IV)

            thermal_dio.ConductionLosses.Add(iv_25)
            thermal_dio.ConductionLosses.Add(iv_150or175)
            thermal_dio.ConductionLossComputationMethod = ThermalComputationMethodType.LookUpTableComputationMethodType
            thermal_dio.ConductionLossesGateDependency = False

            project.Save()

    # =====================================================
    # Helpers: apply transient settings BEFORE running
    # (SIMBA APIs differ; this tries multiple common patterns)
    # =====================================================

    def apply_transient_settings(job, transient_settings):
        job.TransientSolver.CompressScopes = transient_settings["CompressScopes"]
        job.TransientSolver.TimeStep = transient_settings["TimeStep"]
        job.TransientSolver.FixedTimeStep = transient_settings["FixedTimeStep"]
        job.TransientSolver.DualStageElectroThermalAnalysis = transient_settings["DualStageElectroThermalAnalysis"]
        job.TransientSolver.BaseFrequencyParameterEnabled = transient_settings["BaseFrequencyParameterEnabled"]
        job.TransientSolver.BaseFrequency = transient_settings["BaseFrequency"]
        job.TransientSolver.StopAtSteadyState = transient_settings["StopAtSteadyState"]
        job.TransientSolver.SaveInitialPoint = transient_settings["SaveInitialPoint"]

    # Build signal classification in SIMBA
    def build_available_signals_classified(job, verbose=True):
        """
        Build and classify available SIMBA signals.

        Returns:
            dict with keys:
                - "all"   : all signal names
                - "loss"  : signals containing 'Average Total Losses'
                - "temp"  : signals containing 'Junction Temperature'
                - "other" : remaining signals
        """

        all_signals = [sig.Name for sig in job.Signals]

        loss_signals = []
        temp_signals = []
        other_signals = []

        for name in all_signals:
            if "Average Total Losses" in name:
                loss_signals.append(name)

            elif "Junction Temperature" in name:
                temp_signals.append(name)

            else:
                other_signals.append(name)

        if verbose:
            print("\n=== Available SIMBA Signals (All) ===")
            for i, name in enumerate(all_signals, start=1):
                print(f"{i:2d}: {name}")

            print("\n=== Loss Signals (Average Total Losses) ===")
            for i, name in enumerate(loss_signals, start=1):
                print(f"{i:2d}: {name}")

            print("\n=== Temperature Signals (Junction Temperature) ===")
            for i, name in enumerate(temp_signals, start=1):
                print(f"{i:2d}: {name}")

            print("\n=== Other Signals ===")
            for i, name in enumerate(other_signals, start=1):
                print(f"{i:2d}: {name}")

        return {
            "all": all_signals,
            "loss": loss_signals,
            "temp": temp_signals,
            "other": other_signals,
        }

    # =====================================================
    # Helpers: get time base + steady-state alignment by EndTime / last sample
    # =====================================================

    def get_time_base_semi(job):
        """
        Pick a 'legal' reference time base among available signals:
        - choose the time-based signal with the most points
        - return (t_ref, ref_name)
        """
        signal = build_available_signals_classified(job, verbose=False)
        best: Optional[Tuple[int, np.ndarray, str]] = None

        for n in signal["loss"]:
            sig_semi = job.GetSignalByName(n)
            if sig_semi is None:
                continue
            tp_in_f = getattr(sig_semi, "TimePoints", None)
            dp_in_f = getattr(sig_semi, "DataPoints", None)
            if not tp_in_f or not dp_in_f:
                continue
            if len(tp_in_f) < 1:
                continue

            if best is None or len(tp_in_f) > best[0]:
                best = (len(tp_in_f), np.asarray(tp_in_f, dtype=float), n)

        if best is None:
            raise RuntimeError("No time-based signal found to serve as time base.")
        return best[1], best[2]

    def get_time_base(job):
        """
        Pick a 'legal' reference time base among available signals:
        - choose the time-based signal with the most points
        - return (t_ref, ref_name)
        """
        signal_other = build_available_signals_classified(job, verbose=False)
        best: Optional[Tuple[int, np.ndarray, str]] = None

        for n in signal_other["other"]:
            sig_other = job.GetSignalByName(n)
            if sig_other is None:
                continue
            tp_in_f = getattr(sig_other, "TimePoints", None)
            dp_in_f = getattr(sig_other, "DataPoints", None)
            if not tp_in_f or not dp_in_f:
                continue
            if len(tp_in_f) < 1:
                continue

            if best is None or len(tp_in_f) > best[0]:
                best = (len(tp_in_f), np.asarray(tp_in_f, dtype=float), n)

        if best is None:
            raise RuntimeError("No time-based signal found to serve as time base.")
        return best[1], best[2]

    def steady_state_cycles_by_endtime(job, n_cycles_in_f, fsw_in_f, *signals_in_f, names_in_f=None, semi=True,
                                       n_grid=2):
        """
        Returns: dict[label] = {"t": t_ss, "data": y_interp_or_scalar}

        Robust to scalar reference time bases (t_ref.size == 1) by creating a synthetic t_ss grid.
        Also, robust to scalar signals (tp.size == 1).
        """
        if semi:
            t_ref, ref_name = get_time_base_semi(job)
        else:
            t_ref, ref_name = get_time_base(job)

        t_ref = np.asarray(t_ref, dtype=float).ravel()

        t_period_in_f = 1.0 / float(fsw_in_f)
        horizon_time = float(n_cycles_in_f) * t_period_in_f

        # Use solver EndTime if possible, else last time sample
        try:
            t_end_avail = float(job.TransientSolver.get_EndTime())
        except Exception:
            t_end_avail = float(t_ref[-1])

        t_start = t_end_avail - horizon_time

        # --- Build t_ss ---
        # If reference time base has >=2 points, cut from it.
        # If it has only 1 point, synthesize a time grid for averaging/interp.
        if t_ref.size >= 2:
            mask = (t_ref >= t_start) & (t_ref <= t_end_avail)
            if int(np.sum(mask)) < 2:
                # fallback: synthesize if mask too small
                t_ss_in_f = np.linspace(t_start, t_end_avail, n_grid)
            else:
                t_ss_in_f = t_ref[mask]
        else:
            t_ss_in_f = np.linspace(t_start, t_end_avail, n_grid)

        out = {}

        for idx, sig in enumerate(signals_in_f):
            label = (names_in_f[idx] if (names_in_f and idx < len(names_in_f)) else getattr(sig, "Name", f"arg{idx}"))

            if sig is None:
                raise RuntimeError(f"Signal None: {label}")

            tp_in_f = getattr(sig, "TimePoints", None)
            dp_in_f = getattr(sig, "DataPoints", None)
            if tp_in_f is None or dp_in_f is None:
                continue

            t_sig = np.asarray(tp_in_f, dtype=float).ravel()
            y_sig = np.asarray(dp_in_f, dtype=float).ravel()

            if t_sig.size == 0 or y_sig.size == 0:
                continue

            # Scalar signal: keep as scalar (no interpolation needed)
            if t_sig.size == 1 or y_sig.size == 1:
                out[label] = {"t": np.asarray([t_end_avail], float), "data": np.asarray([float(y_sig[-1])], float)}
                continue

            if t_sig.size != y_sig.size:
                continue

            order = np.argsort(t_sig)
            t_sig = t_sig[order]
            y_sig = y_sig[order]

            y = np.interp(t_ss_in_f, t_sig, y_sig, left=float(y_sig[0]), right=float(y_sig[-1]))
            out[label] = {"t": t_ss_in_f, "data": y}

        if not out:
            raise RuntimeError("No signals were processed (out is empty).")

        return out

    # =====================================================
    # Helpers: robust integration / averaging (NumPy compatible)
    # =====================================================
    def compute_avg_total_losses_from_waveforms(
            waveforms_in_f: Dict[str, Dict[str, np.ndarray]],
            keyword: str = "Average Total Losses (W)",
    ) -> Tuple[Dict[str, float], float]:
        """
        Uses the output of steady_state_cycles_by_endtime(): dict[name]={'t':..., 'data':...}
        Works for both:
          - scalar (len(data)==1) -> average = that value
          - waveform -> average via trapezoid / window
        """
        loss_avg: Dict[str, float] = {}

        for name, pack in waveforms_in_f.items():
            if keyword not in name:
                continue

            t = np.asarray(pack["t"], dtype=float).ravel()
            y = np.asarray(pack["data"], dtype=float).ravel()

            if y.size == 0:
                continue

            if y.size == 1:
                loss_avg[name] = float(y[0])  # scalar already average
            else:
                if t.size < 2:
                    # If someone passed waveform with 1-point t, treat as scalar fallback
                    loss_avg[name] = float(y[-1])
                else:
                    time_window = float(t[-1] - t[0])
                    if time_window <= 0:
                        loss_avg[name] = float(np.mean(y))  # fallback
                    else:
                        loss_avg[name] = float(np.trapezoid(y, t) / time_window)

        total = float(sum(loss_avg.values()))
        return loss_avg, total

    def compute_power_losses(job, power_map, t_ss_in_f):
        losses = {}
        time_window = t_ss_in_f[-1] - t_ss_in_f[0]

        for name, cfg in power_map.items():
            v_sig = job.GetSignalByName(cfg["v"])
            i_sig = job.GetSignalByName(cfg["i"])

            if v_sig is None or i_sig is None:
                print(f"[Skip] {name}: signal not found")
                continue

            v_in_f = np.interp(t_ss, v_sig.TimePoints, v_sig.DataPoints)
            i_in_f = np.interp(t_ss, i_sig.TimePoints, i_sig.DataPoints)

            p_avg = np.trapezoid(v_in_f * i_in_f, t_ss_in_f) / time_window
            losses[name] = p_avg

        return losses

    def average_over_window(t, y):
        """
        Average of y(t) over the window [t0, t1]:
            y_avg = (1/T) * ∫ y dt
        Uses np.trapezoid for broad NumPy compatibility.
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if t.size < 2:
            raise RuntimeError("average_over_window: t must have >= 2 points")
        time_window = float(t[-1] - t[0])
        if time_window <= 0:
            raise RuntimeError(f"average_over_window: invalid window length T={time_window}")
        return float(np.trapezoid(y, t) / time_window)

    def check_device_loss_consistency(loss_avg_map_in_f,
                                      diode_tol=1.0,
                                      mosfet_tol=2.0,
                                      verbose=True):
        """
        Check if diode and MOSFET losses are close enough.

        Rules:
          - D1A, D2A, D1B, D2B: difference <= diode_tol
          - Q1, Q2: difference <= mosfet_tol

        Returns:
            dict with results
        """

        # Extract device -> loss mapping
        device_loss = {
            k_in_f.split(" - ")[0]: v_in_f
            for k_in_f, v_in_f in loss_avg_map_in_f.items()
        }

        # Expected devices
        diodes = ["D1A", "D2A", "D1B", "D2B"]
        mosfets = ["Q1", "Q2"]

        result = {
            "diodes_close": True,
            "mosfets_close": True,
            "diode_diffs": {},
            "mosfet_diff": None
        }

        # -------------------
        # Check diodes
        # -------------------
        diode_vals = {}

        for d in diodes:
            if d in device_loss:
                diode_vals[d] = device_loss[d]
            else:
                if verbose:
                    print(f"[Warn] Missing diode: {d}")

        diode_keys = list(diode_vals.keys())

        for i in range(len(diode_keys)):
            for j in range(i + 1, len(diode_keys)):
                d1 = diode_keys[i]
                d2 = diode_keys[j]

                diff = abs(diode_vals[d1] - diode_vals[d2])
                result["diode_diffs"][(d1, d2)] = diff

                if diff > diode_tol:
                    result["diodes_close"] = False

        # -------------------
        # Check MOSFET
        # -------------------
        if all(q in device_loss for q in mosfets):

            q1 = device_loss["Q1"]
            q2 = device_loss["Q2"]

            diff_q = abs(q1 - q2)
            result["mosfet_diff"] = diff_q

            if diff_q > mosfet_tol:
                result["mosfets_close"] = False

        else:
            if verbose:
                print("[Warn] Missing Q1 or Q2")

            result["mosfets_close"] = False

        # -------------------
        # Print summary
        # -------------------
        if verbose:

            # print("\n=== Loss Consistency Check ===")
            #
            # print("\nDiodes:")
            for (d1, d2), diff in result["diode_diffs"].items():
                status_in_f = "OK" if diff <= diode_tol else "NOT OK"
                print(f"{d1} vs {d2}: {diff:.3f} W  [{status_in_f}]")

            print(f"Diodes close: {result['diodes_close']} (tol = {diode_tol} W)")

            print("\nMOSFET:")
            if result["mosfet_diff"] is not None:
                status_in_f = "OK" if result["mosfet_diff"] <= mosfet_tol else "NOT OK"
                print(f"Q1 vs Q2: {result['mosfet_diff']:.3f} W  [{status_in_f}]")

            print(f"MOSFET close: {result['mosfets_close']} (tol = {mosfet_tol} W)")

        return result

    Thermal_data_assignment()

    for percentage, (load_value, weight) in Load_list.items():
        # print(percentage)
        time_step_adj = False
        pass_flag = False
        No_Try = 0
        while not pass_flag:
            # 0) Get project
            project = ProjectRepository(Project_DIR)
            design_main = project.GetDesignByName(TARGET_DESIGN)

            # print("Fsw =", fsw, "Hz")
            design_main.Circuit.SetVariableValue("Fsw", str(fsw))
            # print("RL =", load_value, "ohms")
            design_main.Circuit.SetVariableValue("RL", str(load_value))
            I_dynamic = 16.67 * float(percentage.strip('%')) / 100
            L_dynamic = compute_L_from_NI(N=N_int, I=I_dynamic, Ae=Ae, le=le, mu_r0=get_number_before_A(Core_PN),
                                          beta=b, c=c)
            # print("L =", L_dynamic * 1e6, "uH")
            design_main.Circuit.SetVariableValue("L", str(L_dynamic))
            design_main.Circuit.SetVariableValue("r", str(Rdc_wire))

            Imax = I_dynamic + 300 * 0.5 / (2 * fsw * L_dynamic)
            Imin = I_dynamic - 300 * 0.5 / (2 * fsw * L_dynamic)
            if Imin <= 0:
                mode = "DCM"
                Imin = 0.0
            else:
                mode = "CCM"

            # print(Imax, Imin, mode)

            Bmax = cal_flux_density_in_gauss(N=N_int, I=Imax, Iavg=I_dynamic, le=le, mu_r0=get_number_before_A(Core_PN),
                                             beta=b, c=c)
            Bmin = cal_flux_density_in_gauss(N=N_int, I=Imin, Iavg=I_dynamic, le=le, mu_r0=get_number_before_A(Core_PN),
                                             beta=b, c=c)
            Bpk = (Bmax - Bmin) * 0.5 * 1e-3  # Kgauss
            # print("Flux density-peak:", Bpk)

            Pcore = 1.4 * core_loss_from_B_f(k1=core_dict[Core_PN][2], k2=core_dict[Core_PN][3],
                                             anpha=core_dict[Core_PN][4], beta=core_dict[Core_PN][5],
                                             B_kG=Bpk, f_kHz=fsw * 1e-3, Ae_m2=Ae, le_m=le)
            # print("core loss at", percentage, ":", Pcore, "W")

            # 1) Create job
            job_main = design_main.TransientAnalysis.NewJob()

            # print(type(job_main))
            # print([x for x in dir(job_main) if "Run" in x or "Stop" in x or "Save" in x])
            #
            # ts = job_main.TransientSolver
            # print(type(ts))
            # print([x for x in dir(ts) if "Time" in x or "Step" in x or "End" in x or "Scope" in x or "Compress" in x or "Steady" in x or "Save" in x or "BaseFrequency" in x or "Dual" in x])

            # 2) Setting Simulation
            # 2A) Apply transient settings BEFORE running
            if not time_step_adj:
                transient_settings_main = {
                    "BaseFrequency": fsw,
                    "BaseFrequencyParameterEnabled": True,
                    "DualStageElectroThermalAnalysis": True,
                    "FixedTimeStep": False,
                    "TimeStep": 1e-9,
                    "StopAtSteadyState": True,
                    "CompressScopes": True,
                    "SaveInitialPoint": True,
                }
                apply_transient_settings(job_main, transient_settings_main)
            else:
                transient_settings_main = {
                    "BaseFrequency": fsw,
                    "BaseFrequencyParameterEnabled": True,
                    "DualStageElectroThermalAnalysis": True,
                    "FixedTimeStep": False,
                    "TimeStep": 1e-9,
                    "StopAtSteadyState": True,
                    "CompressScopes": True,
                    "SaveInitialPoint": True,
                }
                apply_transient_settings(job_main, transient_settings_main)
                # job_main.TransientSolver.EndTime = "50e-3,1"

            # print("Base frequency:",job_main.TransientSolver.get_BaseFrequency())
            # print("Time step:",job_main.TransientSolver.get_TimeStep())

            project.Save()
            # 3) Run transient simulation
            status = job_main.Run()

            # print([m for m in dir(job_main) if "run" in m.lower() and "time" in m.lower()])
            rt = getattr(job_main, "RunTime", None)
            # print("RunTime attr:", rt, "second")

            # print(">>> Simulation status:", status)
            # if str(status) != "OK":
            #     raise RuntimeError(job_main.Summary())

            # Loss calculation for Semiconductor Device
            # 4) Check the signal classification function.
            signals = build_available_signals_classified(job_main, verbose=False)
            Semi_loss_signal = signals["loss"]
            # print(f">>> Available signals: {len(Semi_loss_signal)}")

            # 5) Signal filtering
            sig_objs = []
            names = []
            # print(">>> Semi_loss_signal =", Semi_loss_signal)
            for name_var1 in Semi_loss_signal:
                sig_var1 = job_main.GetSignalByName(name_var1)
                # print("Checking:", name_var1, "→", sig_var1)
                if sig_var1 is None:
                    continue
                tp = getattr(sig_var1, "TimePoints", None)
                dp = getattr(sig_var1, "DataPoints", None)
                # print("tp:", tp)
                # print("dp:", dp)
                if tp and dp and len(tp) > 0:
                    sig_objs.append(sig_var1)
                    names.append(name_var1)

            # print(f">>> Time-based signals used: {len(names)}")
            if len(names) == 0:
                raise RuntimeError("No time-based signals found after filtering.")

            # # 6) Align all signals on a common steady-state window
            waveforms = steady_state_cycles_by_endtime(
                job_main,
                n_cycles,
                fsw,
                *sig_objs,
                names_in_f=names,
                semi=True
            )

            # 7) Get the aligned steady-state time vector
            t_ss = next(iter(waveforms.values()))["t"]
            # print("tss:", t_ss)

            # ===============================
            # Losses
            # ===============================

            # (A) Loss aggregation from SIMBA signals "*Average Total Losses (W)*"
            loss_avg_map, total_loss_from_signals = compute_avg_total_losses_from_waveforms(
                waveforms, keyword="Average Total Losses (W)")

            # print("\n=== Avg Total Losses from signals containing 'Average Total Losses (W)' ===")
            # if loss_avg_map:
            #     for k, v in sorted(loss_avg_map.items()):
            #         # print(f"{k}: {v:.3f} W")
            # else:
            #     print("[Warn] No '*Average Total Losses (W)*' signals found.")
            # print(f"Total loss (from signals) = {total_loss_from_signals:.3f} W")

            check_result = check_device_loss_consistency(
                loss_avg_map,
                diode_tol=1.0,
                mosfet_tol=2.0,
                verbose=False
            )
            if check_result["diodes_close"] and check_result["mosfets_close"]:
                pass_flag = True
                # print("results are reliable")
            else:
                # time_step_adj = True
                No_Try +=1
                # print("The timestep already is adjusted to be smaller")

            if No_Try == 1:
                pass_flag = True
                # print("reduce time step but still failed, require estimate")
                X = loss_avg_map["Q1 - Average Total Losses (W)"]
                Y = loss_avg_map["Q2 - Average Total Losses (W)"]
                if X > Y :
                    total_loss_from_signals = total_loss_from_signals -(X+Y) +2*X
                else:
                    total_loss_from_signals = total_loss_from_signals - (X + Y) + 2 * Y

                XD = loss_avg_map["D1A - Average Total Losses (W)"]
                YD = loss_avg_map["D2A - Average Total Losses (W)"]
                if XD > YD :
                    total_loss_from_signals = total_loss_from_signals -2*(XD+YD) +4*XD
                else:
                    total_loss_from_signals = total_loss_from_signals - 2*(XD + YD) + 4 * YD

                # print(f"Total loss (from signals_adjusted) = {total_loss_from_signals:.3f} W")
            # 8)
            other_signal = signals["other"]
            # print(f">>> Available signals: {len(other_signal)}")

            # 9)
            sig_objs_other = []
            names_other = []
            for name_var2 in other_signal:
                sig_var2 = job_main.GetSignalByName(name_var2)
                if sig_var2 is None:
                    continue
                tp = getattr(sig_var2, "TimePoints", None)
                dp = getattr(sig_var2, "DataPoints", None)
                if tp and dp and len(tp) > 3:
                    sig_objs_other.append(sig_var2)
                    names_other.append(name_var2)

            # print(f">>> Time-based signals used: {len(names_other)}")
            if len(names_other) == 0:
                raise RuntimeError("No time-based signals found after filtering.")

            # # 6) Align all signals on a common steady-state window
            waveforms = steady_state_cycles_by_endtime(
                job_main,
                n_cycles,
                fsw,
                *sig_objs_other,
                names_in_f=names_other,
                semi=False
            )
            # 7) Get the aligned steady-state time vector
            t_ss = next(iter(waveforms.values()))["t"]

            # (B) Optional: your custom POWER_MAP-based losses
            total_loss_from_map = None
            if POWER_MAP is not None:
                power_losses = compute_power_losses(job_main, POWER_MAP, t_ss)
                # print("\n=== Power Loss (POWER_MAP) ===")
                # for k, v in power_losses.items():
                    # print(f"{k}: {v:.3f} W")
                total_loss_from_map = float(sum(power_losses.values()))
                # print(f"Total loss (POWER_MAP) = {total_loss_from_map:.3f} W")

            # Choose which loss total to use for efficiency
            # Prefer direct SIMBA-reported average losses if available
            Total_loss = total_loss_from_signals * 1.1 + total_loss_from_map * 1.2 + 2 * Pcore

            # ===============================
            # Output power & efficiency
            # ===============================

            vout_sig = job_main.GetSignalByName("RL - Voltage")
            iout_sig = job_main.GetSignalByName("RL - Current")
            if vout_sig is None or iout_sig is None:
                raise RuntimeError("Missing 'RL - Voltage' and/or 'RL - Current' signals.")

            vout = np.interp(t_ss, np.asarray(vout_sig.TimePoints, float), np.asarray(vout_sig.DataPoints, float))
            iout = np.interp(t_ss, np.asarray(iout_sig.TimePoints, float), np.asarray(iout_sig.DataPoints, float))

            avg_pout = average_over_window(t_ss, vout * iout)
            efficiency = avg_pout / (avg_pout + Total_loss + 1e-30)
            if pass_flag:
                efficiencies.append(round(efficiency, 5))
                eff_ECE = eff_ECE + weight * efficiency
                eff_ECE = round(eff_ECE, 5)

            # print("\n=== Efficiency ===")
            # print(f"Avg Pout = {avg_pout:.3f} W")
            # print(f"Total Loss used = {Total_loss:.3f} W")
            # print(f"Efficiency: {percentage} = {efficiency * 100:.2f} %")
            # print(">>> END SIMULATION FLOW**********************************************************")

    efficiencies.append(eff_ECE)
    return efficiencies