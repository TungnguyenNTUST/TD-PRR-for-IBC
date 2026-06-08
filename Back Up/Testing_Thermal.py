import os
import pandas as pd
from aesim.simba import ProjectRepository, ThermalDataType,ThermalDataSemiconductorType,ThermalComputationMethodType
from aesim.simba import IV_T, EI_VT
import numpy as np

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
# from System.Collections.ObjectModel import ObservableCollection

ROOT_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Simulation File"
Project_DIR = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Simulation File\Interleaved_Boost_Converter.jsimba"
TARGET_DESIGN = "IBC"
LIB_ROOT_MOS = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Paper_01_Component_Library_IBC\Mosfet"
LIB_ROOT_DIO = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign\Paper_01_Component_Library_IBC\Diodes"
#
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

PN_MOS = "E3M0032120K"
MNF_MOS = "Wolfspeed"

PN_DIO = "STPSC30H12C"
MNF_DIO = "ST"

if PN_MOS in os.listdir(LIB_ROOT_MOS):
    thermal_mos.Name = PN_MOS
    thermal_mos.PartNumber = PN_MOS
    thermal_mos.Manufacturer = MNF_MOS
    thermal_mos.ThermalImpedanceType = ThermalDataType.FosterThermalNetworkType
    thermal_mos.SemiconductorType = ThermalDataSemiconductorType.MosfetThermalDataSemiconductorType
    thermal_path = os.path.join(LIB_ROOT_MOS, PN_MOS, "Thermal")
    df_mos_thermal_impedance = pd.read_excel(os.path.join(thermal_path,"foster_params.xlsx"))
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

    ei_on_0_25.EI  = [[x, 0.0] for x, _ in ei_on_800_25.EI ]
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

    ei_off_0_25.EI  = [[x, 0.0] for x, _ in ei_off_800_25.EI ]
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
    print("yes!yes!")
    thermal_dio.Name = PN_DIO
    thermal_dio.PartNumber = PN_DIO
    thermal_dio.Manufacturer = MNF_DIO
    thermal_dio.ThermalImpedanceType = ThermalDataType.FosterThermalNetworkType
    thermal_dio.SemiconductorType = ThermalDataSemiconductorType.DiodeThermalDataSemiconductorType
    thermal_path = os.path.join(LIB_ROOT_DIO, PN_DIO)
    df_dio_thermal_impedance = pd.read_excel(os.path.join(thermal_path,"foster_params.xlsx"))
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