"""
Project-wide path configuration — single source of truth.

All hard-coded paths live here.  Every other module imports from this file
instead of defining its own literals.

Usage:
    from core.paths import FILE_PATH, OUT_DIR, LIB_ROOT, ...

To relocate the project, only PROJECT_ROOT needs to change.
"""

import os

# ── Root ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\Tungtan\PycharmProjects\AutomationDesign"

# ── Input / data files ────────────────────────────────────────────────────────
FILE_PATH     = os.path.join(PROJECT_ROOT, "UserProvidedDataFile.xlsx")
FILE_PATH_DIC = os.path.join(PROJECT_ROOT, "UserProvidedDataFile_DIC.xlsx")
RESULT_PATH   = os.path.join(PROJECT_ROOT, "Data", "20260108_41_efficiency_results.xlsx")

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIR         = os.path.join(PROJECT_ROOT, "Results")
OUT_DIR_PAPER01 = os.path.join(PROJECT_ROOT, "Results", "Paper01")
OUT_MU_STD      = os.path.join(OUT_DIR_PAPER01, "Mu_and_Std.xlsx")

# ── SIMBA simulation files ────────────────────────────────────────────────────
SIM_ROOT_DIR     = os.path.join(PROJECT_ROOT, "Simulation File")
SIM_PROJECT_FILE = os.path.join(SIM_ROOT_DIR, "Interleaved_Boost_Converter.jsimba")

# ── Component library ─────────────────────────────────────────────────────────
LIB_ROOT = os.path.join(PROJECT_ROOT, "Paper_01_Component_Library_IBC")
