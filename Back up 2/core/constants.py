"""
Shared physical constants for IBC Converter design.

Single source of truth — import from here in Loss.py, Inductor_loss.py, etc.
"""

# Switching frequency look-up table: index (1-based key) -> frequency in Hz
# Key = x4 index from the Input sheet (10 options: 20k–110kHz)
Frequency_list: dict = {
    1: 20e3,
    2: 30e3,
    3: 40e3,
    4: 50e3,
    5: 60e3,
    6: 70e3,
    7: 80e3,
    8: 90e3,
    9: 100e3,
    10: 110e3,
}

# Inductance look-up table: index (1-based key) -> inductance in Henries
# Key = x5 index from the Input sheet (10 options: 380–830 µH)
Inductance_list: dict = {
    1: 380e-6,
    2: 430e-6,
    3: 480e-6,
    4: 530e-6,
    5: 580e-6,
    6: 630e-6,
    7: 680e-6,
    8: 730e-6,
    9: 780e-6,
    10: 830e-6,
}
