"""resonance_constants.py v10 — Единый источник всех констант.

FIX: добавлен PHI_SQ, HARM_THRESHOLD (единый порог для всех модулей).
"""
import math

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1
PHI_SQ = PHI * PHI
PHI_INV_SQ = PHI_INV * PHI_INV
PHI_INV_CUBE = PHI_INV ** 3

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141]

HARMONICS = [0, 1, 2, 4, 5, 6, 11]
HARMONIC_WEIGHTS = {0: 1.000, 1: 1.705, 2: 1.136, 4: 1.390, 5: 1.200, 6: 1.250, 11: 1.136}

FIELD_NAMES = [
    # Original 9 semantic fields (indices 0-8) — DO NOT REORDER, see CLAUDE.md
    "mental", "void", "resonance", "geometry", "will",
    "time", "matter", "network", "meta",
    # 2026-04-25: cosmic fields (indices 9-13) appended at the end so that
    # existing 0-8 phases (= i*PHI_INV % 1) remain identical → all 561k+ rules
    # in main brain torus continue to map correctly. New phases added on top.
    "schumann",   # 9 — atmospheric (Earth-ionosphere cavity)
    "solar",      # 10 — solar p-mode / rotation / cycle
    "lunar",      # 11 — synodic/sidereal lunar period
    "galactic",   # 12 — 21cm hydrogen, CMB peak, galactic year
    "cosmic",     # 13 — meta-cosmic (Cs hyperfine, Crab pulsar, generic)
]
FIELD_PHASES = {}
for i, name in enumerate(FIELD_NAMES):
    FIELD_PHASES[name] = (i * PHI_INV) % 1.0

PHI_TARGETS = {'phi': PHI, 'phi_inv': PHI_INV, 'phi_sq': PHI_SQ, 'phi_inv_sq': PHI_INV_SQ, 'phi_inv_cube': PHI_INV_CUBE, 'identity': 1.0}

# FIX L2: единый порог вреда для ВСЕХ модулей
# Расстояние от harm_phase (0.5) меньше этого = антирезонанс
HARM_THRESHOLD = PHI_INV_CUBE  # ~0.236 — фундаментальный порог
HARM_PHASE = 0.5  # антипод Creator (phase=0.0); максимум-далекая точка на круге [0,1)

CRYSTALLIZE_THRESHOLD = FIBONACCI[5]
DREAM_INTERVAL = FIBONACCI[9]
SAVE_INTERVAL = round(PHI ** 7)
MAX_RULES = FIBONACCI[29]
CO_OCCURRENCE_WINDOW = FIBONACCI[5]
MAX_COOCCURRENCE = FIBONACCI[22]


def phi_phase(distance, base_period=1.0):
    if distance <= 0 or base_period <= 0:
        return 0.0
    return (math.log(distance / base_period) / math.log(PHI)) % 1.0


def phi_phase_distance(phase1, phase2):
    diff = abs(phase1 - phase2)
    return min(diff, 1.0 - diff)


def phi_phase_resonance(phase):
    resonant = [0.0, PHI_INV % 1.0, (PHI_INV * 2) % 1.0, (PHI_INV * 3) % 1.0, 0.5]
    return max(0.0, 1.0 - min(phi_phase_distance(phase, rp) for rp in resonant) / PHI_INV_CUBE)


def circular_mean(phases):
    """Корректное среднее фаз на круге [0, 1).
    FIX M4: заменяет арифметическое среднее."""
    if not phases:
        return 0.0
    sin_sum = sum(math.sin(2 * math.pi * p) for p in phases)
    cos_sum = sum(math.cos(2 * math.pi * p) for p in phases)
    angle = math.atan2(sin_sum, cos_sum)
    return (angle / (2 * math.pi)) % 1.0


def is_near_phi_target(ratio, tolerance=0.05):
    best, best_err = None, float('inf')
    for name, target in PHI_TARGETS.items():
        if target == 0:
            continue
        err = abs(ratio - target) / target
        if err < tolerance and err < best_err:
            best, best_err = (name, err), err
    return best


