"""
compositional_binding.py — Продуктивная композициональность (дыра №4).

symbol_binding.py «does NOT compose» (1 символ = 1 слот). Здесь — настоящая
векторно-символьная алгебра (VSA/HRR Плейта) НА ТОРЕ: каждый символ = вектор фаз
T^N, связывание роль⊗наполнитель = по-мерное сложение по кругу (групповая операция,
ОБРАТИМАЯ), композиция нескольких пар = суперпозиция (по-мерный circular_mean).
Запрос роли = развязать и найти ближайший наполнитель. Систематично: работает на
НОВЫХ комбинациях, не виденных при обучении (productivity Фодора). Роле-чувствительно:
bind(agent,X)+bind(patient,Y) ≠ bind(agent,Y)+bind(patient,X).

Ёмкость растёт с N (на скалярной фазе её нет — потому тор, а не круг).
CANON: фазы [0,1), сложение/среднее по кругу, размерности — Fibonacci-натуральные.
"""
import math
import hashlib
from .resonance_constants import (
    PHI, PHI_INV, FIBONACCI, phi_phase_distance, circular_mean,
)


def stable_seed(label):
    """Детерминированный сид [0,1) из строки (hashlib, НЕ builtin hash — тот
    рандомизирован по процессам и давал невоспроизводимые/коллизионные векторы)."""
    return int.from_bytes(hashlib.sha1(str(label).encode("utf-8")).digest()[:8],
                          "big") / float(1 << 64)

DIM = FIBONACCI[6]  # 8 — дефолтная размерность тора; ёмкость растёт с N
GOLDEN32 = 2654435769  # round(2^32 / PHI) — golden-ratio хеш Кнута (phi-нативно)


def vec(seed_phase, dim=DIM):
    """Фазовый вектор T^dim: измерения ДЕКОРРЕЛИРОВАНЫ через golden-ratio хеш (seed,d).
    Декорреляция обязательна для VSA-ёмкости — иначе вектор фактически 1-мерный."""
    s = int((seed_phase % 1.0) * (1 << 32)) & 0xFFFFFFFF
    v = []
    for d in range(dim):
        x = (s ^ ((d + 1) * GOLDEN32)) & 0xFFFFFFFF
        x = (x * GOLDEN32) & 0xFFFFFFFF
        x ^= x >> 16
        v.append((x & 0xFFFFFFFF) / float(1 << 32))
    return v


def bind(role, filler):
    """роль⊗наполнитель = по-мерное сложение по кругу. Обратимо. (dim = len входа)"""
    return [(role[d] + filler[d]) % 1.0 for d in range(len(role))]


def unbind(bound, role):
    """развязать: bound ⊘ role = по-мерное вычитание по кругу -> наполнитель."""
    return [(bound[d] - role[d]) % 1.0 for d in range(len(bound))]


def compose(pairs):
    """суперпозиция списка (role,filler) -> один композит T^N (по-мерный circular_mean)."""
    if not pairs:
        return []
    bound = [bind(r, f) for (r, f) in pairs]
    n = len(bound[0])
    return [circular_mean([b[d] for b in bound]) for d in range(n)]


def torus_distance(a, b):
    """phi-взвешенное расстояние на торе (как в phase_torus)."""
    tot = wsum = 0.0
    for d in range(len(a)):
        w = PHI_INV ** d
        tot += phi_phase_distance(a[d], b[d]) * w
        wsum += w
    return tot / wsum


def fhrr_sim(a, b):
    """FHRR-похожесть: средн. cos(2π·разн.фаз) по ВСЕМ измерениям равновесно.
    Real-часть комплексного скалярного произведения единичных векторов.
    Растёт дискриминативно с N (SNR ~ √N) — в отличие от φ-взвешенной torus_distance."""
    n = len(a)
    return sum(math.cos(2 * math.pi * (a[d] - b[d])) for d in range(n)) / n


def query(composite, role, vocab):
    """развязать роль из композита и найти наполнитель с макс. FHRR-похожестью.
    vocab: dict name -> фазовый вектор. Возвращает (name, similarity)."""
    probe = unbind(composite, role)
    best, bs = None, -1e9
    for name, v in vocab.items():
        s = fhrr_sim(probe, v)
        if s > bs:
            best, bs = name, s
    return best, bs
