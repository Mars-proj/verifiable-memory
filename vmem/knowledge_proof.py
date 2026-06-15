"""
knowledge_proof.py — ВЕРИФИЦИРУЕМОЕ СОСТОЯНИЕ ЗНАНИЯ (углы, которых нет ни у LLM, ни у ML).

Свежий взгляд на движок: факты АДРЕСУЕМЫ и ХЕШИРУЕМЫ -> над всем знанием строится
Merkle-дерево. Один корневой хеш ФИКСИРУЕТ всё состояние знания; можно доказать
наличие конкретного факта, НЕ раскрывая остальные (inclusion proof); любая подмена
любого факта меняет корень. LLM не может: «что именно ты знаешь» невычислимо из весов.

Плюс: знание здесь — ДАННЫЕ (а не веса), значит ОБРАТИМО и СЛИВАЕМО: память двух
мозгов = union отношений за миллисекунды, с сохранением провенанса. Два LLM слить
нельзя (усреднение весов разрушает оба).

CANON: детерминированный хеш (sha1 как в stable_seed/crypto), без рандома.
"""
import hashlib
def hash_data(data):
    if isinstance(data, str): data = data.encode()
    return hashlib.sha256(data).hexdigest()


def fact_leaf(roles, source):
    """Каноническая сериализация факта в лист Merkle: отсортированные роли + источник.
    Детерминирована -> один и тот же факт всегда даёт один лист."""
    items = sorted((str(r), str(w)) for r, w in roles.items())
    src = "" if source is None else _canon(source)
    return hash_data("|".join(f"{r}={w}" for r, w in items) + "#" + src)


def _canon(obj):
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{_canon(obj[k])}" for k in sorted(obj)) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canon(x) for x in obj) + "]"
    return str(obj)


def _pair_hash(a, b):
    return hash_data(a + b) if a <= b else hash_data(b + a)   # порядок-независимо


def merkle_root(leaves):
    """Корень Merkle над списком листьев (хешей). Пустой -> хеш пустоты."""
    if not leaves:
        return hash_data("∅")
    level = list(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_pair_hash(level[i], level[i + 1]))
            else:
                nxt.append(level[i])          # нечётный -> поднимаем как есть
        level = nxt
    return level[0]


def inclusion_proof(leaves, index):
    """Путь доказательства включения листа index: список соседних хешей снизу вверх.
    Позволяет доказать «факт входит в знание с корнем R», НЕ раскрывая прочие факты."""
    proof = []
    level = list(leaves)
    idx = index
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                if i == idx or i + 1 == idx:
                    sib = level[i + 1] if i == idx else level[i]
                    proof.append(sib)
                nxt.append(_pair_hash(level[i], level[i + 1]))
            else:
                nxt.append(level[i])
        idx //= 2
        level = nxt
    return proof


def verify_inclusion(leaf, proof, root):
    """Проверить, что leaf входит в дерево с данным root по proof-пути."""
    h = leaf
    for sib in proof:
        h = _pair_hash(h, sib)
    return h == root
