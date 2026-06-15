"""
compositional_memory.py — продуктивная композиция в живом мозге (дыра №4).

Связывает роль+наполнитель в композит на торе (VSA/FHRR), позволяет ОБРАТНЫЙ запрос
роли. Работает на НОВЫХ сочетаниях (продуктивность Фодора), роле-чувствительно.
Использует ТЕ ЖЕ 34-мерные golden-векторы слов, что и предиктивный слой — одно
представление на слово во всём мозге.

2026-06-12 МАСШТАБ+ВРЕМЯ (другой путь, не GPU):
  • индекс (роль,слово)->отношения: recall O(кандидаты), не O(N) — миллионы фактов;
  • ЛЕНИВЫЕ композиты: vsa-вектор строится по требованию и кэшируется — RAM на факт
    падает с ~3КБ до ~сотен байт, ёмкость x10 на том же железе;
  • честная ёмкость: при переполнении старейший факт удаляется И ИЗ ИНДЕКСА
    (был баг: deque молча выселял, индекс отвечал «призраками»);
  • ВЕРСИИ ФАКТОВ (валидное время): update_fact закрывает старую версию
    (valid_to=t) и открывает новую (valid_from=t) — знание ОБНОВЛЯЕТСЯ без
    переобучения и без молчаливой перезаписи; recall(as_of=t) отвечает «на момент t»,
    history() отдаёт всю линию жизни факта с источниками. LLM так не может в принципе.

CANON: фазы [0,1), роли разнесены через phi, FHRR-cleanup, atomic write.
"""
import os
import json
import tempfile
import sys
from collections import deque, Counter, defaultdict
from .compositional_binding import vec, bind, unbind, compose, query, fhrr_sim, DIM, stable_seed
from .resonance_constants import PHI_INV, PHI_INV_CUBE, FIBONACCI, MAX_RULES

ROLE_ORDER = ["агент", "действие", "пациент"]   # позиционные роли (subject-verb-object)
_FUNC = {"и","в","не","на","с","что","а","к","но","по","из","у","за","о","от","до",
         "для","при","же","бы","ли","так","как","это","его","её","их","он","она","они"}

# Внутреннее отношение = list [composite|None, roles, source, meta]
# meta=None для вечного факта; иначе {"from": t|None, "to": t|None} — валидное время.
_C, _R, _S, _M = 0, 1, 2, 3


def _parts(rel):
    """Терпимый доступ: старые 2/3-кортежи (напр. внешние скрипты) -> 4 слота."""
    comp = rel[_C]
    roles = rel[_R]
    src = rel[_S] if len(rel) > _S else None
    meta = rel[_M] if len(rel) > _M else None
    return comp, roles, src, meta


def _active(meta, as_of=None):
    """Жив ли факт: as_of=None -> текущая версия (нет valid_to);
    as_of=t -> вечный ИЛИ valid_from<=t<valid_to."""
    if meta is None:
        return True
    if as_of is None:
        return meta.get("to") is None
    f, t = meta.get("from"), meta.get("to")
    return (f is None or f <= as_of) and (t is None or as_of < t)


class CompositionalMemory:
    def __init__(self, predictive=None, dim=None, state_dir=None, capacity=None):
        self.predictive = predictive            # источник общих 34D векторов слов
        self.dim = dim or (predictive.dim if predictive else DIM)
        self.state_dir = state_dir
        # ёмкость честная: по умолчанию MAX_RULES (как у языковых правил), env-override
        self.capacity = capacity or int(os.environ.get("LOGOS_FACT_CAPACITY", MAX_RULES))
        self.roles = {}                         # имя роли -> вектор роли
        for i, name in enumerate(["агент", "действие", "пациент", "место", "время"]):
            self.roles[name] = vec(((i + 1) * PHI_INV) % 1.0, self.dim)
        # СТРУКТУРНАЯ ПАМЯТЬ: who-did-what (символы в обучении), элементы — list (см. _C.._M)
        self.relations = deque()
        self._rel_seen = {}                            # ключ ролей -> счётчик повторов
        self.role_resolver = None   # callable(words)->[(role,word)]; ставит мозг (role_engine)
        # ИНДЕКС (#1): (роль,слово) -> список отношений. recall O(кандидаты), не O(N).
        self._role_index = defaultdict(list)
        self._load()
        self._reindex()

    # ------------------------------------------------------------------ индекс
    def _reindex(self):
        self._role_index = defaultdict(list)
        for rel in self.relations:
            self._index_relation(rel)

    def _index_relation(self, rel):
        for r, f in rel[_R].items():
            self._role_index[(r, f)].append(rel)

    def _unindex_relation(self, rel):
        for r, f in rel[_R].items():
            lst = self._role_index.get((r, f))
            if lst is not None:
                try:
                    lst.remove(rel)
                except ValueError:
                    pass
                if not lst:
                    del self._role_index[(r, f)]

    def _append_rel(self, rel):
        """Добавить с ЧЕСТНОЙ ёмкостью: переполнение -> старейший факт уходит
        целиком (из памяти, индекса и дедупа) — никаких «призраков»."""
        while len(self.relations) >= self.capacity:
            old = self.relations.popleft()
            self._unindex_relation(old)
            okey = tuple(sorted(old[_R].items()))
            self._rel_seen.pop(okey, None)
        self.relations.append(rel)
        self._index_relation(rel)

    def _candidates(self, known, as_of=None):
        """ПОЛНОЕ пересечение: отношения, совпадающие по ВСЕМ известным (роль,слово),
        живые на момент as_of. Берём кратчайший индекс-список и фильтруем — O(кратчайший).
        Пусто -> отказ (не угадываем по частичному совпадению — 0% галлюцинаций)."""
        lists = [self._role_index.get((r, f), ()) for r, f in known.items()]
        lists = [l for l in lists if l]
        if len(lists) < len(known):       # хоть одна (роль,слово) вообще не встречалась
            return []
        base = min(lists, key=len)
        return [rel for rel in base
                if all(rel[_R].get(r) == f for r, f in known.items())
                and _active(_parts(rel)[3], as_of)]

    def _composite(self, rel):
        """Ленивый композит: построить по требованию и закэшировать в слоте."""
        if rel[_C] is None:
            rel[_C] = self.compose_frame(list(rel[_R].items()))
        return rel[_C]

    # ------------------------------------------------------------------ обучение
    def _roles_from_words(self, words):
        """Позиционные роли: первые 3 содержательных слова -> агент/действие/пациент."""
        cw = [w for w in words if len(w) > 2 and w.lower() not in _FUNC]
        return [(ROLE_ORDER[i], cw[i]) for i in range(min(3, len(cw)))]

    def learn_triple(self, subject, relation, obj, source=None, valid_from=None):
        """ГИБРИД-ВХОД (#3): принять готовый факт (субъект, действие, объект) от
        СИЛЬНОГО внешнего экстрактора (LLM / dependency-парсер), минуя слабый SVO-разбор.
        Наш слой даёт то, чего у LLM нет: верифицируемое хранение + отказ + ЦИТАТУ.
        valid_from: момент, с которого факт действует (версии фактов)."""
        rf = [("агент", str(subject)), ("действие", str(relation)), ("пациент", str(obj))]
        key = tuple(sorted(rf))
        if key in self._rel_seen:
            # повтор живой версии -> только счётчик; но если ВСЕ версии закрыты —
            # факт стал верен СНОВА -> новая версия (re-activation)
            if any(_active(_parts(rel)[3]) for rel in self._candidates(dict(rf), as_of=None)):
                self._rel_seen[key] += 1
                return None
        meta = {"from": valid_from, "to": None} if valid_from is not None else None
        self._rel_seen[key] = self._rel_seen.get(key, 0) + 1
        rel = [None, {r: w for r, w in rf}, source, meta]   # композит лениво
        self._append_rel(rel)
        return rel

    def update_fact(self, subject, relation, new_obj, source=None, t=None):
        """ОБНОВЛЕНИЕ ЗНАНИЯ БЕЗ ПЕРЕОБУЧЕНИЯ И БЕЗ ПОТЕРИ ИСТОРИИ (у LLM нет аналога:
        там правка факта = fine-tune часами + порча соседей). Все живые версии
        (subject, relation, *) закрываются (valid_to=t), новая открывается (valid_from=t).
        Возврат: (закрыто_версий, новое_отношение|None — None если значение не сменилось)."""
        live = self._candidates({"агент": str(subject), "действие": str(relation)}, as_of=None)
        closed = 0
        for rel in live:
            if rel[_R].get("пациент") == str(new_obj):
                return (0, None)              # уже верно — не плодим версий
        for rel in live:
            if rel[_M] is None:
                rel[_M] = {"from": None, "to": t}
            else:
                rel[_M]["to"] = t
            closed += 1
        new_rel = self.learn_triple(subject, relation, new_obj, source=source, valid_from=t)
        return (closed, new_rel)

    def history(self, known):
        """Линия жизни факта: все версии (живые и закрытые), старые -> новые.
        Каждая: {'roles', 'source', 'from', 'to', 'active'}. Аудит/комплаенс:
        «что мы считали верным на момент T и откуда узнали»."""
        lists = [self._role_index.get((r, f), ()) for r, f in known.items()]
        lists = [l for l in lists if l]
        if len(lists) < len(known):
            return []
        base = min(lists, key=len)
        out = []
        for rel in base:
            if all(rel[_R].get(r) == f for r, f in known.items()):
                _, roles, src, meta = _parts(rel)
                out.append({"roles": dict(roles), "source": src,
                            "from": meta.get("from") if meta else None,
                            "to": meta.get("to") if meta else None,
                            "active": _active(meta)})
        out.sort(key=lambda v: (v["from"] is not None, v["from"] or 0))
        return out

    def learn_relation(self, words, source=None):
        """Связать роли предложения в структурный композит и СОХРАНИТЬ (символы в обучении).
        Роли: через role_engine-резолвер (выученные роли) или позиционно (фолбэк).
        source: ПРОВЕНАНС факта (откуда узнан) — напр. {'text': предложение, 'file': путь};
        позволяет recall ЦИТИРОВАТЬ источник (свойство, которого у LLM нет)."""
        rf = self.role_resolver(words) if self.role_resolver else self._roles_from_words(words)
        if not rf or len(rf) < 2:
            return None
        key = tuple(sorted((r, w) for r, w in rf))   # дедуп: один факт — одна запись
        if key in self._rel_seen:
            self._rel_seen[key] += 1                  # повтор -> только счётчик, не дубль
            return None
        self._rel_seen[key] = 1
        rel = [None, {r: w for r, w in rf}, source, None]
        self._append_rel(rel)
        return rel

    # порог КАЛИБРОВАННОГО ОТКАЗА: VSA-резонанс ниже -> «не знаю» (не выдумываем).
    # phi-натурально; настраивается по кривой точность/отказ (см. recall_calibration_test).
    ABSTAIN_MARGIN = PHI_INV_CUBE        # 0.236 — минимальный отрыв top1 от top2

    # ------------------------------------------------------------------ запросы
    def recall(self, known, query_role, with_confidence=False, with_source=False,
               as_of=None):
        """Ролевой запрос с КАЛИБРОВАННЫМ ОТКАЗОМ + ПРОВЕНАНСОМ — то, чего у LLM нет:
        отвечает когда знает; ЧЕСТНО отказывается когда не уверен; и ЦИТИРУЕТ источник.
        as_of: ответ «на момент t» (версии фактов); None = текущее знание.
        1) СИМВОЛЬНАЯ точность -> (filler, 1.0, 'exact', source).
        2) VSA-резонанс: уверенность = ОТРЫВ top1 от top2 (margin); мал -> ОТКАЗ.
        with_source=True добавляет source (откуда факт). with_confidence -> (filler,conf,mode).
        Оба False -> строка-filler/None (back-compat)."""
        def _ret(filler, conf, mode, src=None):
            if with_source:
                return (filler, round(conf, 4), mode, src)
            if with_confidence:
                return (filler, round(conf, 4), mode)
            return filler
        if not self.relations:
            return _ret(None, 0.0, 'abstain')
        # ИНДЕКС (#1): кандидаты = отношения с общими (роль,слово). Нет общих -> мгновенный
        # отказ (новый запрос); иначе сканируем только маленький набор, не всю память.
        cands = self._candidates(known, as_of=as_of)
        if not cands:
            return _ret(None, 0.0, 'abstain')
        # 1) точное символьное совпадение всех известных ролей -> уверенно, с источником
        for rel in reversed(cands):
            roles = rel[_R]
            if query_role in roles:
                return _ret(roles[query_role], 1.0, 'exact', _parts(rel)[2])
        # 2) VSA-резонанс по MARGIN — ТОЛЬКО над кандидатами (быстро)
        cue = compose([(self.role(r), self._wvec(f)) for r, f in known.items()])
        best, best_src, bs, second = None, None, -1e9, -1e9
        for rel in cands:
            s = fhrr_sim(cue, self._composite(rel))
            if s > bs:
                second = bs; best, best_src, bs = rel[_R], _parts(rel)[2], s
            elif s > second:
                second = s
        margin = bs - second if second > -1e9 else bs
        if best is None or query_role not in best or margin < self.ABSTAIN_MARGIN:
            return _ret(None, max(0.0, margin), 'abstain')   # не уверен -> молчим
        return _ret(best.get(query_role), max(0.0, margin), 'resonant', best_src)

    def recall_all(self, known, query_role, as_of=None, limit=FIBONACCI[10]):
        """ВСЕ точные ответы (мультизначный факт) с источниками — основа multi-hop.
        Только exact-режим (резонанс не используется -> гарантия 0% галлюцинаций
        сохраняется по построению). Возврат: [(filler, source, meta), ...]."""
        out, seen = [], set()
        for rel in reversed(self._candidates(known, as_of=as_of)):
            roles = rel[_R]
            f = roles.get(query_role)
            if f is not None and f not in seen:
                seen.add(f)
                _, _, src, meta = _parts(rel)
                out.append((f, src, meta))
                if len(out) >= limit:
                    break
        return out

    # ------------------------------------------------------------ доказательства/слияние
    def _sorted_leaves(self):
        """Детерминированный список (leaf_hash, rel) по всем ЖИВЫМ фактам,
        отсортированный по хешу листа -> воспроизводимое Merkle-дерево."""
        from .knowledge_proof import fact_leaf
        items = []
        for rel in self.relations:
            _, roles, src, meta = _parts(rel)
            if _active(meta):
                items.append((fact_leaf(roles, src), rel))
        items.sort(key=lambda x: x[0])
        return items

    def knowledge_root(self):
        """MERKLE-КОРЕНЬ всего знания: один хеш фиксирует ВСЁ состояние. Любая подмена
        любого факта -> другой корень. LLM не может предъявить такой коммит знания."""
        from .knowledge_proof import merkle_root
        return merkle_root([h for h, _ in self._sorted_leaves()])

    def prove_fact(self, known, query_role):
        """Доказательство включения факта в знание: (leaf, proof, root, ответ).
        Доказывает «этот факт у меня есть под корнем R», НЕ раскрывая прочие факты.
        None если факта нет (нельзя доказать отсутствующее -> честно)."""
        from .knowledge_proof import fact_leaf, inclusion_proof, merkle_root
        leaves = self._sorted_leaves()
        for i, (h, rel) in enumerate(leaves):
            roles = rel[_R]
            if query_role in roles and all(roles.get(r) == f for r, f in known.items()):
                arr = [x for x, _ in leaves]
                return {"leaf": h, "proof": inclusion_proof(arr, i),
                        "root": merkle_root(arr), "answer": roles[query_role],
                        "source": _parts(rel)[2]}
        return None

    def merge_from(self, other, source_tag=None):
        """СЛИЯНИЕ ЗНАНИЯ двух памятей = union отношений (знание — ДАННЫЕ, не веса).
        Сохраняет провенанс; новые факты получают source_tag-пометку происхождения.
        Возврат: число добавленных. Два LLM слить нельзя (усреднение весов губит оба)."""
        added = 0
        for rel in list(other.relations):
            _, roles, src, meta = _parts(rel)
            key = tuple(sorted(roles.items()))
            if key in self._rel_seen:
                self._rel_seen[key] += 1
                continue
            if source_tag is not None:
                src = dict(src) if isinstance(src, dict) else {"src": src}
                src.setdefault("merged_from", source_tag)
            self._rel_seen[key] = 1
            self._append_rel([None, dict(roles), src, dict(meta) if meta else None])
            added += 1
        return added

    def all_paths(self, start, end, max_depth=FIBONACCI[5], as_of=None):
        """ВСЕ доказуемые пути start->end по рёбрам агент->пациент (любое действие),
        простые (без повторов узлов), глубиной до max_depth. Только EXACT-факты ->
        каждый путь реален и цитируем (0% выдуманных связок). LLM не перечислит пути
        исчерпывающе без галлюцинаций. Возврат: [[{from,rel,to,source}...], ...]."""
        start, end = str(start), str(end)
        paths = []

        def edges(node):
            out = []
            for rel in self._role_index.get(("агент", node), ()):
                _, roles, src, meta = _parts(rel)
                if "пациент" in roles and _active(meta, as_of):
                    out.append((roles.get("действие"), roles["пациент"], src))
            return out

        def dfs(node, visited, acc):
            if len(acc) > max_depth:
                return
            for rel, nxt, src in edges(node):
                if nxt in visited:
                    continue
                step = {"from": node, "rel": rel, "to": nxt, "source": src}
                if nxt == end:
                    paths.append(acc + [step])
                else:
                    dfs(nxt, visited | {nxt}, acc + [step])

        dfs(start, {start}, [])
        return paths

    def explain(self, known, query_role):
        """ОБЪЯСНЕНИЕ ПО ПОСТРОЕНИЮ: ответ + ТОЧНЫЕ факты, на которых он основан
        (а не пост-хок текст, как CoT у LLM). Возврат: {answer, mode, used:[{roles,source}]}.
        Объяснение ВЕРНО по построению: used — это буквально извлечённые факты."""
        cands = self._candidates(known)
        for rel in reversed(cands):
            roles = rel[_R]
            if query_role in roles:
                _, _, src, _ = _parts(rel)
                return {"answer": roles[query_role], "mode": "exact",
                        "used": [{"roles": dict(roles), "source": src}]}
        return {"answer": None, "mode": "abstain", "used": []}

    def forget(self, subject, relation, obj=None):
        """ДОКАЗУЕМОЕ ЗАБЫВАНИЕ (right-to-be-forgotten) — то, чего LLM не может:
        факт вплавлен в веса, удалить нельзя, лишь подавить. Здесь факт УДАЛЯЕТСЯ
        из памяти, индекса и дедупа ПОЛНОСТЬЮ — после recall честно отказывает,
        и это проверяемо. obj=None -> удалить ВСЕ значения (subject,relation).
        Возврат: число удалённых отношений (всех версий, живых и закрытых)."""
        known = {"агент": str(subject), "действие": str(relation)}
        if obj is not None:
            known["пациент"] = str(obj)
        # собрать все совпадающие (включая закрытые версии — для полного стирания)
        lists = [self._role_index.get((r, f), ()) for r, f in known.items()]
        lists = [l for l in lists if l]
        if len(lists) < len(known):
            return 0
        victims = [rel for rel in list(min(lists, key=len))
                   if all(rel[_R].get(r) == f for r, f in known.items())]
        for rel in victims:
            try:
                self.relations.remove(rel)
            except ValueError:
                pass
            self._unindex_relation(rel)
            self._rel_seen.pop(tuple(sorted(rel[_R].items())), None)
        return len(victims)

    def contradictions(self, functional_relations):
        """ДЕТЕКТОР ПРОТИВОРЕЧИЙ: для ФУНКЦИОНАЛЬНЫХ отношений (где у (агент,действие)
        должно быть одно значение — 'столица', 'дата_рождения') находит (агент,действие)
        с НЕСКОЛЬКИМИ ЖИВЫМИ значениями. Возвращает каждое с источниками — аудит видит
        конфликт и обе цитаты, а не молча выбранное (как смешивает LLM).
        Возврат: [{'агент','действие','значения':[(пациент,источник),...]}]."""
        funcs = set(map(str, functional_relations))
        by_key = defaultdict(list)
        for rel in self.relations:
            _, roles, src, meta = _parts(rel)
            act = roles.get("действие")
            if act in funcs and "агент" in roles and "пациент" in roles and _active(meta):
                by_key[(roles["агент"], act)].append((roles["пациент"], src))
        out = []
        for (ag, act), vals in by_key.items():
            distinct = {v for v, _ in vals}
            if len(distinct) > 1:        # одно (агент,действие) -> >1 живое значение = конфликт
                out.append({"агент": ag, "действие": act, "значения": vals})
        return out

    def multihop(self, start, relations, as_of=None):
        """ЦЕПНОЙ ВЫВОД (multi-hop) по фактам: старт-сущность + список действий.
        Шаг i: (агент=текущий, действие=relations[i]) -> пациент = следующий.
        Только EXACT (никакого резонанса) -> гарантия 0% галлюцинаций по построению,
        как у одношагового recall. LLM на multi-hop галлюцинирует особенно сильно;
        здесь каждый шаг — реальный сохранённый факт со СВОИМ источником.

        Возврат: {'answer': итог|None, 'mode': 'exact'|'abstain', 'hops': [...],
                  'abstained_at': i|None}. hops[i] = {'from','rel','to','source'}.
        Воздержание РАСПРОСТРАНЯЕТСЯ: первый же шаг без однозначного факта рвёт цепь."""
        cur = str(start)
        hops = []
        for i, rel in enumerate(relations):
            opts = self.recall_all({"агент": cur, "действие": str(rel)}, "пациент", as_of=as_of)
            if len(opts) != 1:        # 0 = не знаем; >1 = неоднозначно -> честный отказ
                return {"answer": None, "mode": "abstain", "hops": hops,
                        "abstained_at": i, "reason": "no_fact" if not opts else "ambiguous"}
            nxt, src, _meta = opts[0]
            hops.append({"from": cur, "rel": str(rel), "to": nxt, "source": src})
            cur = nxt
        return {"answer": cur, "mode": "exact", "hops": hops, "abstained_at": None}

    # ------------------------------------------------------------------ векторы
    def _wvec(self, word):
        if self.predictive is not None:
            return self.predictive._vec(word)   # общий вектор слова
        return vec(stable_seed(word), self.dim)

    def role(self, name):
        if name not in self.roles:
            self.roles[name] = vec(stable_seed(name), self.dim)  # детерминированно
        return self.roles[name]

    def _vocab(self):
        if self.predictive is not None and self.predictive.wvec:
            return self.predictive.wvec
        return {}

    def compose_frame(self, role_filler):
        """role_filler: [(имя_роли, слово), ...] -> композит T^N."""
        pairs = [(self.role(r), self._wvec(w)) for (r, w) in role_filler]
        return compose(pairs)

    def query_role(self, composite, role_name):
        """Развязать роль из композита -> ближайшее слово словаря мозга. (слово, похожесть)."""
        vocab = self._vocab()
        if not vocab:
            return (None, 0.0)
        return query(composite, self.role(role_name), vocab)

    # ------------------------------------------------------------------ персист
    def save(self):
        """Персист структурной памяти (Canon #4: atomic). Композиты НЕ сохраняем —
        они детерминированно пересобираются из role:filler (stable_seed/phi-роли)."""
        if not self.state_dir:
            return
        rows = []
        for rel in self.relations:
            _, roles, src, meta = _parts(rel)
            row = {"roles": roles, "source": src}
            if meta is not None:
                row["meta"] = meta
            rows.append(row)
        state = {
            "dim": self.dim,
            # v3: роли + провенанс + валидное время. Композит пересобирается лениво.
            "relations": rows,
            "seen": [[list(map(list, k)), n] for k, n in self._rel_seen.items()],
        }
        path = os.path.join(self.state_dir, "compositional_memory.json")
        try:
            fd, tmp = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[!] CompositionalMemory save failed: {e}", file=sys.stderr)

    def _load(self):
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, "compositional_memory.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            if s.get("dim") is not None and s["dim"] != self.dim:
                print(f"[!] CompositionalMemory: dim {s['dim']}≠{self.dim}, state пропущен", file=sys.stderr)
                return
            for item in s.get("relations", []):
                if isinstance(item, dict) and "roles" in item:   # v2/v3: с провенансом
                    roles = item["roles"]; src = item.get("source"); meta = item.get("meta")
                else:                                            # v1: старый формат
                    roles = item; src = None; meta = None
                self.relations.append([None, roles, src, meta])  # композит лениво
            for k, n in s.get("seen", []):
                self._rel_seen[tuple(tuple(p) for p in k)] = n
            print(f"[+] CompositionalMemory: загружено {len(self.relations)} отношений", file=sys.stderr)
        except Exception as e:
            print(f"[!] CompositionalMemory load failed: {e}", file=sys.stderr)
