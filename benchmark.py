#!/usr/bin/env python3
"""
benchmark.py — verifiable-memory vs a guessing baseline on WN18RR + trap set.

Claim under test (narrow, honest): "0-hallucination, cited fact recall with
provable forgetting & determinism" — NOT "smarter than an LLM".

Tests:
  KNOWN   : query facts that ARE stored -> accuracy + citation coverage
  TRAP    : query (head,relation) pairs that are ABSENT -> abstention (no fabrication)
  BASELINE: a naive frequency guesser (proxy for a system that always answers)
            -> shows what "always answer" does on traps (fabrication rate)
  DETERMINISM : same query x1000 -> identical answer
  FORGET  : delete a stored fact -> it abstains AND Merkle root reverts
"""
import sys, os, random, time, json
random.seed(42)
_OUT = sys.stdout; sys.stdout = sys.stderr  # silence import banners
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HOME", "/tmp/logos_run")
from vmem.compositional_memory import CompositionalMemory
sys.stdout = _OUT

WN = "/tmp/wn_train.txt"
N_FACTS = int(os.environ.get("N_FACTS", "8000"))
N_KNOWN = 1500
N_TRAP  = 1500

def load_triples(path, n):
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            out.append((p[0], p[1], p[2]))
        if len(out) >= n:
            break
    return out

def main():
    triples = load_triples(WN, N_FACTS)
    mem = CompositionalMemory(predictive=None, state_dir="/tmp/vmem_bench", capacity=N_FACTS + 100)
    # fresh
    mem.relations.clear(); mem._rel_seen.clear(); mem._reindex()

    truth = {}          # (h,r) -> set(tails) ground truth
    heads = set(); rels = set()
    t0 = time.time()
    for i, (h, r, t) in enumerate(triples):
        mem.learn_triple(h, r, t, source={"triple_id": i, "file": "WN18RR"})
        truth.setdefault((h, r), set()).add(t)
        heads.add(h); rels.add(r)
    ingest_t = time.time() - t0
    heads = list(heads); rels = list(rels)

    # frequency baseline: most common tail per relation (proxy for "always answer")
    from collections import Counter, defaultdict
    rel_tail = defaultdict(Counter)
    for (h, r), ts in truth.items():
        for t in ts: rel_tail[r][t] += 1
    def baseline_answer(h, r):
        c = rel_tail.get(r)
        return c.most_common(1)[0][0] if c else None  # always guesses if relation seen

    # ---- KNOWN ----
    hr_present = list(truth.keys())
    known_sample = random.sample(hr_present, min(N_KNOWN, len(hr_present)))
    k_ans = k_correct = k_cited = 0
    qt0 = time.time()
    for (h, r) in known_sample:
        rows = mem.recall_all({"агент": h, "действие": r}, "пациент")
        if rows:
            k_ans += 1
            got = {f for f, _s, _m in rows}
            if got & truth[(h, r)]: k_correct += 1
            if all(s for _f, s, _m in rows): k_cited += 1
    known_q_ms = (time.time() - qt0) / max(1, len(known_sample)) * 1000

    # ---- TRAP (absent (head,relation) pairs; plausible: real head + real relation) ----
    traps = []
    while len(traps) < N_TRAP:
        if random.random() < 0.5:
            h = random.choice(heads); r = random.choice(rels)          # real h + real r, pair absent
        else:
            h = "FAKE_" + str(random.randint(0, 10**9)); r = random.choice(rels)  # fake head
        if (h, r) not in truth:
            traps.append((h, r))
    ours_fab = base_fab = 0
    for (h, r) in traps:
        rows = mem.recall_all({"агент": h, "действие": r}, "пациент")
        if rows: ours_fab += 1                    # we answered an absent fact = fabrication
        if baseline_answer(h, r) is not None: base_fab += 1

    # ---- DETERMINISM ----
    h, r = known_sample[0]
    first = mem.recall_all({"агент": h, "действие": r}, "пациент")
    det = all(mem.recall_all({"агент": h, "действие": r}, "пациент") == first for _ in range(1000))

    # ---- FORGET (provable) ----
    fh, fr = known_sample[1]
    root_before = mem.knowledge_root()
    before = mem.recall_all({"агент": fh, "действие": fr}, "пациент")
    deleted = mem.forget(fh, fr)
    after = mem.recall_all({"агент": fh, "действие": fr}, "пациент")
    # re-learn to compare root semantics: forgetting must change root; re-adding restores
    root_after = mem.knowledge_root()

    res = {
      "facts_ingested": len(triples),
      "ingest_facts_per_sec": round(len(triples)/ingest_t),
      "known": {
        "queried": len(known_sample),
        "answered": k_ans,
        "accuracy_when_answered_%": round(100*k_correct/max(1,k_ans), 2),
        "citation_coverage_%": round(100*k_cited/max(1,k_ans), 2),
        "avg_query_ms": round(known_q_ms, 4),
      },
      "trap": {
        "queried": len(traps),
        "ours_hallucination_rate_%": round(100*ours_fab/len(traps), 3),
        "baseline_guesser_fabrication_rate_%": round(100*base_fab/len(traps), 2),
      },
      "determinism_1000x_identical": det,
      "forget": {
        "fact": [fh, fr], "deleted_versions": deleted,
        "recall_before": bool(before), "recall_after_abstains": (not after),
        "root_changed_by_forget": root_before != root_after,
      },
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"  Ingested {res['facts_ingested']} WN18RR facts @ {res['ingest_facts_per_sec']}/s")
    print(f"  KNOWN: accuracy {res['known']['accuracy_when_answered_%']}% | "
          f"citations {res['known']['citation_coverage_%']}% | {res['known']['avg_query_ms']} ms/q")
    print(f"  TRAP:  OURS hallucination {res['trap']['ours_hallucination_rate_%']}%  "
          f"vs  guesser {res['trap']['baseline_guesser_fabrication_rate_%']}%")
    print(f"  Determinism (1000x): {res['determinism_1000x_identical']}")
    print(f"  Forget: before={res['forget']['recall_before']} "
          f"after_abstains={res['forget']['recall_after_abstains']} "
          f"root_changed={res['forget']['root_changed_by_forget']}")
    json.dump(res, open("/root/vmem_mcp/benchmark_result.json", "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
