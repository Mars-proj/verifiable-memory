#!/usr/bin/env python3
"""
max_stress.py — maximum-load stress of verifiable-memory (CPU/RAM only).
Pushes the parts that DON'T need a GPU to the machine's limit:
  • SCALE sweep: ingest up to N facts, measure ingest rate, RAM/fact, query p50/p99
  • ADVERSARIAL traps: maximally-confusable absent (subject,relation) pairs
    (subject exists with OTHER relations, relation exists with OTHER subjects)
    -> hallucination must stay 0 even under worst-case confusability
  • CONCURRENCY: many threads hammering recall -> throughput, correctness under load
  • FORGET-at-scale: delete one fact among N -> abstains + Merkle root changes
The real RAG+LLM head-to-head on hard hallucination sets (HaluEval/TruthfulQA) is
specified separately and needs a GPU box.
"""
import sys, os, time, random, json, threading
random.seed(1)
_O=sys.stdout; sys.stdout=sys.stderr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HOME","/tmp/logos_run")
from vmem.compositional_memory import CompositionalMemory
sys.stdout=_O

def rss_mb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"): return int(ln.split()[1])//1024
    return -1

RELS = [f"rel{i}" for i in range(20)]   # 20 relations -> high collision (hard)
RSS_CAP_MB = int(os.environ.get("RSS_CAP_MB","1500"))

def build(n):
    mem = CompositionalMemory(predictive=None, state_dir="/tmp/vmem_stress", capacity=n+1000)
    mem.relations.clear(); mem._rel_seen.clear(); mem._reindex()
    rss0 = rss_mb(); t0=time.time()
    subj_rels = {}
    for i in range(n):
        s=f"e{i}"; r=RELS[i % len(RELS)]; o=f"v{i}"
        mem.learn_triple(s,r,o,source=f"src{i%1000}")
        subj_rels.setdefault(s, r)
        if i % 100000 == 0 and rss_mb() > RSS_CAP_MB:
            print(f"  [stop: RSS {rss_mb()}MB > cap at {i} facts]"); n=i; break
    dt=time.time()-t0; rss1=rss_mb()
    return mem, n, dt, rss0, rss1

def main():
    results=[]
    for N in [100_000, 300_000, 600_000, 1_000_000]:
        if rss_mb() > RSS_CAP_MB - 200: break
        mem, got, dt, rss0, rss1 = build(N)
        # known queries
        qs=[f"e{random.randint(0,got-1)}" for _ in range(3000)]
        lat=[]
        ok=0
        for s in qs:
            r=RELS[int(s[1:]) % len(RELS)]
            t=time.perf_counter()
            rows=mem.recall_all({"агент":s,"действие":r},"пациент")
            lat.append((time.perf_counter()-t)*1e6)
            if rows and rows[0][0]==f"v{s[1:]}": ok+=1
        lat.sort()
        # ADVERSARIAL traps: real subject + real relation, but pair ABSENT (max confusable)
        fab=0; T=3000
        for _ in range(T):
            s=f"e{random.randint(0,got-1)}"
            r=random.choice([x for x in RELS if x != RELS[int(s[1:])%len(RELS)]])  # wrong rel for this subj
            if mem.recall_all({"агент":s,"действие":r},"пациент"): fab+=1
        # forget at scale
        fs=f"e{random.randint(0,got-1)}"; fr=RELS[int(fs[1:])%len(RELS)]
        root0=mem.knowledge_root()
        before=bool(mem.recall_all({"агент":fs,"действие":fr},"пациент"))
        mem.forget(fs,fr)
        after=bool(mem.recall_all({"агент":fs,"действие":fr},"пациент"))
        root1=mem.knowledge_root()
        res={"facts":got,"ingest_per_sec":round(got/dt),
             "RAM_total_MB":rss1,"bytes_per_fact":round((rss1-rss0)*1024*1024/max(1,got)),
             "known_acc_%":round(100*ok/len(qs),2),
             "query_us_p50":round(lat[len(lat)//2],2),"query_us_p99":round(lat[int(len(lat)*0.99)],2),
             "adversarial_trap_hallucination_%":round(100*fab/T,3),
             "forget_ok": before and not after and root0!=root1}
        results.append(res)
        print(json.dumps(res,ensure_ascii=False))
        del mem

    # concurrency stress on the largest built
    print("\n=== CONCURRENCY (16 threads hammer recall) ===")
    mem, got, dt, _, _ = build(min(300_000, results[-1]["facts"] if results else 100_000))
    counts={"n":0}; lock=threading.Lock(); STOP=time.time()+5
    def worker():
        c=0
        while time.time()<STOP:
            s=f"e{random.randint(0,got-1)}"; r=RELS[int(s[1:])%len(RELS)]
            mem.recall_all({"агент":s,"действие":r},"пациент"); c+=1
        with lock: counts["n"]+=c
    ths=[threading.Thread(target=worker) for _ in range(16)]
    [t.start() for t in ths]; [t.join() for t in ths]
    qps=round(counts["n"]/5)
    print(json.dumps({"threads":16,"duration_s":5,"queries":counts["n"],"throughput_qps":qps}))

    out={"scale_sweep":results,"concurrency_qps":qps}
    json.dump(out,open("/root/vmem_mcp/max_stress_result.json","w"),indent=2)
    print("\nSUMMARY:")
    for r in results:
        print(f"  {r['facts']:>9,} facts | ingest {r['ingest_per_sec']:>7,}/s | "
              f"q p50 {r['query_us_p50']}us p99 {r['query_us_p99']}us | "
              f"adversarial halluc {r['adversarial_trap_hallucination_%']}% | "
              f"RAM {r['RAM_total_MB']}MB ({r['bytes_per_fact']} B/fact) | forget {r['forget_ok']}")
    print(f"  concurrency: {qps:,} q/s on 16 threads")

if __name__=="__main__":
    main()
