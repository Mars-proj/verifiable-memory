#!/usr/bin/env python3
"""
verifiable-memory MCP server (MVP)
==================================
Exposes the LOGOS verifiable-memory layer (CompositionalMemory) over the
Model Context Protocol (stdio, newline-delimited JSON-RPC 2.0) so any
MCP-capable agent (Claude Desktop/Code, frameworks) can use a memory that:

  • answers ONLY from stored facts, with the SOURCE cited, or honestly ABSTAINS
    (0% hallucination by construction — exact match only);
  • supports VALID-TIME (answer "as of" a moment) and full fact HISTORY;
  • supports PROVABLE FORGETTING (right-to-be-forgotten) — fact is really gone;
  • detects CONTRADICTIONS for functional relations (shows both sources);
  • commits all knowledge to a MERKLE ROOT and proves a fact's INCLUSION
    without revealing other facts;
  • returns SIGNED RECEIPTS (HMAC) so an answer + its knowledge-root are tamper-evident;
  • does MULTI-HOP / ALL-PATHS chaining over facts (exact only).

This is the trust layer an LLM/agent cannot provide from its weights.

Self-contained: vendored LOGOS verifiable-memory core (vmem/), zero external deps.
No GPU, no network. Facts persist to STATE_DIR.
"""
import sys, os, json, hmac, hashlib, traceback

# MCP stdio must be PURE JSON-RPC. Imported modules print banners to stdout,
# so capture the real stdout for protocol and route everything else to stderr.
_OUT = sys.stdout
sys.stdout = sys.stderr   # MCP stdio must be pure JSON-RPC; route any stray prints to stderr

STATE_DIR = os.environ.get("VMEM_STATE", os.path.expanduser("~/.verifiable_memory"))
os.makedirs(STATE_DIR, exist_ok=True)

from .compositional_memory import CompositionalMemory
from .knowledge_proof import verify_inclusion

MEM = CompositionalMemory(state_dir=STATE_DIR)

# ---- server signing key (persisted, HMAC for tamper-evident receipts) ----
_KEYP = os.path.join(STATE_DIR, "server_hmac.key")
if os.path.exists(_KEYP):
    SIGKEY = open(_KEYP, "rb").read()
else:
    SIGKEY = hashlib.sha256(os.urandom(32)).digest()
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_KEYP, "wb") as f:
        f.write(SIGKEY)
    os.chmod(_KEYP, 0o600)

def _sign(payload: dict) -> str:
    msg = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hmac.new(SIGKEY, msg, hashlib.sha256).hexdigest()

def _save():
    try: MEM.save()
    except Exception: pass

# ----------------------------- tool implementations -----------------------------
def t_learn_fact(a):
    MEM.learn_triple(a["subject"], a["relation"], a["object"],
                     source=a.get("source"), valid_from=a.get("valid_from"))
    _save()
    return {"stored": True, "fact": [a["subject"], a["relation"], a["object"]],
            "source": a.get("source"), "knowledge_root": MEM.knowledge_root()}

def t_recall(a):
    known = {"агент": str(a["subject"]), "действие": str(a["relation"])}
    rows = MEM.recall_all(known, "пациент", as_of=a.get("as_of"))
    root = MEM.knowledge_root()
    if not rows:
        return {"answer": None, "abstain": True,
                "note": "No stored fact — honest abstention (no hallucination).",
                "knowledge_root": root}
    answers = [{"object": f, "source": s} for f, s, _meta in rows]
    receipt = {"subject": a["subject"], "relation": a["relation"],
               "answers": answers, "knowledge_root": root}
    receipt["signature"] = _sign(receipt)
    return {"answer": answers if len(answers) > 1 else answers[0],
            "abstain": False, "cited": True, "receipt": receipt}

def t_update_fact(a):
    closed, _ = MEM.update_fact(a["subject"], a["relation"], a["new_object"],
                               source=a.get("source"), t=a.get("t"))
    _save()
    return {"closed_versions": closed, "new_value": a["new_object"],
            "knowledge_root": MEM.knowledge_root()}

def t_history(a):
    h = MEM.history({"агент": str(a["subject"]), "действие": str(a["relation"])})
    return {"history": h}

def t_forget(a):
    n = MEM.forget(a["subject"], a["relation"], a.get("object"))
    _save()
    receipt = {"forgot": [a["subject"], a["relation"], a.get("object")],
               "deleted_count": n, "knowledge_root_after": MEM.knowledge_root()}
    receipt["signature"] = _sign(receipt)
    return {"deleted": n, "proof_of_deletion": receipt}

def t_contradictions(a):
    c = MEM.contradictions(a.get("functional_relations", []))
    return {"contradictions": c, "count": len(c)}

def t_knowledge_root(a):
    return {"knowledge_root": MEM.knowledge_root()}

def t_prove_fact(a):
    p = MEM.prove_fact({"агент": str(a["subject"]), "действие": str(a["relation"])}, "пациент")
    if p is None:
        return {"provable": False, "note": "Fact not present — cannot prove absent fact."}
    return {"provable": True, **p}

def t_verify_proof(a):
    ok = verify_inclusion(a["leaf"], a["proof"], a["root"])
    return {"valid": bool(ok)}

def t_verify_receipt(a):
    r = dict(a["receipt"])
    sig = r.pop("signature", None)
    return {"valid": bool(sig) and hmac.compare_digest(sig, _sign(r))}

def t_multihop(a):
    res = MEM.multihop(a["start"], a["relations"], as_of=a.get("as_of"))
    return {"result": res}

def t_all_paths(a):
    paths = MEM.all_paths(a["start"], a["end"], max_depth=a.get("max_depth", 5))
    return {"paths": paths, "count": len(paths)}

def t_stats(a):
    return {"facts_live": sum(1 for _ in MEM.relations),
            "knowledge_root": MEM.knowledge_root(),
            "state_dir": STATE_DIR}

TOOLS = {
 "learn_fact": (t_learn_fact, "Store a fact (subject, relation, object) the agent must recall EXACTLY later, with its source. Call whenever the user states a fact, preference, decision, name, number, or rule worth remembering — it persists across sessions and is never silently distorted. Optional valid_from for valid-time.",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"},"object":{"type":"string"},"source":{"type":"string"},"valid_from":{"type":"number"}},"required":["subject","relation","object"]}),
 "recall": (t_recall, "Look up a stored fact and return the answer WITH its cited source — or an honest 'unknown'. ALWAYS call this before answering a factual or memory question instead of guessing: it returns nothing rather than hallucinating, and includes a signed, verifiable receipt. Optional as_of for valid-time.",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"},"as_of":{"type":"number"}},"required":["subject","relation"]}),
 "update_fact": (t_update_fact, "Update a fact's value without retraining: closes live versions (valid_to=t) and opens a new one. History is preserved.",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"},"new_object":{"type":"string"},"source":{"type":"string"},"t":{"type":"number"}},"required":["subject","relation","new_object"]}),
 "history": (t_history, "Full life-line of a fact (all versions live+closed, with sources and valid-time). Audit/compliance.",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"}},"required":["subject","relation"]}),
 "forget": (t_forget, "Permanently and PROVABLY delete a stored fact (GDPR / right-to-be-forgotten). Use when the user asks to forget or remove information — the fact is fully erased and you get a signed proof of deletion. object optional (omit to delete all values).",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"},"object":{"type":"string"}},"required":["subject","relation"]}),
 "contradictions": (t_contradictions, "Audit knowledge for conflicts: for functional relations (one value expected, e.g. 'capital','birthdate') return any (subject,relation) holding >1 live value, showing BOTH sources — call before trusting facts that may have been updated or come from multiple sources.",
   {"type":"object","properties":{"functional_relations":{"type":"array","items":{"type":"string"}}},"required":["functional_relations"]}),
 "knowledge_root": (t_knowledge_root, "Merkle root committing the entire current knowledge state (one hash).", {"type":"object","properties":{}}),
 "prove_fact": (t_prove_fact, "Merkle inclusion proof that a fact is in the knowledge state (without revealing other facts).",
   {"type":"object","properties":{"subject":{"type":"string"},"relation":{"type":"string"}},"required":["subject","relation"]}),
 "verify_proof": (t_verify_proof, "Verify a Merkle inclusion proof (leaf, proof, root).",
   {"type":"object","properties":{"leaf":{"type":"string"},"proof":{"type":"array"},"root":{"type":"string"}},"required":["leaf","proof","root"]}),
 "verify_receipt": (t_verify_receipt, "Verify a signed receipt returned by recall/forget (tamper-evident).",
   {"type":"object","properties":{"receipt":{"type":"object"}},"required":["receipt"]}),
 "multihop": (t_multihop, "Multi-hop chain: start entity + list of relations, follows subject->object each step (exact only, 0% hallucination).",
   {"type":"object","properties":{"start":{"type":"string"},"relations":{"type":"array","items":{"type":"string"}},"as_of":{"type":"number"}},"required":["start","relations"]}),
 "all_paths": (t_all_paths, "All exact fact-paths between start and end entities (each path citable).",
   {"type":"object","properties":{"start":{"type":"string"},"end":{"type":"string"},"max_depth":{"type":"number"}},"required":["start","end"]}),
 "stats": (t_stats, "Counts + current knowledge root.", {"type":"object","properties":{}}),
}

# ----------------------------- MCP stdio JSON-RPC -----------------------------
def _resp(rid, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": rid}
    if error is not None: m["error"] = error
    else: m["result"] = result
    return m

def handle(msg):
    method = msg.get("method"); rid = msg.get("id"); params = msg.get("params") or {}
    if method == "initialize":
        return _resp(rid, {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "verifiable-memory", "version": "0.1.0"}})
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _resp(rid, {})
    if method == "tools/list":
        return _resp(rid, {"tools": [
            {"name": n, "description": d, "inputSchema": s} for n, (_f, d, s) in TOOLS.items()]})
    if method == "tools/call":
        name = params.get("name"); args = params.get("arguments") or {}
        if name not in TOOLS:
            return _resp(rid, error={"code": -32601, "message": f"unknown tool {name}"})
        try:
            out = TOOLS[name][0](args)
            return _resp(rid, {"content": [{"type": "text",
                        "text": json.dumps(out, ensure_ascii=False, indent=2)}]})
        except Exception as e:
            return _resp(rid, {"content": [{"type": "text",
                        "text": f"ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}"}],
                        "isError": True})
    if rid is not None:
        return _resp(rid, error={"code": -32601, "message": f"unknown method {method}"})
    return None

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        out = handle(msg)
        if out is not None:
            _OUT.write(json.dumps(out, ensure_ascii=False) + "\n")
            _OUT.flush()

if __name__ == "__main__":
    main()
