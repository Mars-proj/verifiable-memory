# verifiable-memory

**Memory for AI agents that cannot hallucinate.**
It answers *only* from stored facts — with the source cited — or it honestly says **"I don't know."** Every guarantee below is cryptographic or true by construction, not a prompt trick.

![hallucination 0%](https://img.shields.io/badge/hallucination-0%25-brightgreen)
![CPU only](https://img.shields.io/badge/CPU--only-no%20GPU-blue)
![deps](https://img.shields.io/badge/dependencies-zero-success)
![license](https://img.shields.io/badge/license-MIT-black)
![protocol](https://img.shields.io/badge/MCP-stdio-purple)
![pypi](https://img.shields.io/pypi/v/verifiable-memory-mcp)

> An MCP server + Python SDK. Plug it into any agent (Claude Desktop/Code, LangChain, custom). The LLM phrases; this layer guarantees the facts.

---

## The problem
LLMs store knowledge in weights. So they **hallucinate**, can't **cite**, can't be **edited**, can't **forget**, can't be **audited**. That blocks agents from any high-stakes use — legal, finance, healthcare, compliance, autonomous workflows.

## What you get (an LLM cannot do these from its weights)
- **0% hallucination** — exact match only; unknown → honest abstention.
- **Citations** — every answer carries its `source`.
- **Provable forgetting** (GDPR / right-to-be-forgotten) — the fact is *really* gone; signed proof; Merkle root reverts.
- **Valid-time** — version a fact; ask "as of date T"; full history.
- **Merkle proofs** — commit all knowledge to one hash; prove a fact's inclusion without revealing the rest.
- **Contradiction detection** — surfaces conflicting values *with both sources* instead of silently picking one.
- **Signed receipts** + **determinism** — tamper-evident, same query → same answer.

## Benchmark (reproducible — `python3 benchmark.py`)
Stress-tested to **1,000,000 facts** on a 7 GB CPU box, no GPU:

| Metric | verifiable-memory |
|---|---|
| Hallucination on adversarial traps | **0.0%** |
| Accuracy when answered / citations | **100% / 100%** |
| Query latency (p50 / p99) | **4.4 µs / 14 µs** |
| Throughput | **137,000 q/s** (16 threads) |
| Memory | ~1.2 GB for 1M facts (~1 KB/fact) |
| Provable forget | ✅ root reverts |

vs a naive "always answer" baseline: **0% vs 100% fabrication** on the same traps.

## Install
```bash
pip install verifiable-memory-mcp
verifiable-memory                     # MCP server over stdio
# from source:
git clone https://github.com/Mars-proj/verifiable-memory && cd verifiable-memory
python3 -m vmem.server
```

## Use from Claude Desktop / Code
```json
{
  "mcpServers": {
    "verifiable-memory": {
      "command": "verifiable-memory",
      "args": [],
      "env": { "VMEM_STATE": "~/.verifiable_memory" }
    }
  }
}
```
Then your agent can `learn_fact`, `recall` (cited or abstains), `forget` (provably), `prove_fact`, `contradictions`, `multihop`, and more — 13 tools.

## How it works (1 line)
Facts are stored as data (subject, relation, object + source), indexed for O(1) exact recall; answers are exact-match-or-abstain; the knowledge state commits to a Merkle root. No vectors needed for the verifiable path → 0 fabrication *by construction*.

## Honest scope
This is a **memory / trust layer**, not a reasoning engine and not a better chatbot. It wins on verifiability (cite-or-abstain, forget, determinism, audit), not on open-ended fluency. Pair it with your LLM: LLM = language, this = ground truth.

---

## 🤝 Using this in production?
Need a **hosted API**, on-prem deployment, or help integrating verifiable memory into your agent (legal / fintech / healthcare / agent platforms)?
**→ Pilot & enterprise: Sergey · svobodg@gmail.com**

MIT licensed. PRs welcome.
