# verifiable-memory — MCP server

A drop-in **memory for AI agents that cannot hallucinate**. It answers only from
stored facts **with the source cited**, or it **honestly says "I don't know"** —
and every guarantee below is cryptographic or by-construction, not a prompt trick.

> Built on the LOGOS verifiable-memory layer. No GPU, no network, CPU-only.

## Why (the pain)
LLMs and agents store knowledge in weights: they hallucinate, can't cite, can't be
edited, can't forget, can't be audited. That blocks them from any high-stakes use.
This server is the **trust layer behind your agent**: the LLM phrases, this memory
guarantees the facts.

## Guarantees an LLM cannot give from its weights
- **0% hallucination** — exact match only; unknown → honest abstention.
- **Citations** — every answer carries its `source`.
- **Valid-time** — `update_fact` versions a fact; `recall(as_of=t)` answers "as of t"; full `history`.
- **Provable forgetting (GDPR / right-to-be-forgotten)** — `forget` really deletes; signed proof-of-deletion; the Merkle root reverts.
- **Merkle commitment** — `knowledge_root` commits all knowledge in one hash; `prove_fact` proves inclusion without revealing other facts.
- **Contradiction detection** — for functional relations, surfaces conflicting values with both sources (instead of silently picking one).
- **Signed receipts** — `recall`/`forget` return HMAC receipts; `verify_receipt` detects any tampering.
- **Determinism** — same query → same answer + same signature.

## Tools (13)
`learn_fact, recall, update_fact, history, forget, contradictions, knowledge_root,
prove_fact, verify_proof, verify_receipt, multihop, all_paths, stats`

## Run
```bash
pip install verifiable-memory-mcp
verifiable-memory          # speaks MCP over stdio (JSON-RPC)
# or from source:  python3 -m vmem.server
```
Facts persist to `VMEM_STATE` (default `/root/vmem_mcp/state`).

## Use from Claude Desktop / Code (MCP client config)
```json
{
  "mcpServers": {
    "verifiable-memory": {
      "command": "verifiable-memory",
      "args": [],
      "env": { "VMEM_STATE": "/root/vmem_mcp/state" }
    }
  }
}
```
Then the agent can call e.g. `learn_fact`, and later `recall` — getting a cited
answer or an honest abstention, with a verifiable knowledge root.

## Demo (verified)
```
learn_fact France capital Paris   (src Wikipedia)
learn_fact France capital Lyon    (src some-blog)
recall  France capital   -> [Paris@Wikipedia, Lyon@some-blog] + signed receipt
recall  Atlantis capital -> ABSTAIN (no hallucination)
contradictions [capital] -> France: Paris(Wikipedia) vs Lyon(some-blog)
prove_fact France capital -> Merkle inclusion proof
forget  France capital Lyon -> signed proof-of-deletion; root reverts
recall  France capital   -> Paris only (Lyon provably gone)
```

## Benchmark (reproducible: `python3 benchmark.py`)
On **8,000 real WN18RR facts** + a 1,500 **trap set** (absent head/relation pairs):

| Metric | verifiable-memory | naive "always answer" baseline |
|---|---|---|
| Hallucination on absent facts | **0.0%** | 100% |
| Accuracy when answered (known) | **100%** | — |
| Citation coverage | **100%** | — |
| Determinism (same query ×1000) | **identical** | n/a |
| Provable forget (recall→abstain, Merkle root changes) | **yes** | impossible |
| Ingest | ~160k facts/s | — |
| Query latency | **~5.6 µs** | — |

Honest scope: this proves *faithful, fully-cited recall with 0 fabrication, provable
forgetting and determinism* — the verifiability axes. The baseline is a deliberately
naive guesser (proxy for "a system that always answers"); a head-to-head vs a real
RAG+LLM stack is the next step. We do **not** claim broader/ smarter than an LLM.

## Status
MVP. Core verifiable path is exact/symbolic + Merkle (0% hallucination by construction).
Roadmap: hosted API (pay-per-call), PyPI/npm SDK, optional fuzzy-recall (VSA) behind a flag.
