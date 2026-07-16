# Project 7 — Industry-Integrated AI Synthesis

![](https://github.com/user-attachments/assets/01a444d9-3650-40f8-ae0c-3d41c1f75262)

**AI Security Operations Copilot** — an agent-led SOC copilot that ingests surveillance events and access-control logs, correlates them into risk-scored incidents, retrieves relevant policies, generates analyst-facing summaries with `doc_id` citations, and escalates critical cases through a non-bypassable human-approval gate.

This is the **final synthesis capstone** (Project 7) of the Udacity AI Mastery Capstone. It integrates methods from **five** prior projects into one industry-focused solution.

---

## What this is

A working vertical slice that proves a single architectural bet: that **rules-first fusion + RAG-grounded summarization + non-bypassable policy gates** compose into a system SOC analysts would want to be the *closer* on, not the *typist* on. The thesis is operator-centered — the copilot replaces the part of the shift that's a log-stitcher and a 3am paragraph-writer; the operator keeps judgment, override, and accountability.

On the scaled slice (P1's full corpus):
- **3 sites, 1,000 surveillance events, 9,702 access logs**
- **226 risk-scored incidents** (210 high + 16 critical)
- **16 critical escalations** routed through the human-approval gate; the agent never auto-closes
- **34 tests green** (13 governance/policy + 2 escalate-blocked + 3 P2 threshold-calibration + 16 upload-adapter)

The notebook is tested with **Restart & Run All clean (0 errors across 30 cells)**.

---

## Five prior projects, one system

| Project | Contribution to P7 | Evidence |
|---|---|---|
| **P1** (Programming Foundations) | The copilot runs on P1's full corpus. `p1_pipeline.py` reads P1's `data/raw/*.parquet` (1k events / 10k logs / 3 sites) through `p1_adapter.py` to normalize the schema. | `src/generators/p1_pipeline.py`, `src/utils/p1_adapter.py` |
| **P2** (Statistical Analysis) | The rule engine's confidence threshold is **calibrated** against P2. P2's chi-square + t-test are re-run on the slice the copilot runs against. P2's "validate against precision/recall before deployment" caveat is confirmed as a hard finding: **recall@0.85 = 58%** (Cohen's d 0.21 → 1.50). | `tests/test_threshold_calibration.py`, notebook §2b |
| **P3** (Applied ML) | Rules-first discipline + the leakage-audit lesson. Fusion is rules-only; any future ML upgrade must split by the independence unit (`site` + `time_window`), not shuffled rows. | `src/fusion/rules.py`, `src/fusion/risk_scorer.py` |
| **P5** (Generative AI) | The char-level Transformer on CSIRT/CERT/NIST incident-handling prose is the summarizer's **genre reference** (NIST SP 800-61 / CSIRT response style). The summarizer itself is a citation-grounded LLM call, not the P5 model. | `src/agent/summarizer.py` |
| **P6** (Agentic AI) | The LangGraph governance spine (`src/governance/`) is **lifted verbatim** from the property-management donor. The non-bypassable policy gate that hard-blocked P6's eviction/lockout is reused as P7's human-in-the-loop for `risk_band = critical`. | `src/governance/`, `src/domain/` |

---

## Pipeline

```
P1 raw parquet (project_01_.../data/raw/*.parquet)
        │
        ▼  p1_adapter.py (drop sentinels, normalize zone_id, derive anomaly)
surveillance_events (1k) + access_logs (9.7k)
        │
        ▼  src/fusion/  (4 rules: intrusion_restricted, repeated_denials, cross_anomaly, tailgate_door)
226 risk_scored incidents (210 high + 16 critical)
        │
        ▼  tests/test_threshold_calibration.py  (P2 calibration contract)
        │
        ▼  src/rag/  (Chroma + all-MiniLM-L6-v2 + MMR + category routing)
        │     retrieve_for_incident(incident_type) → KB-XXXXX
        │
        ▼  src/agent/summarizer.py  (Groq llama-3.1-8b-instant, citation guard)
        │     summary_text + recommended_action + citation_doc_ids
        │
        ▼  src/agent/copilot_agent.py  (LangGraph)
              fuse → score → retrieve → summarize → escalate(human_approval)
              case.close is hard-blocked (allow: false)
```

The agent's audit log is hash-chained JSONL; every tool call, reviewer verdict, and human-approval is recorded with `doc_id` traceability for the post-incident review.

**Optional analyst front-end:** the same `run_incident_streaming` entry point is reused by `src/gui/app.py` (Streamlit). The GUI surfaces the same audit chain, the same human-approval gate as a real Grant/Deny button, and an optional "📁 Upload your data" path: drop in two CSVs in any schema, map the columns and enum values via the GUI's `data_editor` widgets, and the copilot runs on the analyst's own data instead of the project corpus. Uploads live in `data/uploads/{uuid}/` and never touch the project's `data/synthetic/` parquet.

---

## Quick start (5 minutes)

```powershell
# 1. Activate the project venv
project_07_final_synthesis\final_venv\Scripts\python.exe -m project_07_final_synthesis.src.generators.run_all
project_07_final_synthesis\final_venv\Scripts\python.exe -m project_07_final_synthesis.src.fusion.incidents
project_07_final_synthesis\final_venv\Scripts\python.exe -m project_07_final_synthesis.src.rag.knowledge_base_loader
project_07_final_synthesis\final_venv\Scripts\python.exe -m project_07_final_synthesis.src.rag.retriever --build

# 2. Run the agent on the seeded critical incident
project_07_final_synthesis\final_venv\Scripts\python.exe -m project_07_final_synthesis.src.agent.copilot_agent --incident INC-000001

# 3. Run the test suite
project_07_final_synthesis\final_venv\Scripts\python.exe -m pytest project_07_final_synthesis\tests\ -v
```

For the LLM summarizer, set `GROQ_API_KEY` in `project_07_final_synthesis/.env` (free key at https://console.groq.com/keys). The deterministic pipeline (generators, fusion, RAG, stub-mode agent) runs without any key.

To run the rubric submission notebook:
```powershell
project_07_final_synthesis\final_venv\Scripts\python.exe project_07_final_synthesis\scripts\build_notebook.py
```
Then open `notebooks/07_integrated_copilot.ipynb` in JupyterLab.

To launch the **Streamlit analyst GUI** (one command, runs in your browser):
```powershell
project_07_final_synthesis\final_venv\Scripts\python.exe -m streamlit run project_07_final_synthesis\src\gui\app.py
```
Pick an incident in the sidebar, press **▶ Run copilot**. Critical-band incidents show a real **Grant/Deny** gate before the agent dispatches `incident.escalate`. Default is stub mode (deterministic, no API key). Tick "Use LLM" and paste a `gsk_…` key to enable the LLM summarizer; the GUI never reads `.env` for the key. Use the **📁 Upload your data** section to drop in your own surveillance + access CSVs (any schema; map columns and enum values via the GUI's `data_editor` widgets) and the copilot runs on the analyst's own data instead of the project corpus.

---

## Project structure

```
project_07_final_synthesis/
├── data/
│   ├── reference/                          # sites, zones, devices, users (shared seed data)
│   ├── operational/                        # surveillance_events, access_logs, incidents
│   ├── knowledge_base/                     # JSONL policies + Chroma vector store
│   ├── synthetic/                          # deterministic Parquet + CSV outputs (project corpus)
│   └── uploads/                            # per-session CSV uploads (data/uploads/{uuid}/)
├── notebooks/
│   └── 07_integrated_copilot.ipynb         # rubric submission (30 cells)
├── src/
│   ├── generators/                         # reference_data + surveillance_events + access_logs + p1_pipeline
│   ├── fusion/                             # rules + risk_scorer + incidents (4 detectors)
│   ├── rag/                                # knowledge_base_loader + retriever (MMR + category routing)
│   ├── agent/                              # copilot_agent + summarizer (Groq adapter)
│   ├── governance/                         # lifted from P6 donor (graph_builder, policy, audit, ...)
│   ├── domain/                             # SOC-specific (policy.yaml, tools, intake, prompts)
│   ├── gui/                                # Streamlit analyst GUI (one-file app)
│   ├── schema.py                           # dataclasses + INCIDENTS_COLS etc.
│   └── utils/                              # constants (SEED=42), io, p1_adapter, upload_adapter
├── tests/                                  # 34 tests: governance + policy + escalate + threshold + upload-adapter
├── reports/
│   ├── agent_graph.png                     # compiled LangGraph topology
│   └── Reflective_Synthesis_Paper.pdf
│   └── Reflective_Synthesis_Paper.md
├── final_venv/                             # pinned venv (187 packages; see requirements.txt)
├── requirements.txt                        # pip freeze of final_venv (+ streamlit 1.59.2 stack)
└── README.md                               # this file
```

---

## What the operator sees

The copilot changes the SOC analyst's job from *triage everything* to *verify what matters*. On the scaled slice:

- **Inbox:** 226 ranked incidents (vs. ~1,000 raw events + 9,700 access logs to triage manually)
- **Top of queue:** 16 critical-band incidents, each with: a rule provenance (`intrusion_restricted`, etc.), a risk score, a one-paragraph summary grounded in a policy doc, a concrete recommended action, and a `doc_id` citation for the duty-manager review
- **Escalation:** every critical routes through the human-approval gate; the agent never auto-closes
- **Audit:** the JSONL hash chain records every tool call, reviewer verdict, and human approval — defensible in a post-incident review
- **GUI:** the Streamlit analyst surface (`src/gui/app.py`) puts all of the above into one screen: pick an incident in the sidebar, press **▶ Run copilot**, watch the audit trail fill in real time, click **Grant** or **Deny** when a critical incident pauses for sign-off. The GUI reuses the same `run_incident_streaming` entry point the CLI uses — no separate API surface.

The defense line that lands: *"The copilot doesn't replace the operator. It replaces the part of the shift that's a typist, a log-stitcher, and a 3am paragraph-writer. The operator keeps the part that matters: judgment, override, and accountability."*

---

## Human-in-the-loop gate

The `human_approval` node in the governance spine is the **only** place that can require a human's real sign-off, and it does so for `incident.escalate` on critical-band incidents. Two paths exist:

- **CLI / notebook / pytest (default):** the node auto-approves and logs `approver="notebook_operator"` in the audit chain. Reproducible, no UI needed.
- **GUI (`COPILOT_HUMAN_GATE=1`):** the node calls `langgraph.types.interrupt(...)` and the graph suspends. The GUI shows a Grant/Deny panel; the analyst's decision resumes the graph via `Command(resume=granted)`. The audit chain records `approver="human_via_interrupt"` and `granted=...`. **The audit chain tells you which path fired for any given row.**

The same env-var pattern (`P7_INCIDENTS_DIR`) lets the GUI route the agent's `incident.summarize` / `incident.score` / `incident.fuse` calls to a per-session upload parquet (`data/uploads/{uuid}/incidents.parquet`) when an analyst uploads their own data — so the agent's summarizer is always prompted with the right row, never a coincidentally-matched synthetic one. The CLI / notebook paths leave the env var unset and use the project synthetic/ parquet as before. Two env vars, two opt-in overrides, zero new abstractions.

---

## Honest limitations

- **Threshold finding (hard):** the fusion threshold (0.85) catches only ~58% of anomaly events on the scaled slice. P2's calibration warning is confirmed as a hard finding. The seed-critical injection is what lets the escalation gate run end-to-end; a real deployment would need to lower the threshold or add a complementary low-confidence rule. Documented in `tests/test_threshold_calibration.py` + notebook §2b.
- **Scale:** still synthetic (P1's corpus), not a real SOC environment. Multi-vendor event-stream connectors are the next 12-18 months of work.
- **Free model:** Groq `llama-3.1-8b-instant` is fast and free but a smaller model. The citation guard is what makes "hallucinated policies are bugs" enforceable on it.
- **One spine fix:** the multi-step plan-loop bug was latent in the P6 donor (all P6 scenarios were single-step); fixed in `src/governance/graph_builder.py` and `src/governance/nodes.py`.
- **GUI is local-only:** the Streamlit server has no auth. Fine for an analyst-facing dev tool, not fine for multi-tenant production. The "Use LLM" toggle is off by default and the GUI never reads `.env` for the Groq key — the typed key is the only source. If you expose beyond localhost, add auth (reverse proxy + OAuth is the standard answer) and re-evaluate the trust model for the per-session upload path.
- **User uploads are scoped to a single GUI session, not multi-tenant.** Uploads live in `data/uploads/{uuid}/` and are visible to anyone with shell access to the same machine. The trust boundary is the local Streamlit server.

---

## References

- **Prior projects:** ["../project_01_reproducible_workflows/"](https://github.com/Soliman2020/master_capstone_01), ["../project_02_statistical_analysis/"](https://github.com/Soliman2020/master_capstone_02), ["../project_03_ML/"](https://github.com/Soliman2020/master_capstone_03), ["../project_05_generative_ai/"](https://github.com/Soliman2020/master_capstone_05), ["../project_06_agentic_ai/"](https://github.com/Soliman2020/master_capstone_06)
- **P5 corpus source:** NIST SP 800-61 Rev. 2 & Rev. 3, ENISA publications, JPCERT/CC English (per-source licenses in ["project_05_generative_ai/src/dataset.py"](https://github.com/Soliman2020/master_capstone_05/blob/master/src/dataset.py) → SOURCE_CATALOG)
- **P3 leakage-audit lesson:** ["project_03_ML/notebooks/modeling.ipynb"](https://github.com/Soliman2020/master_capstone_03/blob/master/notebooks/modeling.ipynb) §"A second look"
- **P6 donor:** the property-management LangGraph agent whose ["src/governance/"](https://github.com/Soliman2020/master_capstone_06/tree/master/src/governance) was lifted into P7
- **Program docs:** the `docs/` directory at the repo root (`01_capstone_program_guide.md` through `07_full_project_bundle_index.md`)

