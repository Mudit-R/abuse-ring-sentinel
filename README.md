# abuse-ring-sentinel

A graph ML system I built to detect coordinated fraud rings in payment data — promo abuse syndicates, return fraud loops, chargeback collusion clusters, and ATO surges.

The core idea: tabular models (XGBoost, LightGBM) are genuinely good at catching individual bad actors. They fail completely when 20 fake accounts coordinate together because each one looks totally fine in isolation. You need to look at the *graph* of who's connected to whom.

Built and tested on PaySim (6.36M transactions, 3.28M accounts). Full benchmark numbers below.

---

## Why I built this

I was looking at fraud detection papers and kept running into the same problem: most systems treat each transaction or account independently. But real fraud rings don't work that way — they share devices, delivery addresses, promo codes, and UPI IDs. The signal is in the *connections*, not the individual data points.

I wanted to see if I could actually close that gap — a system that catches the ring topology *and* still runs fast enough to sit in a real payment gateway (< 15ms SLA).

Spoiler: you can, but the latency problem requires a nearline Redis caching layer, not real-time graph traversal.

---

## What it does

Two models work together:

**Stage 1 — XGBoost (22 engineered features)**  
Catches obvious individual signals: balance drain ratios, transaction velocity spikes, out/in degree ratios, temporal clustering. Fast (~0.5ms).

**Stage 2 — Graph Attention Network (GATv2)**  
Looks at the 2-hop neighborhood around each account. Identifies multi-account convergence patterns — e.g. 15 accounts all funneling money to the same exit node within 4 hours. The GNN embeddings are pre-computed and cached in Redis, so serving latency stays sub-1ms instead of the ~85ms it would cost to run live graph traversal.

Both stages have to agree before escalating. This eliminates most false positives.

---

## Numbers

Evaluated on held-out temporal test split (no leakage — see `tests/test_leakage.py`):

| Model | PR-AUC | ROC-AUC | Precision@100 | Latency |
|---|---|---|---|---|
| Logistic Regression | 0.0715 | 0.6948 | 1.0% | < 1.2ms |
| XGBoost standalone | 0.0861 | 0.8725 | 92.0% | < 5.8ms |
| GAT standalone | 0.0448 | **0.9129** | 13.0% | ~85ms |
| **Two-Stage Cascade** | **0.0892** | **0.9085** | **92.0%** | **0.78ms** |

XGBoost standalone already hits 92% Precision@100 but completely misses distributed rings. GAT catches the rings but is too slow for production and gets overwhelmed by class imbalance (0.13% fraud rate). The cascade gets both.

### Adversarial test

I manually constructed a "low-and-slow" ring: 25 accounts, each making ₹20k transactions (below any obvious threshold), all attached to 100 clean decoy accounts to blend in.

- XGBoost: **0% catch rate** — every account looks clean individually
- GAT: **84% catch rate** — picks up the convergence pattern, misses 4 decoy-blurred accounts
- CARE-GNN (with similarity-based neighbor pruning): **92% catch rate**

### Cost model

Using ₹350 per false positive (analyst review time) and ₹42,000 per false negative (average ring payout before detection):

| System | Expected Cost / 10k Tx | Latency | Production-ready? |
|---|---|---|---|
| Logistic Regression | ₹474,369 | 1.2ms | ✅ (but terrible cost) |
| **GAT graph alone** | **₹546,700** | **~85ms** | ❌ (too slow + worst cost) |
| XGBoost alone | ₹91,172 | 5.8ms | ✅ |
| **Two-Stage Cascade** | **₹77,107** | **0.78ms** | ✅ |

The most counterintuitive result: the graph model *alone* is actually the worst performer on cost (₹546,700 — worse than logistic regression). Here's why.

Under 0.13% fraud prevalence, 99.87% of accounts are clean. When you run graph message passing on this, the signal from the tiny minority of fraud nodes gets averaged out by the massive majority of clean neighbors. The model effectively learns to predict "clean" for everything — 0% recall, 13 fraud cases missed per 10k transactions, 13 × ₹42,000 = ₹546,000 in missed payouts.

It also runs at ~85ms per transaction. Payment gateways need < 15ms. So it fails on both dimensions individually.

The cascade fixes both problems:
- XGBoost handles the recall problem (catches individual fraud signals)
- Graph handles the ring problem (catches coordinated multi-account collusion)
- Redis nearline caching fixes the latency (pre-computed embeddings, O(1) lookup at 0.78ms)

The ₹14,065 improvement over standalone XGBoost (₹91,172 → ₹77,107) comes entirely from the graph catching coordinated ring cases that XGBoost misses — specifically accounts that look individually clean but are funneling money to a shared exit node.

---

## Architecture

```
src/
├── models/
│   ├── gat.py              # GATv2 multi-head attention
│   ├── hgt.py              # Heterogeneous Graph Transformer (multi-entity)
│   ├── pc_gnn.py           # PC-GNN: handles extreme class imbalance
│   ├── care_gnn.py         # CARE-GNN: camouflage defense
│   └── contrastive.py      # InfoNCE loss for ring representation learning
├── api/
│   ├── main.py             # FastAPI serving layer
│   ├── schemas.py          # Pydantic request/response models
│   └── razorpay_simulator.py  # payment stream simulator
├── explainability/
│   ├── briefing_engine.py  # LLM tactical briefings per alert
│   └── counterfactual.py   # "what would make this account safe" engine
├── evaluation/
│   ├── adversarial.py      # low-and-slow ring evasion benchmark
│   └── calibration.py      # isotonic regression calibration
├── cache/
│   └── redis_client.py     # nearline embedding store
├── features/
│   └── engineer.py         # 22-feature tensor construction
└── drift/
    └── psi.py              # population stability index monitor
```

---

## Research models used

Three papers meaningfully changed the results:

**PC-GNN (Liu et al., WWW 2021)** — Pick-and-Choose GNN  
Standard message passing dilutes minority fraud signal when 99.87% of nodes are clean. PC-GNN does label-aware balanced sampling during training. Lifted GNN minority PR-AUC from 0.0448 → 0.0825 (+42.8%).

**CARE-GNN (Dou et al., CIKM 2020)** — Camouflage-aware GNN  
Fraud rings deliberately attach to clean accounts to dilute their neighbor signal. CARE-GNN prunes low-similarity edges before aggregation. Boosted adversarial catch rate from 84% → 92%.

**InfoNCE Contrastive Loss (adapted from DiG-In-GNN, AAAI 2024)**  
Pulls fraud ring embeddings together in representation space, pushes them away from clean nodes. Makes the classifier boundary cleaner under imbalance.

---

## Running it

```bash
pip install -r requirements.txt

# reproduce all benchmark numbers
python scripts/reproduce_benchmark.py

# run the tier 4 research benchmarks (PC-GNN, CARE-GNN)
python scripts/evaluate_tier4_research.py

# adversarial evasion test
python scripts/evaluate_adversarial.py

# feature ablation study
python scripts/run_ablation_study.py

# start the API + web console
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
# then open http://localhost:8001/console.html
```

```bash
# run the full test suite (45 tests)
python -m pytest tests/ -v
```

---

## Web console

The console has four panels:

**Risk Evaluator** — Enter an account ID or use the presets (Syndicate Mule, Smurfing Ring, ATO Surge, Clean Retail). Returns fraud probability, a 4-way sub-risk breakdown (promo/return/chargeback/ATO), SHAP feature attribution bars, an LLM investigator briefing, and a counterfactual recommendation.

**Graph Topology Sandbox** — 3D WebGL view of transaction networks (Three.js), plus a 2D canvas mode with a time-scrubber showing ring formation over 72 hours (Seed → Velocity Burst → Drain).

**Live Payment Stream** — Simulated high-throughput feed with planted ring injection buttons.

**PSI Drift Monitor** — Population Stability Index covariate shift tracker. Triggers alerts when feature distributions drift past retraining thresholds.

---

## Important notes

**No auto-blocking.** Every model output is advisory. The system produces a risk tier (LOW/MEDIUM/HIGH/CRITICAL) and recommended action. Actual account actions require explicit human analyst confirmation, logged to a durable audit trail.

**Threshold math.** The Bayes-optimal decision boundary for a single transaction is T* ≈ 0.83% (C_FP=₹350, C_FN=₹42k). In practice, flagging everything above 0.83% would produce thousands of alerts per day, overwhelming any analyst team. The cascade operates at T*=0.42, which minimizes total expected cost under realistic investigation capacity.

**PaySim disclosure.** PaySim simulates individual fraud transactions, not native multi-agent collusion. The ring topology we evaluate against is constructed by clustering accounts connected to shared fraud exit nodes — this is documented in the benchmark scripts.

---

## Things that didn't work (and why)

**Real-time GNN inference** — Running graph aggregation on every payment event took ~85ms per request. Non-starter for payment gateways. The fix was nearline Redis caching: GNN embeddings are computed in background batches and served from cache at < 1ms.

**LightGBM had the highest recall (93.2%) but lowest Precision@100 (2.0%)** — It catches nearly everything but produces enormous false positive rates. Useful as a coarse first-pass filter, not as a production alert system.

**Standard GCN/GraphSAGE on PaySim** — Both collapsed to 0% recall under class imbalance. Without minority-aware sampling (PC-GNN) or explicit ring-structure labels, isotropic message passing averages away the fraud signal.

---

## Tests

```
tests/test_api.py           — 15 API endpoint tests
tests/test_features.py      — 11 feature engineering tests  
tests/test_graph_builder.py — 9 graph construction tests
tests/test_leakage.py       — 4 temporal/structural leakage checks
tests/test_tier4_models.py  — 6 PC-GNN, CARE-GNN, FraudHGT, InfoNCE tests
```

All 45 pass. The leakage tests are probably the most important — they verify that no test-set edges exist in the training graph and that ground truth labels aren't leaked into input features.

---

*Built by Mudit*
