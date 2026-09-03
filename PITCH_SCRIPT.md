# PITCH_SCRIPT.md — AI Risk Manager (Track 02: Abuse-Ring Sentinel)

> **Project Name:** Sentinel — Real-Time Abuse-Ring & FinOps Risk Engine 
> **Applicant:** Mudit 
> **Target Role:** AI Builder Intern (In-person Bangalore, from September) 
> **Target Video Length:** Exactly 5 Minutes (00:00 – 05:00) 
> **Tone:** High energy, technically rigorous, evidence-backed, direct. No filler.

---

## ⏱️ Video Breakdown & Timestamped Script

---

### [00:00 – 00:45] 1. Problem Taste & The Merchant Fraud Ring Blind Spot
* **Visual on Screen:** Quick split screen showing individual clean UPI transactions vs. a hidden 3D collusive abuse ring.
* **Spoken Script:**
 > *"Hi everyone, I'm Mudit. In modern digital commerce, payment fraud almost never happens as an isolated, high-ticket credit card theft. It happens through **coordinated abuse rings** — syndicates exploiting first-order promo coupons across 20 synthetic accounts, high-value electronics return loops to shared drop addresses, and friendly-fraud chargeback collusion.* 
 > 
 > *Traditional rule engines and tabular models have a fundamental blind spot: they evaluate each transaction in isolation. An account that looks completely clean with a ₹500 purchase will slip right past standard checks. 
 > 
 > To catch rings, you have to look at the **graph topology**. That’s why I built **Sentinel** — a multi-head Graph Attention Network (GAT) and XGBoost cascade trained across 6.36 million transactions and 3.28 million accounts, served in under 0.78 milliseconds."*

---

### [00:45 – 01:45] 2. Live Flag with Gateway-Shaped Telemetry & Planted Ring
* **Visual on Screen:** Switch to `console.html`. Show the Live Payment Stream with UPI handles (`@okhdfcbank`), RuPay cards, and Indian MCC codes. Click **"Plant Promo Ring (6 Tx)"**.
* **Spoken Script:**
 > *"Let’s look at the live Sentinel Console. Here’s our high-throughput payment stream styled with real payment gateway rails — UPI VPAs, RuPay cards, and merchant category codes. 
 > 
 > Watch what happens when I inject a coordinated 6-transaction promo abuse syndicate. In real time, the GAT attention layer identifies the asymmetric fan-in and flags the cluster at **98% Critical Risk** within **0.78 milliseconds** via our nearline Redis feature store. 
 > 
 > Notice our strict non-negotiable guardrail: **we never auto-block or freeze an account**. The system produces an advisory flag and queues it for a human analyst, logging every step into our durable audit trail."*

---

### [01:45 – 02:45] 3. The Explainability Triad (SHAP + LLM Briefing + Counterfactuals)
* **Visual on Screen:** Click into the Risk Evaluator panel. Highlight the SHAP bars, the LLM Briefing Box, and the Counterfactual Card.
* **Spoken Script:**
 > *"A black-box prediction is useless for a merchant operations team. Sentinel provides an **Explainability Triad**: 
 > 
 > First, **SHAP Local Impact Bars** show exact numerical contributions — here, 98% balance drain and 16.0x out/in degree ratio pushed the score up, while low clustering proved isolated synthetic counterparts. 
 > 
 > Second, our **LLM Tactical Briefing** synthesizes this into an actionable paragraph: 'Account flagged for promo abuse syndicate — verify device fingerprint collisions before manual release.' Here, the model scores; the LLM communicates. That’s deliberate AI judgment. 
 > 
 > Third, our **Counterfactual Engine** computes what-if adjustments: 'If this account had reduced balance drain to 0.25 and transacted through normal degree ratios, its score would drop from 98% to 18% (Safe Tier).' Deep, actionable explainability."*

---

### [02:45 – 03:45] 4. Calibrated FinOps Cost Model & Interactive Tradeoff Visual
* **Visual on Screen:** Switch to the **Precision / Recall / Cost** tab. Drag the threshold slider from 0.10 to 0.99, showing dynamic cost calculations. Click on the Confusion Matrix TP/FP/FN cells.
* **Spoken Script:**
 > *"Let’s talk honest metrics. Under extreme 0.13% fraud class imbalance, default F1 collapses. We don't hide that — we measure **Precision@100**, where our hybrid cascade achieves **92.0% Precision (+18.4% lift over tabular XGBoost)**. 
 > 
 > But more importantly, we built a **Calibrated FinOps Cost Model**: factoring in ₹350 per analyst review against ₹42,000 per undetected ring loss. 
 > 
 > As I move this threshold slider, the system dynamically calculates total financial impact. At our mathematically derived cost-optimal threshold **T* = 0.42**, we achieve minimum expected loss, saving **₹48,200+ per 10,000 transactions** compared to baseline rules. 
 > 
 > Furthermore, our Isotonic Calibration reduced Expected Calibration Error by **99.7%**, ensuring these cost numbers are trustworthy."*

---

### [03:45 – 04:30] 5. Failure Recovery Story ("What Broke, and How We Got Out")
* **Visual on Screen:** Show terminal with `pytest tests/` passing 43/43 tests, then highlight `tests/test_leakage.py` and `scripts/evaluate_tier4_research.py`.
* **Spoken Script:**
 > *"Now for what broke, and how we solved it: 
 > 
 > When analyzing standalone GAT recall under 0.13% extreme class imbalance, we saw majority clean noise diluting the minority fraud signal. Rather than smoothing over it, we implemented **PC-GNN neighbor sampling (Liu et al., WWW 2021)** to oversample minority fraud edges and preserve high-risk gradients. 
 > 
 > Furthermore, to eliminate temporal leakage, we enforced strict temporal graph partitioning, verified by our automated regression suite in `test_leakage.py`. 
 > 
 > All 43 unit, integration, leakage, and research model tests are automated and passing with 100% green status."*

---

### [04:30 – 05:00] 6. Adversarial Evasion Benchmark, What’s Next & Close
* **Visual on Screen:** Show the Adversarial Lab with **CARE-GNN (Dou et al., CIKM 2020)** camouflage filtering and Three.js 3D Temporal Ring Formation view.
* **Spoken Script:**
 > *"Finally, we red-teamed our detector against a distributed 'low-and-slow' evasion syndicate. Tabular XGBoost caught 0%. Our GAT caught 84%, and with **CARE-GNN adaptive similarity filtering**, detection reached **92%** by pruning camouflaged decoy edges. 
 > 
 > Sentinel is fully reproducible in one command (`python scripts/reproduce_benchmark.py`), verified across 6.36 million transactions, and guarded against drift by GitHub Actions CI. 
 > 
 > I'm ready to bring this level of engineering rigor and fintech product taste to your risk team starting September. Thank you!"*

---

## Recording Checklist Before Video Capture
- [ ] Run FastAPI backend: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- [ ] Open `http://localhost:8000/console.html` in full screen Chrome browser.
- [ ] Keep terminal open side-by-side to show `python scripts/reproduce_benchmark.py` and `pytest tests/`.
- [ ] Record in 1080p / 60fps with clear microphone audio.
