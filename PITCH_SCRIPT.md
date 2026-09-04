# PITCH_SCRIPT.md — AI Risk Manager (Track 02: Abuse-Ring Sentinel)

> **Project Name:** Abuse Ring Sentinel  
> **Applicant:** Mudit  
> **Target Role:** AI Builder Intern (In-person Bangalore, from September)  
> **Target Video Length:** Exactly 5 Minutes (00:00 – 05:00)  
> **Tone:** Natural, conversational, confident, evidence-backed. Spoken directly as a solo engineer.

---

## 5-Minute Spoken Teleprompter Script

---

### [00:00 – 00:45] The Hook & The Problem
* **On Screen**: Web Console home screen (https://abuse-ring-sentinel-rp.vercel.app/console.html).
* **Say Naturally**:
> "Hey everyone, I'm Mudit, and this is Abuse Ring Sentinel.
>
> When people think of payment fraud, they usually imagine someone using a stolen credit card. But on modern payment gateways, two-factor authentication and OTPs already stop most of that.
>
> The real problem that quietly bleeds merchant margins is organized abuse rings. We are talking about syndicates using device farms to drain first-order promo coupons, running return loops on expensive electronics with empty boxes, or doing friendly-fraud chargebacks.
>
> If an account spends 500 rupees with normal velocity, a standard rule engine or isolated machine learning model lets it right through. They look at each account by itself, so they are completely blind to the hidden web connecting them.
>
> To catch these rings, you have to look at the graph connections between devices, bank accounts, and addresses. So I built Sentinel to do exactly that, running in under one millisecond."

---

### [00:45 – 01:45] Live Stream & The Guardrail
* **On Screen**: Click **Live Payment Stream** tab. Point to incoming transactions, then click **"Plant Promo Ring (6 Tx)"**.
* **Say Naturally**:
> "Let's jump into the live console.
>
> Here is our live transaction stream, styled like real gateway traffic. You can see incoming payments with UPI IDs and cards.
>
> Now, watch what happens when I plant a coordinated promo ring of six accounts.
>
> Immediately, the graph attention layer spots the multi-account pattern and flags the cluster as Critical Risk. And it does this in 0.78 milliseconds, because the graph embeddings are precomputed in Redis instead of querying a massive database on the live checkout path.
>
> Now, here is a critical design choice: Sentinel never auto-blocks or freezes an account automatically. Auto-blocking legitimate merchants causes huge support headaches. Instead, it generates an advisory risk tier and routes it to a human investigator with a durable audit log."

---

### [01:45 – 02:45] The Explainability Triad
* **On Screen**: Click **Risk Evaluator** tab. Pick the preset **"Promo-Abuse Ring"** and click **"Score Transaction"**.
* **Say Naturally**:
> "A risk score by itself doesn't help an operations team. If an analyst gets an alert, they need to know why.
>
> So I built what I call an Explainability Triad.
>
> First, these SHAP bars show the exact math behind the score. For this account, a 98 percent balance drain and a 16x degree ratio pushed the risk up, while low clustering proved it was an isolated dummy account.
>
> Second, the tactical briefing translates those technical graph metrics into plain English for the analyst, telling them exactly what to verify before releasing funds.
>
> And third, the counterfactual card shows what the account would need to look like to be considered safe. So it's completely transparent, not a black box."

---

### [02:45 – 03:45] Real FinOps Cost Math & Tradeoffs
* **On Screen**: Click **Precision / Recall / Cost** tab. Drag the threshold slider back and forth to show the cost recalculating live.
* **Say Naturally**:
> "Now let's talk about the metrics, and I want to be completely honest here.
>
> In real payment data, fraud is rare—about 0.13 percent. If you use a default threshold of 0.50, standard F1 score collapses because clean transactions outnumber fraud by almost a thousand to one.
>
> So instead of hiding behind vanity metrics, I measure Precision at 100 alerts, where this cascade hits 93.5 percent precision.
>
> Even better, I built a FinOps Cost Model. An unnecessary manual review costs about 350 rupees of analyst time. But missing a fraud ring costs about 42,000 rupees.
>
> As I move this threshold slider, you can see the expected business loss changing dynamically. The sweet spot is a threshold of 0.42, which minimizes total financial loss and saves over 48,000 rupees per 10,000 transactions."

---

### [03:45 – 04:45] What Broke & How I Solved It
* **On Screen**: Switch to your terminal window and run `python -m pytest tests/` to show all 59 tests passing green.
* **Say Naturally**:
> "Now for what broke while I was building this, and what I did to fix it.
>
> First, standard Graph Attention Networks completely failed when I tested them on their own. Because 99 percent of neighbors are clean, standard message passing diluted the fraud signal. So I implemented PC-GNN sampling to specifically oversample minority fraud edges and paired it with XGBoost in a two-stage cascade.
>
> Second, in adversarial evasion tests, fraudsters injected 500 fake connections to clean accounts. Standard graph attention recall dropped from 84 percent down to 19 percent. To stop this camouflage, I implemented a Chebyshev spectral filter and similarity pruning, keeping ring recall at 84 percent.
>
> Third, live graph traversals took 85 milliseconds, which blew the 15-millisecond gateway speed budget. I moved graph calculation nearline into Redis, cutting live response time to 0.78 milliseconds.
>
> As you can see on the terminal right now, all 59 unit, integration, and leakage tests are automated and passing green."

---

### [04:45 – 05:00] Close
* **On Screen**: Switch back to the Web Console or look into your camera.
* **Say Naturally**:
> "To wrap up, Abuse Ring Sentinel is live, tested, and built for real production constraints. It catches the collusive networks that tabular models miss, stays well under checkout latency limits, and protects merchant margins.
>
> I'm Mudit, and I'm really excited about the opportunity to bring this level of engineering rigor to Razorpay. Thanks for watching!"
- [ ] Record in 1080p / 60fps with clear microphone audio.
