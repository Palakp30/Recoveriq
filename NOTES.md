# Day 4 notes

Built `src/batch_runner.py` (`run_batch(transactions, engine)` — loops
`DecisionEngine.decide()`, no new logic) and `app.py` (Streamlit dashboard,
repo root). The dashboard calls the existing Day 3 `DecisionEngine`/
`ActionExecutor` over all 300 holdout transactions; it adds one new thing —
a seeded `np.random.default_rng(42)` Bernoulli draw per transaction against
`true_prob_<final_action>` — to turn probabilities into a labeled
**simulated** recovered/not-recovered outcome. No new eligibility rules,
no edits to `data/*.csv`, `simulator/*.py`, or `src/features.py`.

Metrics shown: Revenue at Risk, Revenue Recovered (simulated), Recovery
Rate, Transactions Processed (300), plus two callouts read directly from
`outputs/full_policy_comparison.json` — **85.0% of oracle revenue** and
**65.7% policy override rate**. Interventions and outcome breakdowns are
bar charts over `final_action` and a derived `outcome_category`
(Recovered / Not Recovered / Held for Manual Review / Rejected at
Execution). Transaction trace view exposes `ml_scores`,
`expected_recovery_values`, `policy_checks`, `model_recommendation` vs
`final_action`/`policy_override`/`override_reason`, and the simulated
outcome — all read directly off `DecisionResult`/`ExecutionResult`.

`streamlit run app.py` verified launching without error (HTTP 200, no
traceback, checked via both a direct bare-mode run of `app.py` and a real
headless `streamlit run`). Added `streamlit==1.63.0` to `requirements.txt`.
