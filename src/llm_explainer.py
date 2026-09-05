"""
llm_explainer.py

Day 5 explanation-only LLM layer, user-facing name "RecoverIQ Decision
Assistant". explain_decision() turns an ALREADY-FINALIZED
DecisionEngine.build_recommendation_context() dict into a short,
human-readable explanation via the Gemini API (google-genai SDK).

This module has NO action authority:
  - it never calls ActionExecutor
  - it never constructs a PolicyCheckResult
  - it never reads or writes DecisionEngine/PolicyEngine state
  - it cannot influence final_action -- the decision it explains has
    already been made by the time this function is called

It only reads a context dict and returns a string.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Explicit path (repo_root/.env) rather than cwd-based discovery, so this
# loads correctly regardless of the working directory `streamlit run` (or
# any other caller) was launched from.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

MODEL = "gemini-flash-latest"  # gemini-2.0-flash is retired; this alias always
# points to Google's current recommended Flash model, avoiding a hardcoded
# version number that goes stale (confirmed via client.models.list() against
# this account: gemini-2.0-flash absent, gemini-2.5-flash already retired
# "no longer available to new users", gemini-flash-latest works)
MAX_OUTPUT_TOKENS = 500  # gemini-flash-latest has "thinking" on by default,
# which consumes this same budget before writing visible text -- 300 was
# enough to be entirely consumed by thinking tokens with nothing left for
# the answer (finish_reason=MAX_TOKENS, empty text). Combined with
# thinking_config below (which disables thinking outright), 500 leaves
# comfortable headroom for a genuine 3-4 sentence answer.

SYSTEM_PROMPT = (
    "You are RecoverIQ Decision Assistant.\n\n"
    "You explain and summarize decisions that have ALREADY been finalized "
    "by RecoverIQ's deterministic ML and policy system. You have NO "
    "authority to choose, modify, approve, reject, or execute actions.\n\n"
    "Use ONLY facts supplied in the JSON context. Never invent transaction "
    "details, scores, expected recovery values, policy rules, policy "
    "reasons, financial amounts, actions, outcomes, or performance "
    "metrics. You may explain relationships between facts already present "
    "in the context, but you must not create unsupported facts.\n\n"
    "For transaction questions, explain:\n"
    "- the model recommendation,\n"
    "- the final action,\n"
    "- whether policy overrode the recommendation,\n"
    "- relevant ML scores or expected recovery values,\n"
    "- relevant policy checks/reasons,\n"
    "- eligible alternatives when supplied.\n\n"
    "If the model recommendation and final action are the same, clearly "
    "say that policy did not override the recommendation. If they differ, "
    "clearly explain which policy constraint caused the override.\n\n"
    "If the JSON context represents a batch-level summary (for example it "
    "contains fields like transactions_processed, intervention_counts, or "
    "policy_override_rate) rather than a single transaction, summarize and "
    "interpret those supplied batch metrics instead of applying the "
    "single-transaction explanation structure above -- do not recalculate "
    "any of them. Clearly distinguish any field prefixed simulated_ (one "
    "seeded Bernoulli draw per transaction, a demo simulation, not an "
    "observed real payment) from fields such as oracle_revenue, "
    "system_revenue, pct_of_oracle_revenue, and policy_override_rate "
    "(a deterministic counterfactual evaluation against the holdout's "
    "hidden true-probability labels). Never call any of this production "
    "performance.\n\n"
    "If the JSON context includes a 'user_question' field, answer that "
    "specific question first, grounded only in the supplied context, then "
    "add a brief overall explanation if relevant. If the user's question "
    "asks you to change, approve, reject, execute, or override the "
    "decision, refuse and state that you can only explain the "
    "already-finalized decision.\n\n"
    "If the context does not contain enough information to answer a "
    "question, say plainly that the information is not available. Never "
    "claim that a payment was actually recovered unless an actual "
    "recovery result is explicitly supplied. Do not introduce information "
    "outside the supplied JSON.\n\n"
    "Keep responses concise (3-4 sentences for a single-transaction "
    "explanation; a short paragraph for a batch summary) and suitable for "
    "a fintech operations user. You are an explanation and insight layer "
    "only. You cannot change the decision."
)


def explain_decision(context: dict, api_key: str) -> str:
    """
    context: the dict returned by DecisionEngine.build_recommendation_context()
    api_key: the Gemini API key (read from GEMINI_API_KEY by the caller,
             e.g. via os.getenv() after this module's load_dotenv() call --
             never hardcoded here).

    Returns the explanation text, or a "[Explanation unavailable: ...]"
    string on any failure. Never raises.
    """
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=json.dumps(context),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (response.text or "").strip()
    except genai_errors.ClientError as e:
        return f"[Explanation unavailable: request rejected (status {e.status})]"
    except genai_errors.ServerError as e:
        return f"[Explanation unavailable: Gemini server error (status {e.status})]"
    except genai_errors.APIError as e:
        return f"[Explanation unavailable: API error (status {e.status})]"
    except Exception as e:
        return f"[Explanation unavailable: {type(e).__name__}]"
