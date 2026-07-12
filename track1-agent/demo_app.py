"""Screen-recording demo UI: type a query, watch the router decide, in real time.

Not part of the hackathon submission (the container only needs agent.py). This
is purely to show what's happening under the hood: the local fine-tuned router
making a free decision, the prompt-based baseline paying for the same decision,
and the real Fireworks answer call, side by side.

Run with: streamlit run demo_app.py
"""
import os
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.fireworks_client import chat
from router.infer_router import predict as finetuned_predict
from src.baseline_router import classify as baseline_classify

MODEL_CHEAP = os.environ["MODEL_CHEAP"]
MODEL_EXPENSIVE = os.environ["MODEL_EXPENSIVE"]


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


st.set_page_config(page_title="Query Router Demo", page_icon="🔀", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🔀 Token-Efficient Query Router")
st.caption(
    "Fine-tuned local router vs. a prompt-based baseline — built for AMD Developer "
    "Hackathon Act II's Track 1 (Hybrid Token-Efficient Routing Agent)."
)



prompt = st.text_area("Enter a query", height=100, placeholder="e.g. What is 12 + 7?")
run = st.button("Run through router", type="primary", disabled=not prompt.strip())

if run:
    with st.status("Routing query...", expanded=True) as status:
        st.write("**Step 1 — Fine-tuned router** (local forward pass, zero tokens)")
        t0 = time.time()
        finetuned_label = finetuned_predict(prompt)
        finetuned_latency_ms = (time.time() - t0) * 1000
        finetuned_model = MODEL_EXPENSIVE if finetuned_label == "hard" else MODEL_CHEAP
        st.write(
            f"Decision: `{finetuned_label}` → routes to `{short_name(finetuned_model)}` "
            f"({finetuned_latency_ms:.0f}ms, 0 tokens)"
        )

        st.write("**Step 2 — Prompt-based baseline** (real Fireworks call, for comparison only)")
        baseline_result = baseline_classify(prompt)
        baseline_model = MODEL_EXPENSIVE if baseline_result["label"] == "hard" else MODEL_CHEAP
        st.write(
            f"Decision: `{baseline_result['label']}` → would route to `{short_name(baseline_model)}` "
            f"({baseline_result['tokens']} tokens spent just deciding)"
        )

        st.write(f"**Step 3 — Answering via `{short_name(finetuned_model)}`**")
        answer = chat(finetuned_model, prompt, max_tokens=700)
        answer_tokens = answer["total_tokens"]

        if baseline_model == finetuned_model:
            baseline_query_tokens = baseline_result["tokens"] + answer_tokens
        else:
            st.write(f"Baseline disagreed, calling `{short_name(baseline_model)}` too for a fair comparison")
            baseline_answer = chat(baseline_model, prompt, max_tokens=700)
            baseline_query_tokens = baseline_result["tokens"] + baseline_answer["total_tokens"]

        status.update(label="Done", state="complete")

    st.subheader("Answer")
    st.write(answer["text"])

    st.session_state.history.append({
        "query": prompt[:60] + ("..." if len(prompt) > 60 else ""),
        "finetuned_tokens": answer_tokens,
        "baseline_tokens": baseline_query_tokens,
    })


if st.session_state.history:
    st.subheader("Tokens per query, this session")
    df = pd.DataFrame(st.session_state.history)
    chart_df = df.set_index("query")[["finetuned_tokens", "baseline_tokens"]]
    chart_df.columns = ["Fine-tuned router", "Prompt-based baseline"]
    st.bar_chart(chart_df)

    st.subheader("Query log")
    st.dataframe(df, use_container_width=True, hide_index=True)

with st.sidebar:
    st.subheader("Model tiers")
    st.markdown(f"**Cheap:** `{short_name(MODEL_CHEAP)}`")
    st.markdown(f"**Escalation:** `{short_name(MODEL_EXPENSIVE)}`")
    st.divider()
    st.subheader("Session totals")
    n = len(st.session_state.history)
    finetuned_total = sum(h["finetuned_tokens"] for h in st.session_state.history)
    baseline_total = sum(h["baseline_tokens"] for h in st.session_state.history)
    saved = baseline_total - finetuned_total
    saved_pct = (saved / baseline_total * 100) if baseline_total else 0
    st.metric("Queries run", n)
    st.metric("Fine-tuned router — total tokens", finetuned_total)
    st.metric("Prompt-based baseline — total tokens", baseline_total)
    st.metric("Tokens saved by fine-tuning", saved, delta=f"{saved_pct:.0f}% fewer tokens" if n else None)
    if st.button("Reset session"):
        st.session_state.history = []
        st.rerun()
