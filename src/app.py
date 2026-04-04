import os
import re
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core_logic import (
    BASELINE_FACTS,
    get_rag_components,
    get_sector_filter,
    get_soft_search_results,
    verify_response,
    self_correct,
)

IDENTITY_REPLY = (
    "I am the Salem Balhamer Holding Group's corporate intelligence assistant. "
    "I can answer questions about SBH's business sectors — including Real Estate, Trading, "
    "Contracting, Industrial, and Services — as well as its leadership, history, and strategy. "
    "How can I help you today?"
)

_IDENTITY_PATTERNS = re.compile(
    r"\b(who|what)\s+(are|is)\s+(you|this|ur)\b"
    r"|\bwhat\s+(can|do)\s+you\b"
    r"|\byour\s+(name|purpose|role|job|function)\b"
    r"|\bintroduce\s+yourself\b"
    r"|\bwhat\s+are\s+you\b",
    re.IGNORECASE,
)


def is_identity_question(text: str) -> bool:
    return bool(_IDENTITY_PATTERNS.search(text))


st.set_page_config(page_title="Salem Balhamer Intel", page_icon="🏢", layout="wide")
st.title("Salem Balhamer Corporate Intelligence")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Groq API Key", type="password")
    st.info("Using Groq: Llama 3.1 8B (Generator) & Llama 3.3 70B (Auditor)")

api_key = api_key_input or os.environ.get("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your API Key in the sidebar or set the GROQ_API_KEY environment variable.")
    st.stop()


@st.cache_resource
def load_components(key):
    return get_rag_components(key)


try:
    vectorstore, gen_llm, critic_llm = load_components(api_key)
except Exception as e:
    st.error(f"Failed to load system components: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the group..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if is_identity_question(prompt):
            final_reply = IDENTITY_REPLY
            st.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
        else:
            with st.status("🧠 Analyzing Context...") as s:
                sector = get_sector_filter(prompt, critic_llm)
                s.update(label=f"Routed to: {sector}")

                docs = get_soft_search_results(vectorstore, prompt, sector)
                context = "\n\n".join([d.page_content for d in docs])
                s.update(label="Knowledge Retrieved", state="complete")

            # FIX: BASELINE_FACTS imported from core_logic — single source of truth.
            # The duplicate hardcoded block that was here has been removed.
            system_prompt = f"""
            ### ROLE: Corporate Analyst
            Rules: Use ONLY the provided context and baseline facts to answer the user's query. You are permitted to synthesize related themes if the text contains relevant corporate strategy.
            
            CRITICAL THRESHOLD: If the context and baseline facts are completely irrelevant and offer absolutely no foundational information to address the query, you must output EXACTLY the word "UNANSWERABLE" and nothing else. Do not apologize.
            
            {BASELINE_FACTS}
            
            RETRIEVED CONTEXT: {context}
            """

            # FIX: Conversation history is now built into the message list so the LLM
            # has full context of prior turns and can answer follow-up questions correctly.
            messages = [SystemMessage(content=system_prompt)]
            for msg in st.session_state.messages[:-1]:  # exclude the current prompt
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=prompt))

            initial_res = gen_llm.invoke(messages).content.strip()

            # FIX: full_audit_context defined once before the if/else block so it's
            # always available to both the expander and the self_correct call.
            full_audit_context = f"{BASELINE_FACTS}\n{context}"

            if "UNANSWERABLE" in initial_res.upper():
                final_reply = "I lack the corporate documentation to answer this question. The current database does not contain this information."
                with st.expander("🛡️ Auditor's Report"):
                    st.info("Valid Refusal: Context was insufficient. Audit bypassed to save compute.")
            else:
                with st.expander("🛡️ Auditor's Report"):
                    audit_result = verify_response(critic_llm, full_audit_context, initial_res)
                    st.json(audit_result)

                if audit_result.get("is_hallucinated", False):
                    st.warning("⚠️ Discrepancy detected. Re-evaluating context...")
                    final_reply = self_correct(
                        gen_llm,
                        full_audit_context,
                        prompt,
                        initial_res,
                        audit_result.get("reason", "Unknown hallucination"),
                    )
                else:
                    final_reply = initial_res

            st.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})