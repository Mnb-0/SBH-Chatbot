import os
import re
import streamlit as st

from dotenv import load_dotenv # <-- ADD THIS

# Load environment variables before anything else
load_dotenv() # <-- ADD THIS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core_logic import (
    BASELINE_FACTS,
    AUDIT_MIN_LENGTH,
    MAX_HISTORY_TURNS,
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

# FIX: Expanded to catch greetings and casual openers so they never hit the
# RAG pipeline or the 70B critic — they get an instant friendly reply instead.
_CASUAL_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings|salaam|marhaba)[\s!?.]*$"
    r"|^\s*(thanks|thank you|thx|ty)[\s!?.]*$"
    r"|\b(who|what)\s+(are|is)\s+(you|this|ur)\b"
    r"|\bwhat\s+(can|do)\s+you\b"
    r"|\byour\s+(name|purpose|role|job|function)\b"
    r"|\bintroduce\s+yourself\b"
    r"|\bwhat\s+are\s+you\b"
    r"|\btell me about yourself\b"
    r"|\bwhat can you do\b",
    re.IGNORECASE,
)

_GREETING_REPLIES = {
    "hi": "Hello! How can I help you with Salem Balhamer Holding Group today?",
    "hello": "Hello! How can I help you with Salem Balhamer Holding Group today?",
    "hey": "Hey there! Feel free to ask me anything about SBH.",
    "thanks": "You're welcome! Let me know if there's anything else I can help with.",
    "thank you": "You're welcome! Let me know if there's anything else I can help with.",
}


def get_casual_reply(text: str) -> str | None:
    """Returns a canned reply for greetings/identity questions, or None if not casual."""
    if not _CASUAL_PATTERNS.search(text):
        return None
    lower = text.strip().lower().rstrip("!?. ")
    return _GREETING_REPLIES.get(lower, IDENTITY_REPLY)


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

        # FIX: Casual messages (greetings, identity questions) are caught here and
        # returned instantly — no RAG retrieval, no routing, no 70B critic call.
        # This was the main cause of slow replies and wasted tokens on simple inputs.
        casual_reply = get_casual_reply(prompt)
        if casual_reply:
            st.markdown(casual_reply)
            st.session_state.messages.append({"role": "assistant", "content": casual_reply})

        else:
            with st.status("🧠 Analyzing Context...") as s:
                sector = get_sector_filter(prompt, gen_llm)
                s.update(label=f"Routed to: {sector}")

                docs = get_soft_search_results(vectorstore, prompt, sector)
                context = "\n\n".join([d.page_content for d in docs])
                s.update(label="Knowledge Retrieved", state="complete")

            system_prompt = f"""
            You are a knowledgeable, friendly assistant for Salem Balhamer Holding Group.
            You talk like a real person — warm, clear, and natural — not like a formal corporate brochure.
            Keep answers focused and easy to read. Use short paragraphs. Avoid bullet-point overload unless a list genuinely helps.

            STRICT RULES:
            1. Only use information from the retrieved context and baseline facts below. Never invent or guess.
            2. Match the user's language — if they write in Arabic, reply fully in Arabic.
            3. If you only have partial information, share what you know and honestly say you don't have the full picture.
            4. Don't quote specific numbers, financials, or headcounts unless they appear explicitly in the context.
            5. If someone asks you to compare SBH with competitors or criticise the company, politely steer the conversation back and suggest they reach out to the team directly.
            6. Never open with hollow phrases like "Certainly!", "Of course!", or "Great question!". Just answer.
            7. When your answer is incomplete or the person might need more help, end with something like:
               "Feel free to reach out to the team directly — you can email Info@SalemBalhamer.com or call +966 138127397."
            8. If the context and baseline facts have absolutely nothing relevant to the query, output ONLY the single word: UNANSWERABLE

            {BASELINE_FACTS}

            RETRIEVED CONTEXT: {context}
            """

            messages = [SystemMessage(content=system_prompt)]
            for msg in st.session_state.messages[:-1][-MAX_HISTORY_TURNS:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=prompt))

            initial_res = gen_llm.invoke(messages).content.strip()

            full_audit_context = f"{BASELINE_FACTS}\n{context}"

            if "UNANSWERABLE" in initial_res.upper():
                final_reply = (
                    "That's outside what I currently have on record. "
                    "For the most accurate information, please reach out to the SBH team directly at "
                    "+966 138127397 or visit us at Balhamer Business Gate, Dammam."
                )
                with st.expander("🛡️ Auditor's Report"):
                    st.info("Valid Refusal: Context was insufficient. Audit bypassed to save compute.")

            # FIX: Short answers (greetings that slipped through, one-liners) skip
            # the 70B critic entirely. Auditing a 10-word answer with a 70B model
            # is pure token waste — the risk of hallucination in short factual
            # answers is negligible.
            elif len(initial_res) < AUDIT_MIN_LENGTH:
                final_reply = initial_res
                with st.expander("🛡️ Auditor's Report"):
                    st.info("Short answer: Audit skipped to save compute.")

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