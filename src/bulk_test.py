import os
import pandas as pd
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from core_logic import get_rag_components, verify_response, get_sector_filter

TEST_QUESTIONS = [
    "Who is the Chairman of Salem Balhamer Holding?",
    "When was the group officially established?",
    "What are the five main business sectors of the group?",
    "List the residential complexes managed by the Real Estate sector.",
    "What plastic products does the Trading sector provide?",
    "Does the contracting sector have projects in NEOM?",
    "What is the CEO's favorite car brand?", 
    "Does the group own a private space exploration company?", 
    "How does the group support Saudi Vision 2030?",
    "What is the group's mission regarding industrial expansion?"
]

def run_bulk_test(api_key):
    print("🚀 Initializing Audited RAG Test...")
    vectorstore, gen_llm, critic_llm = get_rag_components(api_key)
    results = []
    
    for query in TEST_QUESTIONS:
        target_sector = get_sector_filter(query, critic_llm)
        search_kwargs = {"k": 6}
        
        if target_sector != "General":
            search_kwargs["filter"] = Filter(
                should=[FieldCondition(key="metadata.sector", match=MatchValue(value=target_sector))]
            )

        docs_and_scores = vectorstore.similarity_search_with_score(query, **search_kwargs)
        context = "\n\n".join([d.page_content for d, _ in docs_and_scores]) if docs_and_scores else "No context found."

        system_prompt = f"""
        ### ROLE: Corporate Analyst
        Rules: Use ONLY the provided context to answer the user's query. You are permitted to synthesize related themes (e.g., summarizing strategic goals to answer a "mission" question) if the text contains relevant corporate strategy.
        
        CRITICAL THE THRESHOLD: If the context is completely irrelevant (e.g., asks about cars, space, or unrelated entities) and offers absolutely no foundational information to address the query, you must output EXACTLY the word "UNANSWERABLE" and nothing else. Do not apologize.
        
        CONTEXT: {context}
        """
        
        try:
            res = gen_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])
            answer = res.content.strip()
            
            # --- THE DETERMINISTIC SHORT CIRCUIT ---
            if "UNANSWERABLE" in answer.upper():
                is_hallucinated = False
                reason = "Valid Refusal: Context was insufficient."
                answer = "I lack the documentation to answer this." # Clean up the output for the CSV
            else:
                # Only run the expensive audit on actual claims
                audit_result = verify_response(critic_llm, context, answer)
                is_hallucinated = audit_result.get("is_hallucinated", False)
                reason = audit_result.get("reason", "N/A")
                
        except Exception as e:
            answer = f"Error: {e}"
            is_hallucinated = True
            reason = str(e)

        results.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Question": query,
            "Routed_Sector": target_sector,
            "Answer": answer,
            "Hallucinated": is_hallucinated,
            "Audit_Reason": reason,
            "Context_Snippet": context[:200] + "..."
        })
        print(f"Done: {query[:30]:<30} | Hallucinated: {is_hallucinated}")

    df = pd.DataFrame(results)
    df.to_csv("audited_corporate_test.csv", index=False)
    print("\nTest complete. Review 'audited_corporate_test.csv' for results.")

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set your GROQ_API_KEY environment variable before running the test.")
        exit(1)
    run_bulk_test(api_key)