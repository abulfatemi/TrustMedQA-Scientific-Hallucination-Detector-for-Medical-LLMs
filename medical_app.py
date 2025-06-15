import streamlit as st

# Title
st.title("🩺 TrustMedQA: Scientific Verifier for LLM-Generated Medical Answers")
st.markdown("Check if an LLM-generated medical answer contains hallucinations, using scientific evidence and entailment analysis.")

# Question
question = st.text_input("Enter a medical question:", "")

from helper_functions import get_llm_answer_chain, extract_claims, extract_answer, get_evidence, get_entailment_roberta, is_hallucinated_with_explanation,extract_answer

if st.button("Analyze"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer and analyzing claims..."):
            # Get Answer and Claims
            chain=get_llm_answer_chain()

            response = chain(question)
            answer1=extract_answer(response)
            claims= extract_claims({'answer':response})
            st.subheader("💬 LLM Answer")
            st.markdown(answer1)

            # Analyze Claim
            st.subheader("🔍 Claim Analysis")
            path='/content/pubmedqa_biobert.index'

            labels=[]
            confidences=[]
            for idx, claim in enumerate(claims):
                st.markdown(f"**Claim {idx+1}:** {claim}")

                # Get Evidence
                evidence = get_evidence(claim,path)

                # Check Entailment
                label, confidence = get_entailment_roberta(claim, evidence)
                labels.append(label)
                confidences.append(confidence)

                # Show individual claim result
                st.markdown(f"- **Entailment Result:** `{label}`")
                st.markdown(f"- **Confidence:** `{confidence:.2f}`")

                with st.expander("📚 View Retrieved Evidence"):
                    st.markdown(evidence)
            # Final Result
            res=is_hallucinated_with_explanation(labels,confidences)
            st.markdown(f"**Hallucination Result:** `{res}`")
