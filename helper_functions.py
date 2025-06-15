# -*- coding: utf-8 -*-







# Load LLM Model
from langchain_google_genai import GoogleGenerativeAI
from key import key
Google_api_key=key
llm=GoogleGenerativeAI(google_api_key=Google_api_key,model='gemini-1.5-flash',temperature=0)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import faiss
import torch
import re
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")
import pickle
with open("/content/pubmedqa_texts.pkl", "rb") as f:
    texts = pickle.load(f)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer_entail = AutoTokenizer.from_pretrained("CNT-UPenn/Bio_ClinicalBERT_for_seizureFreedom_classification")
model_entail = AutoModelForSequenceClassification.from_pretrained("CNT-UPenn/Bio_ClinicalBERT_for_seizureFreedom_classification")


def get_llm_answer_chain():
    prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful medical assistant.
Answer the following question and also 2-4 factual claims it uses to support answer using clear scientific language.

Question: {question}
Answer:"""
)

    chain=LLMChain(llm=llm,prompt=prompt)
    return(chain)

def get_evidence(question,path):
  index=faiss.read_index(path)
  query_vec = model.encode([question])
  D, I = index.search(query_vec, 5)
  top_passages = [texts[i] for i in I[0]]
  evidence = "\n\n".join([" ".join(p['contexts']) for p in top_passages])
  return(evidence)

def extract_answer(d):
    prompt = PromptTemplate(
              input_variables=["response"],
              template="""
You are a medical answer extraction assistant.
In the response it is clearly stated which are supporting claims and answer. 
Just give answer in 2-3 sentences

Response:
"{response}"

""")
    chain=LLMChain(llm=llm,prompt=prompt)
    rep=chain({'response':d})
    return(rep['text'].strip()) 


def extract_claims(answer_text):
    # Simple sentence splitter (can be improved)
    prompt = PromptTemplate(
              input_variables=["answer"],
              template="""
You are a medical claim extraction assistant.
In the answer it is clearly stated which are supporting claims just give those claims in one sentence

Answer:
"{answer}"

Return only the list of one-sentence claims, numbered.
"""
)
    claim_chain = LLMChain(llm=llm, prompt=prompt)
    claims_text = claim_chain.run(answer_text)

    # Extract numbered claims like: 1. Claim..., 2. Claim...
    claims = re.findall(r'\d+\.\s+(.*)', claims_text)
    return [c.strip() for c in claims if c.strip()]

import torch.nn.functional as F
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Move the model to the specified device
model_entail.to(device)


def get_entailment(premise, hypothesis):
    inputs = tokenizer_entail.encode_plus(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model_entail(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Labels: 0 = contradiction, 1 = neutral, 2 = entailment
    labels = ["Contradiction", "Neutral", "Entailment"]
    prediction = labels[probs.argmax()]
    confidence = probs.max()

    return prediction, confidence


def is_hallucinated_with_explanation(labels, confidences):
    entailment_threshold = 0.9
    contradiction_threshold = 0.9
    contradiction_ratio_cutoff = 0.5
    min_entailment_ratio = 0.3

    total = len(labels)
    entailment_support = 0
    contradiction_count = 0
    neutral_count = 0
    contradiction_confidence_sum = 0

    for label, conf in zip(labels, confidences):
        if label == "Entailment" and conf >= entailment_threshold:
            entailment_support += 1
        elif label == "Contradiction" and conf >= contradiction_threshold:
            contradiction_count += 1
            contradiction_confidence_sum += conf
        elif label == "Neutral":
            neutral_count += 1

    entailment_ratio = entailment_support / total if total > 0 else 0
    contradiction_ratio = contradiction_count / total if total > 0 else 0
    neutral_ratio = neutral_count / total if total > 0 else 0
    
    if entailment_support == 0 and contradiction_count == 0 and neutral_count == total:
        return "Unverifiable", (
            f"All {total} evidence items are neutral. The claims could not be verified or contradicted based on available scientific evidence."
        )

    # Case 1: Contradiction dominates
    if contradiction_ratio >= contradiction_ratio_cutoff and (contradiction_confidence_sum / max(1, contradiction_count)) >= contradiction_threshold:
        return "Hallucinated", (
            f"{contradiction_count} out of {total} evidence items contradict the claims with high confidence "
            f"(avg contradiction confidence: {contradiction_confidence_sum / contradiction_count:.2f})."
        )

    # Case 2: Strong entailment support
    if entailment_support > 0 and entailment_ratio >= min_entailment_ratio:
        return "Supported", (
            f"{entailment_support} out of {total} evidence items entail the claims with high confidence "
            f"(entailment ratio: {entailment_ratio*100:.1f}%)."
        )

    # Case 3: Weak entailment and/or mostly neutral
    if entailment_support > 0 and entailment_ratio < min_entailment_ratio:
        return "Hallucinated", (
            f"Only {entailment_support} out of {total} evidence items support the claims "
            f"(entailment ratio: {entailment_ratio*100:.1f}%), which is below the threshold ({min_entailment_ratio*100:.0f}%)."
        )

    # Default fallback
    return "Unverifiable", (
        f"The evidence is mostly neutral or inconclusive. There is insufficient support to verify or falsify the claims."
    )


 
