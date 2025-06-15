

from datasets import load_dataset

ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
texts=[x['context'] for x in ds['train']]


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb")

embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

import faiss
dimension = embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "pubmedqa_biobert.index")
