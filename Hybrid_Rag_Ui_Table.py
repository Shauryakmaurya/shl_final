import json
import os
import pandas as pd
import faiss
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import requests

# Load data
with open("testshl_data_cleaned.json", "r") as f:
    shl_data = json.load(f)

# Prepare corpus and metadata
corpus = []
metadata = []
for entry in shl_data:
    content_parts = [
        entry.get("title", ""),
        entry.get("content", ""),
        entry.get("pdf_content", "")
    ]
    combined = "\n".join(filter(None, content_parts))
    corpus.append(combined)
    test_types = entry.get("test_types", [])
    test_type_str = ", ".join(test_types) if test_types else "Not specified"
    metadata.append({
        "title": entry.get("title", ""),
        "url": entry.get("url", ""),
        "test_type": test_type_str,
        "duration": entry.get("duration_minutes", "Not specified"),
        "adaptive_irt": entry.get("adaptive_irt", False),
        "remote_testing": entry.get("remote_testing", False)
    })

# Embed with MPNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

embeddings = embed(corpus)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# BM25 for lexical similarity
vectorizer = TfidfVectorizer().fit(corpus)
corpus_tfidf = vectorizer.transform(corpus)

def hybrid_search(query, top_k=20, alpha=0.5):
    query_embedding = embed([query])
    _, faiss_ids = index.search(query_embedding, top_k)
    faiss_scores = [1 - np.linalg.norm(query_embedding - embeddings[i]) for i in faiss_ids[0]]

    query_tfidf = vectorizer.transform([query])
    bm25_scores = corpus_tfidf.dot(query_tfidf.T).toarray().flatten()
    bm25_top_ids = np.argsort(bm25_scores)[-top_k:][::-1]

    all_ids = list(set(faiss_ids[0]) | set(bm25_top_ids))
    results = []
    for i in all_ids:
        bm25_score = bm25_scores[i]
        faiss_score = 1 - np.linalg.norm(query_embedding - embeddings[i])
        combined_score = alpha * faiss_score + (1 - alpha) * bm25_score
        results.append((combined_score, i))
    results.sort(reverse=True)
    filtered_results = results[:10] if len(results) > 10 else results[:max(1, len(results))]
    return [metadata[i] for _, i in filtered_results], [corpus[i] for _, i in filtered_results]

# Build prompt and call LLaMA via Groq
GROQ_API_KEY = "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"

def call_llama(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an SHL test recommender. Suggest suitable assessments based on user input."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def format_table(results):
    df = pd.DataFrame(results)
    df["Remote"] = df["remote_testing"].apply(lambda x: "✅" if x else "❌")
    df["Adaptive/IRT"] = df["adaptive_irt"].apply(lambda x: "✅" if x else "❌")
    df["URL"] = df["url"].apply(lambda x: f"[Link]({x})")
    return df[["title", "test_type", "duration", "Remote", "Adaptive/IRT", "URL"]].rename(columns={
        "title": "Title",
        "test_type": "Test Type",
        "duration": "Duration (min)"
    })

def query_rag_system(query):
    top_meta, top_docs = hybrid_search(query)
    context = "\n\n".join(top_docs)
    prompt = f"Here is the context of available SHL tests:\n\n{context}\n\nBased on this, suggest the most relevant assessments for the following job description or query:\n{query}"
    llama_response = call_llama(prompt)
    print("\nTop Matches:")
    print(format_table(top_meta).to_markdown(index=False))
    print("\nLLaMA Suggestion:")
    print(llama_response)

# Example usage
# query_rag_system("Looking for a test to assess civil engineering graduates with aptitude in transportation and water resources")
