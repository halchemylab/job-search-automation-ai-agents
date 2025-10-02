#!/usr/bin/env python3

import os
import argparse
import numpy as np
from openai import OpenAI
import pypdf
import pickle

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    print("Text extracted successfully.")
    return text

def get_text_chunks(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks."""
    print("Splitting text into chunks...")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    print(f"Created {len(chunks)} chunks.")
    return chunks

def get_embeddings(texts, client, model="text-embedding-3-small"):
    """Gets embeddings for a list of texts."""
    print(f"Generating embeddings for {len(texts)} chunks...")
    response = client.embeddings.create(input=texts, model=model)
    print("Embeddings generated.")
    return [embedding.embedding for embedding in response.data]

def create_or_load_vector_store(resume_file, storage_file="vector_store.pkl", force_reindex=False):
    """Creates a vector store from a resume or loads it from a file."""
    if os.path.exists(storage_file) and not force_reindex:
        print(f"Loading vector store from {storage_file}...")
        with open(storage_file, 'rb') as f:
            return pickle.load(f)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 1. Extract text
    resume_text = extract_text_from_pdf(resume_file)
    
    # 2. Split into chunks
    text_chunks = get_text_chunks(resume_text)
    
    # 3. Get embeddings
    embeddings = get_embeddings(text_chunks, client)
    
    # 4. Create and save vector store
    vector_store = {
        "chunks": text_chunks,
        "vectors": np.array(embeddings)
    }
    
    print(f"Saving vector store to {storage_file}...")
    with open(storage_file, 'wb') as f:
        pickle.dump(vector_store, f)
    print("Vector store saved.")
        
    return vector_store

def find_similar_chunks(query_embedding, vector_store, top_k=5):
    """Finds the most similar chunks to a query embedding."""
    print("Finding similar chunks...")
    query_vector = np.array(query_embedding)
    similarities = np.dot(vector_store["vectors"], query_vector) / (np.linalg.norm(vector_store["vectors"], axis=1) * np.linalg.norm(query_vector))
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    print(f"Found {len(top_indices)} relevant chunks.")
    return [vector_store["chunks"][i] for i in top_indices]

def answer_query(query, resume_file, force_reindex=False):
    """Answers a query about a resume using a RAG approach."""
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Create or load vector store
    vector_store = create_or_load_vector_store(resume_file, force_reindex=force_reindex)
    
    # 2. Get query embedding
    query_embedding = get_embeddings([query], client)[0]
    
    # 3. Find similar chunks
    relevant_chunks = find_similar_chunks(query_embedding, vector_store)
    
    # 4. Build the prompt
    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""
    You are a helpful assistant. Answer the following query based on the provided resume context. 
    
    **Resume Context:**
    {context}
    
    **Query:**
    {query}
    
    **Answer:**
    """
    
    # 5. Get answer from OpenAI
    print("Getting answer from OpenAI...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that answers questions about a resume based on the context provided."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Answer received.")
    return response.choices[0].message.content


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Query a resume using OpenAI.")
    parser.add_argument("--resume-file", type=str, default="resume.pdf", help="Path to the resume file.")
    parser.add_argument("--query", type=str, required=True, help="Query to ask about the resume.")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing of the resume.")
    
    args = parser.parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    answer = answer_query(args.query, args.resume_file, args.force_reindex)
    
    print("\n---")
    print("Query:")
    print(args.query)
    print("\nAnswer:")
    print(answer)
    print("---")

if __name__ == "__main__":
    main()
