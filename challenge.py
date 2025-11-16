"""
Optimized Regulation Processing Pipeline (Memory-Safe + Performance)
-------------------------------------------------------------------
Enhanced for cross-lingual semantic similarity between Finnish and English regulations
Modified for structured Requirement JSON format
"""

import os
import json
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
import networkx as nx
from pathlib import Path
import faiss
import time
from datetime import datetime
import gc
import pickle  


# CONFIGURATION & RATE LIMITING


API_KEY = "API_KEY"

try:
    genai.configure(api_key=API_KEY)
    print("✅ API key configured")
except Exception as e:
    print(f"❌ API configuration failed: {e}")

# Configuration
MAX_CHARS = 30000
CHUNK_SIZE = 3000
OVERLAP = 250
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BATCH_SIZE = 10

def rate_limited_call(func, *args, **kwargs):
    """Add rate limiting and retry logic to API calls."""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            time.sleep(0.5)
            return result
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"All retries failed for {func.__name__}: {e}")
                raise e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            time.sleep(RETRY_DELAY * (attempt + 1))


# 1. ENHANCED EMBEDDING FUNCTIONS FOR CROSS-LINGUAL SEMANTICS


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks for embedding."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap
        if start >= end:
            break
    return chunks

def gemini_embedding(text: str):
    """
    Enhanced Gemini embedding for cross-lingual semantic similarity.
    Gemini's text-embedding-004 is multilingual and captures semantic meaning
    across languages like Finnish and English.
    """
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return [0.0] * 768

    # Clean text to reduce token usage - preserve multilingual content
    text = ' '.join(text.split())[:MAX_CHARS]

    try:
        # Use text-embedding-004 which is designed for multilingual semantic similarity
        if len(text) < MAX_CHARS:
            return rate_limited_call(
                genai.embed_content, 
                model="models/text-embedding-004", 
                content=text,
                task_type="retrieval_document"  # Enhanced for document retrieval across languages
            )["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return [0.0] * 768

    # For longer texts, use chunking with semantic preservation
    chunks = chunk_text(text)
    if not chunks:
        return [0.0] * 768

    avg_vec = None
    n = 0
    for chunk in chunks[:5]:
        try:
            emb = np.array(rate_limited_call(
                genai.embed_content,
                model="models/text-embedding-004", 
                content=chunk,
                task_type="retrieval_document"
            )["embedding"])
            if avg_vec is None:
                avg_vec = emb
            else:
                avg_vec = (avg_vec * n + emb) / (n + 1)
            n += 1
        except Exception as e:
            print(f"Chunk embedding failed: {e}")
            continue

    return avg_vec.tolist() if avg_vec is not None else [0.0] * 768


# 2. MODIFIED DOCUMENT PROCESSING FOR STRUCTURED REQUIREMENT JSON


def detect_language(text):
    """
    Simple language detection to identify Finnish vs English content.
    This helps in understanding the composition of our dataset.
    """
    if not text:
        return "unknown"
    
    # Simple heuristic based on common Finnish characters and words
    finnish_indicators = ['ä', 'ö', 'å', 'suomi', 'finland', 'tässä', 'mukaan', 'voi']
    text_lower = text.lower()
    
    finnish_score = sum(1 for indicator in finnish_indicators if indicator in text_lower)
    
    if finnish_score > 2:
        return "finnish"
    else:
        return "english"

def extract_requirements_from_structured_json(raw):
    """
    Directly extract requirements from structured JSON without LLM extraction.
    JSON format: {"Requirement": [{"name": "...", "id": "...", "content": "..."}]}
    """
    requirements = []
    
    if "Requirement" in raw and isinstance(raw["Requirement"], list):
        for req in raw["Requirement"]:
            if isinstance(req, dict) and "name" in req and "content" in req:
                # Create a meaningful requirement string that includes name and content
                requirement_str = f"{req['name']}: {req['content']}"
                requirements.append(requirement_str)
    
    return requirements

def extract_text_for_embedding(raw):
    """
    Extract text for embedding from requirement content fields.
    This is what gets embedded to find semantic similarities between documents.
    """
    content_parts = []
    
    if "Requirement" in raw and isinstance(raw["Requirement"], list):
        for req in raw["Requirement"]:
            if isinstance(req, dict) and "content" in req and req["content"]:
                content_parts.append(req["content"])
    
    # Combine all requirement contents - this becomes the document text for embedding
    combined_text = "\n\n".join(content_parts)
    return combined_text[:20000]  # Limit length

def stream_documents_structured(folder_path, max_files=None):
    """Modified document streaming for structured requirement JSON format."""
    all_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".json")]
    
    if max_files:
        all_files = all_files[:max_files]
    
    print(f"Found {len(all_files)} structured JSON files to process in {folder_path}")
    
    for filename in all_files:
        fp = os.path.join(folder_path, filename)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # Extract text for embedding (from content fields)
            text_for_embedding = extract_text_for_embedding(raw)
            
            # Directly extract requirements from structure (no LLM needed)
            requirements = extract_requirements_from_structured_json(raw)
            
            doc_id = filename.replace(".json", "")
            language = detect_language(text_for_embedding)
            
            yield {
                "id": doc_id,
                "filename": filename,
                "text": text_for_embedding,  # This gets embedded for document-level similarity
                "language": language,
                "embedding": None,
                "cluster": None,
                "risk_category": None,
                "requirements": requirements,  # Pre-extracted from structure
                "raw_json": raw
            }
        except Exception as e:
            print(f"Failed to load {fp}: {e}")


# 3. ENHANCED CLUSTERING FOR CROSS-LINGUAL SEMANTIC SIMILARITY


def cluster_documents_crosslingual(documents, num_clusters=10):
    """
    Enhanced clustering that leverages Gemini's multilingual embeddings
    to find semantic similarities between Finnish and English regulations.
    """
    if len(documents) == 0:
        return documents
    
    # Handle cases with too few documents for clustering
    if len(documents) <= num_clusters:
        print(f"Only {len(documents)} documents - assigning each to its own cluster")
        for i, doc in enumerate(documents):
            doc["cluster"] = i
        return documents

    try:
        # Use Gemini's multilingual embeddings which capture semantic meaning across languages
        embeddings = np.vstack([d["embedding"] for d in documents])
        
        # Report language distribution in clusters for debugging
        languages = [d.get("language", "unknown") for d in documents]
        print(f"Language distribution: {len([l for l in languages if l == 'finnish'])} Finnish, "
              f"{len([l for l in languages if l == 'english'])} English")
        
        # Use cosine similarity for semantic clustering
        # Gemini embeddings are designed to work well with cosine similarity across languages
        if len(documents) > 1000:
            print("Using MiniBatchKMeans for large dataset...")
            clustering = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
        else:
            print("Using AgglomerativeClustering with cosine metric...")
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters, 
                metric='cosine',  # Cosine distance works well with semantic embeddings
                linkage="average"
            )
        
        labels = clustering.fit_predict(embeddings)
        for doc, label in zip(documents, labels):
            doc["cluster"] = int(label)
        
        # Analyze cluster language mixing (for debugging)
        clusters_language_mix = {}
        for doc in documents:
            cluster = doc["cluster"]
            lang = doc.get("language", "unknown")
            if cluster not in clusters_language_mix:
                clusters_language_mix[cluster] = {"finnish": 0, "english": 0, "unknown": 0}
            clusters_language_mix[cluster][lang] += 1
        
        print("Cluster language mix (Finnish/English):")
        for cluster, counts in clusters_language_mix.items():
            print(f"  Cluster {cluster}: {counts['finnish']} Finnish, {counts['english']} English")
        
        # Free memory
        del embeddings
        gc.collect()
        
    except Exception as e:
        print(f"Clustering failed: {e}. Assigning default clusters...")
        for i, doc in enumerate(documents):
            doc["cluster"] = i % num_clusters
    
    return documents


# 4. MODIFIED REQUIREMENT PROCESSING FOR STRUCTURED JSON


def gemini_generate(prompt, model="gemini-2.0-flash"):
    """General wrapper for Gemini LLM responses with error handling."""
    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini generation failed: {e}")
        return f"Error: {e}"

def extract_all_requirements_structured(documents, batch_size=10):
    """
    For structured JSON, requirements are already extracted.
    This function just verifies and formats them.
    """
    if not documents:
        return documents
        
    print(f"Using pre-extracted requirements from {len(documents)} structured JSON files...")
    
    for doc in documents:
        # Requirements are already populated from the JSON structure
        # Just ensure they're in the right format
        if not doc.get("requirements"):
            doc["requirements"] = []
        else:
            # Optional: Clean up requirement strings
            doc["requirements"] = [req.strip() for req in doc["requirements"] if req and req.strip()]
    
    return documents


# 5. ENHANCED OVERLAP DETECTION FOR CROSS-LINGUAL SEMANTICS


def find_overlapping_requirements_crosslingual(documents, embedding_dim=768, threshold=0.90, k=5):
    """
    Enhanced overlap detection using Gemini's multilingual embeddings
    to find semantic similarities between requirements in different languages.
    """
    if len(documents) <= 1:
        return []
        
    # Build requirements list with language info
    reqs = []
    req_ids = []
    req_languages = []
    
    for d in documents:
        for r in d["requirements"]:
            if len(r) > 10:  # Filter very short requirements
                reqs.append(r)
                req_ids.append(d["id"])
                req_languages.append(d.get("language", "unknown"))
    
    if len(reqs) <= 1:
        print("Not enough requirements found for overlap detection")
        return []
    
    print(f"Processing {len(reqs)} requirements for cross-lingual overlap detection...")
    print(f"Requirements by language: {len([l for l in req_languages if l == 'finnish'])} Finnish, "
          f"{len([l for l in req_languages if l == 'english'])} English")
    
    # Batch process embeddings
    req_embeddings = []
    
    for i, req in enumerate(tqdm(reqs, desc="Embedding requirements")):
        try:
            emb = gemini_embedding(req)
            req_embeddings.append(np.array(emb, dtype=np.float32))
        except Exception as e:
            print(f"Failed to embed requirement {i}: {e}")
            continue
        
        if i % 50 == 0:
            gc.collect()
    
    if len(req_embeddings) <= 1:
        return []
    
    req_embeddings = np.vstack(req_embeddings)
    
    try:
        # Use FAISS for efficient similarity search
        faiss.normalize_L2(req_embeddings)
        index_flat = faiss.IndexFlatIP(embedding_dim)
        index_flat.add(req_embeddings)
        
        overlaps = []
        k_search = min(k+1, len(req_embeddings))
        distances, neighbors = index_flat.search(req_embeddings, k_search)
        
        for i in range(len(distances)):
            for dist, neighbor_idx in zip(distances[i][1:], neighbors[i][1:]):
                # Check if this is a cross-lingual match
                is_crosslingual = (req_languages[i] != req_languages[neighbor_idx])
                
                if (dist >= threshold and 
                    i != neighbor_idx and 
                    req_ids[i] != req_ids[neighbor_idx]):
                    
                    overlaps.append({
                        "doc1": req_ids[i],
                        "req1": reqs[i],
                        "doc2": req_ids[neighbor_idx],
                        "req2": reqs[neighbor_idx],
                        "similarity": float(dist),
                        "cross_lingual": is_crosslingual,
                        "languages": f"{req_languages[i]}-{req_languages[neighbor_idx]}"
                    })
        
        cross_lingual_count = len([o for o in overlaps if o["cross_lingual"]])
        print(f"Found {len(overlaps)} overlapping pairs ({cross_lingual_count} cross-lingual)")
        return overlaps
    except Exception as e:
        print(f"FAISS search failed: {e}")
        return []


# 6. OPTIMIZED PIPELINE EXECUTION FOR STRUCTURED JSON


def embed_and_save_documents_structured(folder_path, output_file="financial_regulation_embeddings.jsonl", batch_size=BATCH_SIZE, max_files=None):
    """Process ALL structured JSON documents in batches."""
    print(f"Starting embedding process for structured JSON documents in {folder_path}")
    
    all_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".json")]
    if max_files:
        all_files = all_files[:max_files]
    
    total_docs = len(all_files)
    print(f"Total structured JSON documents to process: {total_docs}")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        processed_count = 0
        batch_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch_files = all_files[i:i + batch_size]
            current_batch = []
            
            for filename in batch_files:
                fp = os.path.join(folder_path, filename)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    
                    # Use the structured extraction functions
                    text_for_embedding = extract_text_for_embedding(raw)
                    requirements = extract_requirements_from_structured_json(raw)
                    doc_id = filename.replace(".json", "")
                    language = detect_language(text_for_embedding)
                    
                    current_batch.append({
                        "id": doc_id,
                        "filename": filename,
                        "text": text_for_embedding,
                        "language": language,
                        "embedding": None,
                        "cluster": None,
                        "risk_category": None,
                        "requirements": requirements,
                        "raw_json": raw
                    })
                except Exception as e:
                    print(f"Failed to load {fp}: {e}")
                    continue
            
            for doc in tqdm(current_batch, desc=f"Embedding batch {batch_count}"):
                try:
                    doc["embedding"] = gemini_embedding(doc["text"])
                    out_f.write(json.dumps({
                        "id": doc["id"],
                        "filename": doc["filename"],
                        "text": doc["text"],
                        "language": doc["language"],
                        "embedding": doc["embedding"],
                        "cluster": doc["cluster"],
                        "risk_category": doc["risk_category"],
                        "requirements": doc["requirements"],
                    }) + "\n")
                    out_f.flush()
                    processed_count += 1
                except Exception as e:
                    print(f"Failed to embed document {doc['id']}: {e}")
                    continue
            
            batch_count += 1
            print(f"Processed {processed_count}/{total_docs} documents")
            gc.collect()
    
    print(f"✅ Completed embedding {processed_count} structured JSON documents")

def load_embedded_documents(file_path="financial_regulation_embeddings.jsonl", max_docs=None):
    """Load with optional document limit."""
    documents = []
    if not os.path.exists(file_path):
        print(f"Embeddings file not found: {file_path}")
        return documents
        
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            try:
                documents.append(json.loads(line))
            except Exception as e:
                print(f"Failed to load line {i}: {e}")
                continue
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def run_pipeline_structured(folder_path, embedding_output="financial_regulation_embeddings.jsonl", max_docs=None, force_reembed=False):
    """
    Enhanced pipeline for structured JSON format with Requirement arrays.
    """
    print("Starting pipeline for structured Requirement JSON format...")
    start_time = datetime.now()
    
    # Step 1: Embed documents using structured JSON processing
    if force_reembed or not os.path.exists(embedding_output):
        print("🔄 Embedding structured JSON documents...")
        embed_and_save_documents_structured(folder_path, embedding_output, max_files=max_docs)
    else:
        print("📁 Using existing embeddings...")
        existing_docs = load_embedded_documents(embedding_output, max_docs=1)
        if len(existing_docs) < 1:
            print("🔄 Existing embeddings file is empty or corrupted. Re-embedding...")
            embed_and_save_documents_structured(folder_path, embedding_output, max_files=max_docs)
    
    # Step 2: Load documents
    print("Loading documents...")
    documents = load_embedded_documents(embedding_output, max_docs)
    print(f"Loaded {len(documents)} documents")
    
    if len(documents) == 0:
        print("❌ No documents found! Exiting pipeline.")
        return [], [], [], nx.Graph()
    
    # Step 3: Enhanced cross-lingual clustering
    num_clusters = max(1, min(10, len(documents)))
    print(f"Clustering {len(documents)} documents into {num_clusters} clusters using cross-lingual semantic similarity...")
    documents = cluster_documents_crosslingual(documents, num_clusters=num_clusters)
    
    # Step 4: Risk labels
    print("Assigning risk labels...")
    documents = assign_risk_labels(documents)
    
    # Step 5: Use pre-extracted requirements from structured JSON (no LLM extraction needed)
    print("Using pre-extracted requirements from structured JSON...")
    documents = extract_all_requirements_structured(documents)
    
    # Step 6: Cross-lingual overlap detection
    overlaps = []
    if len(documents) > 1:
        print("Finding cross-lingual semantic overlaps...")
        overlaps = find_overlapping_requirements_crosslingual(documents, threshold=0.85, k=5)
    else:
        print("Skipping overlap detection (only 1 document)")
    
    # Step 7: Contradictions
    contradictions = []
    if overlaps:
        print("Finding contradictions...")
        contradictions = find_contradictions(documents, overlaps[:10])
    else:
        print("Skipping contradiction detection (no overlaps found)")
    
    # Step 8: Build graph
    print("Building graph...")
    G = build_graph(documents, overlaps, contradictions)
    
    # Save results
    print("Saving results...")
    save_processed_docs(documents)
    
    try:
        with open("clean1_regulation_graph.pickle", "wb") as f:
            pickle.dump(G, f)
        print("✅ Graph saved successfully")
    except Exception as e:
        print(f"Failed to save graph: {e}")
    
    save_relationships(overlaps, contradictions)
    
    # Cleanup
    gc.collect()
    
    end_time = datetime.now()
    print(f"✅ Structured JSON pipeline completed in {end_time - start_time}")
    print(f"📊 Processed {len(documents)} documents, found {len(overlaps)} overlaps, {len(contradictions)} contradictions")
    
    # Print cross-lingual insights
    cross_lingual_overlaps = [o for o in overlaps if o.get("cross_lingual", False)]
    if cross_lingual_overlaps:
        print(f"🌍 Found {len(cross_lingual_overlaps)} cross-lingual semantic matches between Finnish and English regulations!")
    
    return documents, overlaps, contradictions, G

# Keep other utility functions the same
def assign_risk_labels(documents):
    """Batch process risk label assignment."""
    clusters = list(set([d["cluster"] for d in documents]))
    print(f"Assigning risk labels to {len(clusters)} clusters...")
    
    for c in clusters:
        cluster_docs = [d for d in documents if d["cluster"] == c]
        print(f"LLM summarizing cluster {c} ({len(cluster_docs)} documents)...")
        label = summarize_cluster_with_llm(documents, c)
        for d in documents:
            if d["cluster"] == c:
                d["risk_category"] = label
        time.sleep(2)
    
    return documents

def summarize_cluster_with_llm(documents, cluster_id, max_docs=2):
    """Optimized cluster summarization."""
    texts = [d["text"] for d in documents if d["cluster"] == cluster_id]
    if not texts:
        return f"cluster_{cluster_id}"
    
    sample = "\n---\n".join(texts[:max_docs])
    prompt = f"""
    Analyze these financial regulation texts and identify the main risk category.
    Respond with ONLY 2-3 words describing the risk category.

    Examples: "credit risk", "liquidity risk", "operational risk", "market risk"

    Texts:
    {sample[:2000]}
    """
    try:
        result = rate_limited_call(gemini_generate, prompt)
        result = result.strip().split('\n')[0].strip()
        return result if result else f"cluster_{cluster_id}"
    except Exception as e:
        print(f"Cluster summarization failed for cluster {cluster_id}: {e}")
        return f"cluster_{cluster_id}"

def is_contradiction(req1, req2):
    prompt = f"""
    
    You are a regulatory compliance analyst. Compare two financial regulatory requirements.

    Task:
    Determine whether the two requirements define DIFFERENT SPECIFICATIONS for the SAME POLICY REQUIREMENT.

    Rules:
    - Focus ONLY on differences in limits, thresholds, numbers, timing, conditions, triggers, or procedural requirements.
    - Ignore superficial wording differences (e.g., "valuer" vs "appraiser").
    - If both requirements specify the SAME rule in different wording → answer "NO".
    - If they specify DIFFERENT thresholds, numbers, or procedural rules → answer "YES".

    Output:
    Respond ONLY with "YES" or "NO".

    Requirement A:
    {req1}

    Requirement B:
    {req2}

    """
    try:
        out = rate_limited_call(gemini_generate, prompt)
        return "YES" in out.upper()
    except:
        return False

def find_contradictions(documents, overlaps, max_checks=20):
    contradictions = []
    checks = 0
    for o in overlaps:
        if checks >= max_checks:
            break
        try:
            doc1 = next(d for d in documents if d["id"] == o["doc1"])
            doc2 = next(d for d in documents if d["id"] == o["doc2"])
            if is_contradiction(o["req1"], o["req2"]):
                contradictions.append(o)
            checks += 1
            time.sleep(1)
        except StopIteration:
            continue
    return contradictions

def build_graph(documents, overlaps, contradictions):
    G = nx.Graph()
    for d in documents:
        G.add_node(d["id"], type="document", risk=d["risk_category"], requirements=d["requirements"], cluster=d["cluster"])
    for o in overlaps:
        G.add_edge(o["doc1"], o["doc2"], type="overlap", similarity=o["similarity"])
    for c in contradictions:
        G.add_edge(c["doc1"], c["doc2"], type="contradiction", req1=c["req1"], req2=c["req2"])
    return G

def save_processed_docs(documents, path="clean1_processed_documents.json"):
    serializable = []
    for d in documents:
        serializable.append({
            "id": d["id"],
            "text": d["text"][:500] + "..." if len(d["text"]) > 500 else d["text"],
            "language": d.get("language", "unknown"),
            "embedding": "..." if d["embedding"] else [],
            "cluster": d["cluster"],
            "risk_category": d["risk_category"],
            "requirements": d["requirements"],
        })
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

def save_relationships(overlaps, contradictions, path="clean1_relationships.json"):
    with open(path, "w") as f:
        json.dump({"overlaps": overlaps, "contradictions": contradictions}, f, indent=2)


# RUN STRUCTURED JSON PIPELINE


if __name__ == "__main__":
    genai.configure(api_key=API_KEY)
    
    # Run the structured JSON pipeline
    run_pipeline_structured(
        "/home/rishika/Suomen-Pankki-Junction-2025/CLEAN_FINANCIAL_REGULATION",
        max_docs=None,
        force_reembed=True
    )