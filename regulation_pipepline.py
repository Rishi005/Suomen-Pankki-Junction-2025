"""
Full Regulation Processing Pipeline
-----------------------------------

This pipeline processes .di.json financial legislation files and performs:

1. Load & normalize documents
2. Gemini embeddings
3. Cosine-similarity clustering (risk group identification)
4. Requirement extraction using Gemini
5. Overlap detection using cosine similarity
6. Contradiction detection using Gemini
7. Build a knowledge graph (risk cluster, overlaps, contradictions)
8. Save & load all computed data
9. Query any document to retrieve:
   - Risk cluster
   - Overlaps
   - Contradictions
"""

import os
import json
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from tqdm import tqdm 


###############################################################
# 1. GEMINI CONFIGURATION
###############################################################

genai.configure(api_key="API_KEY")

def gemini_generate(prompt, model="gemini-1.5-pro"):
    """General wrapper for Gemini LLM responses."""
    response = genai.GenerativeModel(model).generate_content(prompt)
    return response.text.strip()


# def gemini_embedding(text, model="models/text-embedding-004"):
#     """Return a numeric embedding vector."""
#     result = genai.embed_content(
#         model=model,
#         content=text
#     )
#     return np.array(result["embedding"], dtype=np.float32)


MAX_CHARS = 30000  # Gemini fails above 36k bytes; stay safe ~30k
CHUNK_SIZE = 3000  # adjustable
OVERLAP = 250

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Splits text into overlapping chunks so each can be embedded safely.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap  # maintain coherence

    return chunks


def gemini_embedding(text: str):
    """
    Embeds long documents using chunking + averaging.
    Returns a single embedding vector (list of floats).
    """

    # Empty / weird text → return zero vector
    if not text or not isinstance(text, str):
        return [0.0] * 768

    # Short text → embed directly
    if len(text) < MAX_CHARS:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )["embedding"]

    # Long text → chunk, embed, average
    chunks = chunk_text(text)

    vectors = []
    for chunk in tqdm(chunks):
        try:
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk
            )["embedding"]
            vectors.append(np.array(emb))
        except Exception as e:
            print(f"Embedding failed for chunk: {e}")

    if not vectors:
        return [0.0] * 768

    # Return centroid vector
    final_vec = np.mean(vectors, axis=0)
    return final_vec.tolist()



###############################################################
# 2. LOAD & NORMALIZE ALL .di.json DOCUMENTS
###############################################################

def load_documents(folder_path):

    # folder = Path(folder_path)
    # documents = []

    # for fp in sorted(folder.iterdir()):

    #     with open(fp, "r", encoding="utf-8") as f:
    #         raw = json.load(f)

        # if isinstance(raw.get("pages"), list):

    documents = []
    for filename in os.listdir(folder_path):
        # if filename.endswith(".di.json"):
        # print(filename)
        fp = os.path.join(folder_path, filename)
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)

        paragraphs = []
        for p in raw.get("pages", []):
            # each page may have a 'paragraphs' list (strings)
            page_pars = p.get("paragraphs")
            if isinstance(page_pars, list):
                paragraphs.extend([str(x) for x in page_pars if x is not None])
            # sometimes there's 'text' on the page
            elif isinstance(p.get("text"), str):
                paragraphs.append(p["text"])
        if paragraphs:
            text =  "\n\n".join(paragraphs)

        # doc_id = fp.stem
        doc_id = filename.replace(".di.json", "")

        documents.append({
            "id": doc_id,
            "raw_json": raw,
            "text": text,
            "embedding": None,
            "cluster": None,
            "risk_category": None,
            "requirements": [],
        })

    return documents

# def load_documents(folder_path):
#     """Load all .di.json files and extract plain text."""
#     documents = []
#     for filename in os.listdir(folder_path):
#         # if filename.endswith(".di.json"):
#         print(filename)
#         fp = os.path.join(folder_path, filename)
#         with open(fp, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         doc_id = filename.replace(".di.json", "")

#         # Most .di.json structure: {"text": "...", ...}
#         text = raw.get("paragraphs", "")
#         print(text)

#         documents.append({
#             "id": doc_id,
#             "raw_json": raw,
#             "text": text,
#             "embedding": None,
#             "cluster": None,
#             "risk_category": None,
#             "requirements": [],
#         })
#     return documents


###############################################################
# 3. GENERATE EMBEDDINGS
###############################################################

def embed_documents(documents):
    """Generate embeddings for each document text."""
    for d in documents:
        print(f"Embedding {d['id']}...")
        d["embedding"] = gemini_embedding(d["text"])
    return documents


###############################################################
# 4. CLUSTER DOCUMENTS BY COSINE-SIMILARITY → RISK GROUPS
###############################################################

def cluster_documents(documents, num_clusters=10):
    """Cluster documents into risk categories."""
    embeddings = np.vstack([d["embedding"] for d in documents])

    # Agglomerative clustering chosen for interpretability
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        affinity="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings)

    # Assign cluster labels
    for doc, label in zip(documents, labels):
        doc["cluster"] = int(label)

    return documents


def summarize_cluster_with_llm(documents, cluster_id):
    """LLM assigns a human-readable risk category."""
    texts = [d["text"] for d in documents if d["cluster"] == cluster_id]
    sample = "\n---\n".join(texts[:5])

    prompt = f"""
    You are analyzing FINANCIAL REGULATION.

    Below are several passages from regulatory documents which all fall in the same cluster.
    Your task is to identify the SINGLE MOST LIKELY RISK CATEGORY (e.g., credit risk, liquidity risk, operational risk).

    Provide only a short answer.

    TEXT SAMPLE:
    {sample}
    """

    return gemini_generate(prompt)


def assign_risk_labels(documents):
    clusters = list(set([d["cluster"] for d in documents]))

    for c in clusters:
        print(f"LLM summarizing cluster {c}...")
        label = summarize_cluster_with_llm(documents, c)

        for d in documents:
            if d["cluster"] == c:
                d["risk_category"] = label

    return documents


###############################################################
# 5. REQUIREMENT EXTRACTION (LLM)
###############################################################

def extract_requirements(text):
    """LLM extracts requirements from raw legal text."""
    prompt = f"""
    Extract ALL regulatory requirements from the text below.

    Return ONLY a JSON array of short requirement statements.

    TEXT:
    {text}
    """
    out = gemini_generate(prompt)

    # Try parsing LLM output as JSON
    try:
        return json.loads(out)
    except:
        return []


def extract_all_requirements(documents):
    for d in documents:
        print(f"Extracting requirements for {d['id']}...")
        d["requirements"] = extract_requirements(d["text"])
    return documents


###############################################################
# 6. OVERLAP + CONTRADICTION DETECTION
###############################################################

def find_overlapping_requirements(documents, threshold=0.92):
    """Identify requirement pairs that are essentially duplicates."""
    overlaps = []

    all_reqs = []
    for d in documents:
        for r in d["requirements"]:
            all_reqs.append((d["id"], r))

    # Embed once for efficiency
    req_texts = [r[1] for r in all_reqs]
    req_embs = np.array([gemini_embedding(t) for t in req_texts])

    sim = cosine_similarity(req_embs)

    for i in range(len(all_reqs)):
        for j in range(i+1, len(all_reqs)):
            if sim[i][j] >= threshold:
                overlaps.append({
                    "doc1": all_reqs[i][0],
                    "req1": all_reqs[i][1],
                    "doc2": all_reqs[j][0],
                    "req2": all_reqs[j][1],
                    "similarity": float(sim[i][j])
                })

    return overlaps


def is_contradiction(req1, req2):
    """LLM checks if requirements contradict each other."""
    prompt = f"""
    Determine if the following two regulatory requirements CONTRADICT each other.

    Requirement A:
    {req1}

    Requirement B:
    {req2}

    Respond ONLY with "YES" or "NO".
    """

    out = gemini_generate(prompt, model="gemini-1.5-flash")
    return "YES" in out.upper()


def find_contradictions(documents, overlaps, max_checks=50):
    """
    Find contradictions inside same risk group.
    We limit checks because LLM calls are expensive.
    """
    contradictions = []
    checks = 0

    for o in overlaps:
        if checks >= max_checks:
            break

        doc1 = next(d for d in documents if d["id"] == o["doc1"])
        doc2 = next(d for d in documents if d["id"] == o["doc2"])

        # Only compare if same risk category
        if doc1["risk_category"] != doc2["risk_category"]:
            continue

        if is_contradiction(o["req1"], o["req2"]):
            contradictions.append(o)

        checks += 1

    return contradictions


###############################################################
# 7. BUILD KNOWLEDGE GRAPH
###############################################################

def build_graph(documents, overlaps, contradictions):
    G = nx.Graph()

    # Add document nodes
    for d in documents:
        G.add_node(
            d["id"],
            type="document",
            risk=d["risk_category"],
            requirements=d["requirements"],
            cluster=d["cluster"]
        )

    # Add overlap edges
    for o in overlaps:
        G.add_edge(
            o["doc1"],
            o["doc2"],
            type="overlap",
            similarity=o["similarity"]
        )

    # Add contradiction edges
    for c in contradictions:
        G.add_edge(
            c["doc1"],
            c["doc2"],
            type="contradiction",
            req1=c["req1"],
            req2=c["req2"]
        )

    return G


###############################################################
# 8. SAVE + LOAD FUNCTIONS
###############################################################

def save_processed_docs(documents, path="processed_documents.json"):
    """Save everything except graph."""
    serializable = []
    for d in documents:
        serializable.append({
            "id": d["id"],
            "raw_json": d["raw_json"],
            "text": d["text"],
            "embedding": d["embedding"].tolist(),
            "cluster": d["cluster"],
            "risk_category": d["risk_category"],
            "requirements": d["requirements"],
        })
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_processed_docs(path="processed_documents.json"):
    with open(path) as f:
        return json.load(f)


def save_relationships(overlaps, contradictions, path="relationships.json"):
    with open(path, "w") as f:
        json.dump({
            "overlaps": overlaps,
            "contradictions": contradictions
        }, f, indent=2)


def load_relationships(path="relationships.json"):
    with open(path) as f:
        rel = json.load(f)
    return rel["overlaps"], rel["contradictions"]


###############################################################
# 9. QUERY FUNCTIONS
###############################################################

def query_document(doc_id, documents, overlaps, contradictions):
    """Return risk cluster, overlaps & contradictions for a given document."""
    d = next(doc for doc in documents if doc["id"] == doc_id)

    doc_overlaps = [o for o in overlaps if o["doc1"] == doc_id or o["doc2"] == doc_id]
    doc_contradictions = [c for c in contradictions if c["doc1"] == doc_id or c["doc2"] == doc_id]

    return {
        "id": doc_id,
        "risk_category": d["risk_category"],
        "cluster": d["cluster"],
        "overlaps": doc_overlaps,
        "contradictions": doc_contradictions,
    }


###############################################################
# 10. MAIN PIPELINE
###############################################################

def run_pipeline(folder_path):
    print("Loading documents...")
    documents = load_documents(folder_path)

    print("Embedding...")
    documents = embed_documents(documents)

    print("Clustering...")
    documents = cluster_documents(documents, num_clusters=10)

    print("Assigning risk labels...")
    documents = assign_risk_labels(documents)

    print("Extracting requirements...")
    documents = extract_all_requirements(documents)

    print("Finding overlaps...")
    overlaps = find_overlapping_requirements(documents)

    print("Finding contradictions...")
    contradictions = find_contradictions(documents, overlaps)

    print("Building graph...")
    G = build_graph(documents, overlaps, contradictions)

    print("Saving outputs...")
    save_processed_docs(documents)
    nx.write_gpickle(G, "regulation_graph.gpickle")
    save_relationships(overlaps, contradictions)

    print("Done.")
    return documents, overlaps, contradictions, G


###############################################################
# RUN IF EXECUTED DIRECTLY
###############################################################

if __name__ == "__main__":
    run_pipeline("/home/rishika/Suomen-Pankki-Junction-2025/FINANCIAL_REGULATION")
