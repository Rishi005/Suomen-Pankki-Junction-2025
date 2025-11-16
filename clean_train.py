"""
Simplified JSON cleaner + Chroma RAG index builder
- No LangChain dependencies
- No complex text splitting
- Direct ChromaDB + Google AI usage
"""

import os
import sys
import json
import time
import hashlib
import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
import google.generativeai as genai
import random


# Config / Defaults

DEFAULT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
BATCH_SIZE = 3
HASH_FILENAME = "file_hashes.json"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


# Simple Document Class


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Simple Text Splitter (Basic chunking)


def simple_text_splitter(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Simple text splitter that breaks text into chunks"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence end
        sentence_breaks = ['. ', '! ', '? ', '\n\n', '\n']
        found_break = -1
        for breaker in sentence_breaks:
            pos = text.rfind(breaker, start, end)
            if pos > start:
                found_break = pos + len(breaker)
                break
        
        if found_break != -1:
            chunks.append(text[start:found_break])
            start = found_break
        else:
            # Just break at the max size
            chunks.append(text[start:end])
            start = end
    
    return chunks


# ChromaDB Vector Store


class ChromaVectorStore:
    def __init__(self, persist_directory: str, embedding_function: Any = None):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_function
        
        try:
            self.collection = self.client.get_collection("documents")
        except Exception:
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, documents: List[Document]):
        if not documents:
            return
        
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{hash(doc.page_content)}_{i}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        return ChromaRetriever(self.collection, search_kwargs or {})

class ChromaRetriever:
    def __init__(self, collection, search_kwargs: Dict[str, Any]):
        self.collection = collection
        self.search_kwargs = search_kwargs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.collection.query(
            query_texts=[query],
            n_results=self.search_kwargs.get("k", 3)
        )
        
        documents = []
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))
        
        return documents


# Google AI Embeddings


class GoogleGenerativeAIEmbeddings:
    def __init__(self, model: str = "models/text-embedding-004"):
        self.model = model
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text
                )
                embeddings.append(result["embedding"])
            except Exception as e:
                logging.error(f"Embedding failed: {e}")
                embeddings.append([0.0] * 768)
        return embeddings


# Enhanced Gemini Client


class GeminiClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        if not api_key:
            raise ValueError("API key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.last_call_time = 0
        self.min_call_interval = 3.0  # Increased to avoid rate limits

    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                current_time = time.time()
                time_since_last_call = current_time - self.last_call_time
                if time_since_last_call < self.min_call_interval:
                    sleep_time = self.min_call_interval - time_since_last_call + random.uniform(1, 2)
                    logging.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                
                time.sleep(random.uniform(0.5, 1.0))
                
                logging.info(f"Gemini API call (attempt {attempt + 1})...")
                response = self.model.generate_content(prompt)
                self.last_call_time = time.time()
                
                if response.text:
                    return response.text
                else:
                    raise ValueError("Empty response")
                    
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"Gemini call failed (attempt {attempt + 1}): {error_msg}")
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    delay = [10, 30, 60][min(attempt, 2)]
                    logging.info(f"Rate limit hit, waiting {delay}s...")
                    time.sleep(delay)
                else:
                    delay = [5, 10, 20][min(attempt, 2)]
                    logging.info(f"API error, waiting {delay}s...")
                    time.sleep(delay)
        
        return '[]'


# JSON Processing


CLEAN_PROMPT_TEMPLATE = """Extract all of the Requirement content from these JSON files. 
For each file, return exactly:
{
  "Requirement": [ {"name": "string", "id": "string", "content": "string"} ],
  "risk_type": "market/credit/liquidity/operational"
}

Return ONLY a JSON array. No explanations. If no Requirement, use {"Requirement": [], "risk_type": "operational"}.

Files:
{files_block}

Output (JSON array only):"""

def build_batch_prompt(batch_inputs: List[Tuple[str, str]]) -> str:
    blocks = []
    for i, (path, content) in enumerate(batch_inputs):
        safe_path = os.path.basename(path)
        if len(content) > 50000:  # More aggressive truncation
            content = content[:50000] + "\n...TRUNCATED..."
        blocks.append(f"FILE {i}: {safe_path}\n{content}\n")
    return CLEAN_PROMPT_TEMPLATE.replace("{files_block}", "\n".join(blocks))

def safe_json_parse(json_str: str, batch_index: int, debug_dir: str) -> List[Dict]:
    json_str = json_str.strip()
    
    # Try direct parse first
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
    except:
        pass
    
    # Try to extract JSON from markdown
    try:
        if '```json' in json_str:
            start = json_str.find('```json') + 7
            end = json_str.find('```', start)
            if end != -1:
                json_content = json_str[start:end].strip()
                return json.loads(json_content)
    except:
        pass
    
    # Try to find array pattern
    try:
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(json_str[start:end])
    except:
        pass
    
    logging.error(f"JSON parsing failed for batch {batch_index}")
    debug_path = os.path.join(debug_dir, f"batch_{batch_index}_failed.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    
    return []


# Utility Functions


def compute_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def json_to_docs(file_path: str) -> List[Document]:
    """Convert JSON file to documents with simple chunking"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        raw = load_json_text(file_path)
        doc = Document(page_content=raw, metadata={"source": str(file_path)})
        chunks = simple_text_splitter(raw)
        return [Document(page_content=chunk, metadata={"source": str(file_path), "chunk": i}) 
                for i, chunk in enumerate(chunks)]

    # Convert data to string
    if isinstance(data, (dict, list)):
        text_content = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        text_content = str(data)
    
    # Split into chunks
    chunks = simple_text_splitter(text_content)
    
    return [Document(
        page_content=chunk, 
        metadata={"source": str(file_path), "chunk": i}
    ) for i, chunk in enumerate(chunks)]


# Main Pipeline


def main(
    data_dir: str,
    persist_dir: str = "persist",
    cleaned_out: str = "cleaned_json",
    api_key: str = None,
    model: str = DEFAULT_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
    force_rebuild: bool = False,
):
    start_time = time.time()

    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Please provide GOOGLE_API_KEY")

    data_dir = os.path.abspath(data_dir)
    persist_dir = os.path.abspath(persist_dir)
    cleaned_out = os.path.abspath(cleaned_out)

    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs(cleaned_out, exist_ok=True)

    # Load existing hashes
    hashes_path = os.path.join(persist_dir, HASH_FILENAME)
    stored_hashes = {}
    if os.path.exists(hashes_path):
        try:
            with open(hashes_path, "r", encoding="utf-8") as f:
                stored_hashes = json.load(f)
        except Exception:
            stored_hashes = {}

    # Find JSON files
    all_files = glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    logging.info(f"Found {len(all_files)} JSON files")

    # Compute hashes
    changed_files = []
    current_hashes = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(compute_md5, path): path for path in all_files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                h = fut.result()
                current_hashes[path] = h
                if stored_hashes.get(path) != h:
                    changed_files.append(path)
            except Exception as e:
                logging.warning(f"Failed hashing {path}: {e}")

    logging.info(f"Changed files: {len(changed_files)}")

    # Initialize Gemini
    gemini = GeminiClient(api_key=api_key, model=model)

    # Process files
    target_files = all_files if force_rebuild else changed_files
    successful_batches = 0
    
    if target_files:
        logging.info(f"Processing {len(target_files)} files in batches of {batch_size}")
        batch_inputs = []
        for path in target_files:
            try:
                txt = load_json_text(path)
                batch_inputs.append((path, txt))
            except Exception as e:
                logging.warning(f"Failed to read {path}: {e}")

        # Process batches
        total_batches = (len(batch_inputs) + batch_size - 1) // batch_size
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            batch_num = i // batch_size + 1
            logging.info(f"Batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            prompt = build_batch_prompt(batch)
            try:
                out = gemini.generate_with_retry(prompt)
            except Exception as e:
                logging.error(f"Gemini call failed for batch {batch_num}: {e}")
                continue

            # Parse response
            cleaned_list = safe_json_parse(out, batch_num, cleaned_out)
            if not cleaned_list:
                continue

            # Save cleaned files
            success_count = 0
            for (path, _), cleaned in zip(batch, cleaned_list):
                out_path = os.path.join(cleaned_out, os.path.basename(path))
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned, f, ensure_ascii=False, indent=2)
                    success_count += 1
                except Exception as e:
                    logging.error(f"Failed saving {path}: {e}")
            
            if success_count > 0:
                successful_batches += 1
                logging.info(f"✅ Batch {batch_num}: {success_count}/{len(batch)} files saved")

    # Save hashes
    try:
        with open(hashes_path, "w", encoding="utf-8") as f:
            json.dump(current_hashes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Couldn't save hashes: {e}")

    # Build vector store if we processed anything
    if successful_batches > 0 or force_rebuild:
        logging.info("Building Chroma vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        if force_rebuild:
            logging.info("Force rebuild: creating new index")
            all_docs = []
            for path in all_files:
                all_docs.extend(json_to_docs(path))
            
            if all_docs:
                vectorstore = ChromaVectorStore(persist_directory=persist_dir, embedding_function=embeddings)
                # Clear and recreate collection
                try:
                    vectorstore.client.delete_collection("documents")
                except:
                    pass
                vectorstore = ChromaVectorStore(persist_directory=persist_dir, embedding_function=embeddings)
                vectorstore.add_documents(all_docs)
                logging.info(f"✅ Built index with {len(all_docs)} chunks")
        else:
            # Update existing
            try:
                vectorstore = ChromaVectorStore(persist_directory=persist_dir, embedding_function=embeddings)
                if changed_files:
                    new_docs = []
                    for path in changed_files:
                        new_docs.extend(json_to_docs(path))
                    
                    if new_docs:
                        # Remove old versions
                        for path in changed_files:
                            try:
                                vectorstore.collection.delete(where={"source": path})
                            except:
                                pass
                        
                        vectorstore.add_documents(new_docs)
                        logging.info(f"Updated index with {len(new_docs)} new chunks")
            except Exception as e:
                logging.warning(f"Failed to update store: {e}")

    elapsed = time.time() - start_time
    logging.info(f"Completed in {elapsed:.1f}s")
    return {"successful_batches": successful_batches}

if __name__ == "__main__":
    # Run with your paths
    main(
        data_dir="/home/rishika/Suomen-Pankki-Junction-2025/FINANCIAL_REGULATION",
        persist_dir="/home/rishika/Suomen-Pankki-Junction-2025/persist_dir",
        cleaned_out="/home/rishika/Suomen-Pankki-Junction-2025/CLEAN2_FINANCIAL_REGULATION",
        api_key="API_KEY",
        batch_size=2,  # Even smaller batches
        force_rebuild=True,
    )
