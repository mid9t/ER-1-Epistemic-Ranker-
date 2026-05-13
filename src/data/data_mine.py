import random
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from datasets import load_dataset
import ir_datasets


# ================= CONFIG =================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Epic Ranker Ratio: 1 pos : 1 hard neg : 2 easy negs
NUM_QUERIES = 50_000 
EASY_NEGS_PER_QUERY = 2
HARD_NEGS_PER_QUERY = 1
OOD_SIZE = 5000

random.seed(42)

print("=== Phase 2 Data Mining: Optimized IR Pipeline ===")

# 1. Load MS MARCO + Core Data
dataset = ir_datasets.load("msmarco-passage")
queries_df = pd.read_csv(RAW_DIR / "queries.train.tsv", sep="\t", header=None, names=["query_id", "query_text"])

# FIX 1: Use \s+ to handle space-separated TREC format properly
qrels = pd.read_csv(RAW_DIR / "qrels.train.tsv", sep=r"\s+", engine="python", header=None, names=["query_id", "0", "passage_id", "label"])

# 2. Fast Lookups (The Dictionary Optimization)
qrel_dict = qrels.groupby("query_id")["passage_id"].apply(list).to_dict()

print("Indexing hard negatives for O(1) lookup...")
hard_neg_ds = load_dataset(
    "sentence-transformers/msmarco-hard-negatives",
    data_files="msmarco-hard-negatives-bm25_1k.jsonl.gz",
    split="train",
    streaming=True,
)

# monitor = BakeMonitor(window_size=1000)

hard_neg_dict = {}
for row in tqdm(hard_neg_ds, desc="Mapping Hard Negs"):
    qid = int(row["qid"])
    hard_neg_dict[qid] = row["neg"]["bm25"]
    # monitor.update()

# monitor.save_log()

# 3. Pair Construction Loop
pairs = []
queries_to_process = queries_df.head(NUM_QUERIES)

doc_store = dataset.docs_store()

for _, row in tqdm(queries_to_process.iterrows(), total=len(queries_to_process), desc="Baking Pairs"):
    qid = row["query_id"]
    q_text = row["query_text"]
    
    # 1. Positives
    if qid in qrel_dict:
        raw_pid = qrel_dict[qid][0] # Take first positive
        
        # FIX 2: Check for NaN and safely cast via int to strip any `.0` floats
        if pd.notna(raw_pid):
            pid = str(int(raw_pid)) 
            try:
                doc = doc_store.get(pid)
                if doc:
                    pairs.append({"query_id": qid, "query_text": q_text, "passage_id": pid, 
                                  "passage_text": doc.text, "label": 1, "pair_type": "positive"})
            except KeyError:
                pass # Skip if doc genuinely doesn't exist in the corpus

    # 2. Hard Negatives 
    try:
        q_key = int(qid) # Reuse the 'qid' extracted at the top of the loop
        if q_key in hard_neg_dict:
            for pid in hard_neg_dict[q_key][:HARD_NEGS_PER_QUERY]:
                pid_str = str(int(pid)) # Safe cast
                try:
                    doc = doc_store.get(pid_str)
                    if doc:
                        pairs.append({"query_id": qid, "query_text": q_text, "passage_id": pid_str, 
                                    "passage_text": doc.text, "label": 0, "pair_type": "hard_negative"})
                except KeyError:
                    pass
    except (ValueError, TypeError):
        pass # If qid is strangely malformed, just skip hard negatives, don't break the loop

    # 3. Easy Negatives (Random)
    for _ in range(EASY_NEGS_PER_QUERY):
        rand_pid = str(random.randint(0, 8841823))
        try:
            doc = doc_store.get(rand_pid)
            if doc:
                pairs.append({"query_id": qid, "query_text": q_text, "passage_id": rand_pid, 
                              "passage_text": doc.text, "label": 0, "pair_type": "easy_negative"})
        except KeyError:
            pass

# 4. Improved OOD Generation (Shuffled Passages)
print("Generating OOD slice (shuffled semantics)...")
ood = []
# Ensure we have docs to sample from
if pairs:
    sample_docs = [pairs[i]['passage_text'] for i in range(min(len(pairs), 500))]
    for _ in range(OOD_SIZE):
        source_text = random.choice(sample_docs).split()
        random.shuffle(source_text)
        ood.append({
            "query_id": -1, "query_text": "gibberish query",
            "passage_id": -1, "passage_text": " ".join(source_text),
            "label": 0, "pair_type": "ood"
        })

# 5. Save to Optimized Parquet
df_pairs = pd.DataFrame(pairs)
df_ood = pd.DataFrame(ood)

# Ensure data types are optimized for Parquet
if not df_pairs.empty:
    df_pairs['label'] = df_pairs['label'].astype('int8')
    df_pairs.to_parquet(PROCESSED_DIR / "train_pairs.parquet", index=False, compression='snappy')
if not df_ood.empty:
    df_ood.to_parquet(PROCESSED_DIR / "ood_slice.parquet", index=False, compression='snappy')

print(f"✅ Success! Training pairs saved to {PROCESSED_DIR}")