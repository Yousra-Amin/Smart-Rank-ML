import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw")

top1000_file = os.path.join(data_path, "top1000.dev")
queries_file = os.path.join(data_path, "queries.dev.tsv")
qrels_file = os.path.join(data_path, "qrels.dev.tsv")

print("Checking files...")
print("Top1000 exists:", os.path.exists(top1000_file))
print("Queries exists:", os.path.exists(queries_file))
print("Qrels exists:", os.path.exists(qrels_file))

print("\nFile sizes:")
print("top1000.dev:", os.path.getsize(top1000_file), "bytes")
print("queries.dev.tsv:", os.path.getsize(queries_file), "bytes")
print("qrels.dev.tsv:", os.path.getsize(qrels_file), "bytes")

print("\nLoading first 5 rows of top1000.dev...")
preview = pd.read_csv(
    top1000_file,
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    nrows=5
)
print(preview)

print("\nLoading queries and qrels...")
queries = pd.read_csv(
    queries_file,
    sep="\t",
    names=["qid", "query"]
)

qrels = pd.read_csv(
    qrels_file,
    sep="\t",
    names=["qid", "unused", "pid", "relevance"]
)

print("\nQueries shape:", queries.shape)
print("Qrels shape:", qrels.shape)

print("\nQueries preview:")
print(queries.head())

print("\nQrels preview:")
print(qrels.head())

print("\nNow counting rows in full top1000.dev safely...")
row_count = 0
chunk_number = 0

for chunk in pd.read_csv(
    top1000_file,
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    chunksize=50000
):
    row_count += len(chunk)
    chunk_number += 1
    print(f"Processed chunk {chunk_number}, total rows so far: {row_count}")

print("\nTotal rows in top1000.dev:", row_count)


# 🔽 ADD THIS BELOW 🔽

missing_query = 0
missing_passage = 0

for chunk in pd.read_csv(
    top1000_file,
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    chunksize=50000
):
    missing_query += chunk["query"].isnull().sum()
    missing_passage += chunk["passage"].isnull().sum()

print("\nMissing query values:", missing_query)
print("Missing passage values:", missing_passage)