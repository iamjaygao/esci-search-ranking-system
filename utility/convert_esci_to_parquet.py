import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ROOT_DIR
file_path = f"{ROOT_DIR}/esci-data/esci-s_dataset/esci.json.zst"
out_path = f"{ROOT_DIR}/esci-data/esci-s_dataset/esci_s_products.parquet"
chunksize = 100_000 # Only load 100k rows into RAM at a time

# Downstream code (reranking/advanced_features.py) only reads asin/price/stars/
# ratings/category. The raw dataset also has an `attr` struct field whose keys
# vary per row (book metadata), so pyarrow infers a different schema per chunk
# and ParquetWriter.write_table fails once the schema changes between chunks.
KEEP_COLS = ["asin", "price", "stars", "ratings", "category"]

print("Loading data in chunks to prevent memory crash...")

# Create an iterator that yields chunks of the dataframe
reader = pd.read_json(file_path, lines=True, compression="zstd", chunksize=chunksize)

parquet_writer = None

for i, chunk in enumerate(reader):
    print(f"Processing chunk {i+1}...")

    chunk = chunk[[c for c in KEEP_COLS if c in chunk.columns]]
    # category arrives as a list column; join to a string so its Arrow type
    # stays stable across chunks regardless of list length/emptiness.
    if "category" in chunk.columns:
        chunk["category"] = chunk["category"].apply(
            lambda x: " > ".join(x) if isinstance(x, list) else x
        )

    # Convert the pandas chunk to a PyArrow Table
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    # Initialize the writer on the very first chunk so it learns the schema (columns/types)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(out_path, table.schema)

    # Append the chunk to the file
    parquet_writer.write_table(table)

# Close the file connection
if parquet_writer:
    parquet_writer.close()

print(f"Successfully saved full dataset to {out_path}!")