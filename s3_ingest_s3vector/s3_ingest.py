# s3_ingest.py

import os
import boto3

from llama_index.core import (
    Document,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# %%

# ---------- Config from environment ----------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
AWS_PROFILE = os.environ.get("AWS_PROFILE")

# S3 where raw docs live
S3_BUCKET_ENV = os.environ.get("S3_BUCKET", "skiing-coach")
S3_OBJECT_KEY_ENV = os.environ.get("S3_OBJECT_KEY")

# S3 Vectors config
# You must create this vector bucket + index ahead of time
VECTOR_BUCKET_NAME = os.environ.get("VECTOR_BUCKET_NAME", "skiing-rag-vectors")      # e.g. "skiing-rag-vectors"
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "skiing-rag-index")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))


# ---------- AWS session & clients ----------
if os.getenv("AWS_EXECUTION_ENV") or os.getenv("ECS_CONTAINER_METADATA_URI"):
    AWS_PROFILE = None

if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    session = boto3.Session(region_name=AWS_REGION)

s3_client = session.client("s3")
s3vectors_client = session.client("s3vectors")


# ---------- LLM & embedding config (must match your agent) ----------

llm = OpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

embed_model = OpenAIEmbedding(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIM,
    api_key=OPENAI_API_KEY,
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024


# %%

# ---------- Helpers ----------

def fetch_s3_text(bucket: str, key: str) -> str:
    """Download one S3 object and decode as UTF-8 text."""
    print(f"[INFO] Downloading s3://{bucket}/{key}")
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("utf-8", errors="ignore")


def build_document(bucket: str, key: str) -> Document | None:
    """Build a LlamaIndex Document from one S3 object."""
    text = fetch_s3_text(bucket, key)
    if not text.strip():
        print("[WARN] S3 object is empty; skipping")
        return None

    metadata = {
        "source_bucket": bucket,
        "source_key": key,
        "source_url": f"s3://{bucket}/{key}",
        "title": os.path.basename(key),
    }

    # doc_id makes re-ingest idempotent per file
    doc_id = f"s3://{bucket}/{key}"

    return Document(
        text=text,
        metadata=metadata,
        doc_id=doc_id,
    )


def _nodes_to_s3_vectors(doc: Document, nodes: list) -> None:
    """
    Convert nodes to embeddings and write them into an S3 Vectors index.
    We batch in chunks (<= 500 per call) to follow S3 Vectors guidance.
    """
    vectors = []

    for idx, node in enumerate(nodes):
        content = node.get_content()
        if not content.strip():
            continue

        # Get embedding for this chunk
        embedding = embed_model.get_text_embedding(content)

        key = f"{doc.doc_id}#chunk-{idx}"

        # Metadata must be JSON-serializable and <= limits

        preview = content[:400]  # adjust length if you like

        metadata = {
            "doc_id": doc.doc_id,
            "chunk_index": idx,
            "title": doc.metadata.get("title"),
            "source_bucket": doc.metadata.get("source_bucket"),
            "source_key": doc.metadata.get("source_key"),
            "text_preview": preview,
        }

        vectors.append(
            {
                "key": key,
                "data": {"float32": embedding},
                "metadata": metadata,
            }
        )

    if not vectors:
        print("[INFO] No non-empty chunks to index.")
        return

    # S3 Vectors recommends batching writes (up to 500 per request) :contentReference[oaicite:1]{index=1}
    BATCH_SIZE = 500
    for start in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[start : start + BATCH_SIZE]
        print(f"[INFO] Writing {len(batch)} vectors to S3 Vectors "
              f"(bucket={VECTOR_BUCKET_NAME}, index={VECTOR_INDEX_NAME})")
        s3vectors_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=batch,
        )


def ingest_single_object(bucket: str, key: str) -> None:
    """Ingest one S3 object into the S3 Vectors index."""
    print(f"[INFO] Ingesting s3://{bucket}/{key}")

    doc = build_document(bucket, key)
    if doc is None:
        print("[INFO] Nothing to ingest.")
        return

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents([doc])

    _nodes_to_s3_vectors(doc, nodes)
    print("[INFO] Ingest finished.")



# %%
# ---------- Entry point ----------

if __name__ == "__main__":
    if not S3_BUCKET_ENV or not S3_OBJECT_KEY_ENV:
        raise SystemExit("S3_BUCKET and S3_OBJECT_KEY must be set in the environment.")

    ingest_single_object(S3_BUCKET_ENV, S3_OBJECT_KEY_ENV)


