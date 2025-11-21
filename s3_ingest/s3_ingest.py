# s3_ingest.py

import os
import boto3

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth


# ---------- Config from environment ----------

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
AWS_PROFILE = os.environ.get("AWS_PROFILE", None)
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]  # e.g. 1wk2ifieb17sgemolaba.us-east-2.aoss.amazonaws.com
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "skiing-rag-docs")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
VEC_FIELD = os.environ.get("VEC_FIELD", "vec")
TEXT_FIELD = os.environ.get("TEXT_FIELD", "text")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

S3_BUCKET_ENV = os.environ.get("S3_BUCKET")
S3_OBJECT_KEY_ENV = os.environ.get("S3_OBJECT_KEY")

# OpenSearch Serverless service name
OPENSEARCH_SERVICE = os.environ.get("OPENSEARCH_SERVICE", "aoss")


# ---------- AWS session & auth ----------

if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
else:
    session = boto3.Session(region_name=AWS_REGION)

creds = session.get_credentials().get_frozen_credentials()
auth = AWS4Auth(
    creds.access_key,
    creds.secret_key,
    AWS_REGION,
    OPENSEARCH_SERVICE,
    session_token=creds.token,
)

s3_client = session.client("s3")


# ---------- LLM & embedding config (must match your agent) ----------

llm = OpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

embed_model = OpenAIEmbedding(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIM,
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512


# ---------- OpenSearch vector store ----------

vector_client = OpensearchVectorClient(
    OPENSEARCH_ENDPOINT,
    OPENSEARCH_INDEX,
    EMBEDDING_DIM,
    embedding_field=VEC_FIELD,
    text_field=TEXT_FIELD,
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

vector_store = OpensearchVectorStore(vector_client)


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


def ingest_single_object(bucket: str, key: str) -> None:
    """Ingest one S3 object into the OpenSearch vector index."""
    print(f"[INFO] Ingesting s3://{bucket}/{key}")

    doc = build_document(bucket, key)
    if doc is None:
        print("[INFO] Nothing to ingest.")
        return

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Attach to existing vector store (no full rebuild)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents([doc])

    index.insert_nodes(nodes)
    print("[INFO] Ingest finished.")


# ---------- Entry point ----------

if __name__ == "__main__":
    if not S3_BUCKET_ENV or not S3_OBJECT_KEY_ENV:
        raise SystemExit("S3_BUCKET and S3_OBJECT_KEY must be set in the environment.")

    ingest_single_object(S3_BUCKET_ENV, S3_OBJECT_KEY_ENV)
