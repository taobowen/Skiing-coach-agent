import os
import boto3
import asyncio
from typing import List
from langdetect import detect
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.opensearch import OpensearchVectorStore, OpensearchVectorClient
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth

# -----------------------------
# Config (env overrides allowed)
# -----------------------------
PROFILE_NAME = os.getenv("AWS_PROFILE")
AWS_REGION   = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET    = os.getenv("CORPUS_BUCKET", "skiing-coach")
S3_PREFIX    = os.getenv("CORPUS_PREFIX", "RagDoc/")  # <-- include trailing slash
PERSIST_DIR  = os.getenv("RAG_STORE_DIR", "rag_store")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DIM = 1024   # <- IMPORTANT: bge-m3 outputs 1024
TEXT_FIELD = "text"
VEC_FIELD = "vec"
SERVICE = 'aoss'

# OpenSearch Serverless
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "skiing-rag-docs")  # must already exist with vec mapping
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")


# %%
# -----------------------------
# AWS clients
# -----------------------------
session = boto3.Session(profile_name=PROFILE_NAME, region_name=AWS_REGION)
s3 = session.client("s3")
translate = session.client("translate", region_name=AWS_REGION)

sts = session.client("sts")
creds = session.get_credentials()
auth = AWSV4SignerAuth(creds, AWS_REGION, SERVICE)
print("Boto3 identity:", sts.get_caller_identity())   # compare to `aws sts get-caller-identity`
print("Boto3 region: ", session.region_name)

# %%
def list_txt_keys(bucket: str, prefix: str) -> List[str]:
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.lower().endswith(".txt"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def get_s3_text(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="ignore")


def detect_lang(text: str) -> str:
    try:
        return detect(text[:1000]) if text.strip() else "en"
    except Exception:
        return "en"


def zh_to_en(text: str) -> str:
    if not text.strip():
        return text
    out = translate.translate_text(Text=text, SourceLanguageCode="zh", TargetLanguageCode="en")
    return out["TranslatedText"]


# -----------------------------
# Ingest pipeline
# -----------------------------
def load_and_pretranslate_docs(bucket: str, prefix: str) -> List[Document]:
    keys = list_txt_keys(bucket, prefix)
    print(f"[INFO] Found {len(keys)} .txt files under s3://{bucket}/{prefix}")
    docs: List[Document] = []
    for k in keys:
        try:
            raw = get_s3_text(bucket, k)
        except Exception as e:
            print(f"[WARN] Read failed: s3://{bucket}/{k} :: {e}")
            continue

        lang = detect_lang(raw)
        text_for_index = zh_to_en(raw) if lang.startswith("zh") else raw

        meta = {
            "source_bucket": bucket,
            "source_key": k,
            "source_url": f"s3://{bucket}/{k}",
            "title": os.path.basename(k),
            "original_lang": "zh" if lang.startswith("zh") else "en",
            # Optional label metadata — set it if you have it:
            # "label": "EL" / "OS" / ...
        }
        docs.append(Document(text=text_for_index, metadata=meta))

    print(f"[INFO] Loaded {len(docs)} documents (pre-translated zh→en when needed)")
    return docs

# %%
# 1) Embeddings (set BEFORE building index)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY, logprobs=False, default_headers={})
Settings.llm = llm
Settings.chunk_size = 512
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
Settings.embed_model = embed_model


# 2) Load & pre-translate (EN output only)
raw_docs = load_and_pretranslate_docs(S3_BUCKET, S3_PREFIX)
if not raw_docs:
    print("[WARN] No documents loaded; exiting.")

# 3) Build via LlamaIndex → upserts into OpenSearch
client = OpensearchVectorClient(
    OPENSEARCH_ENDPOINT,
    OPENSEARCH_INDEX,
    DIM,
    embedding_field=VEC_FIELD,
    text_field=TEXT_FIELD,
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
)

vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# %%
index = VectorStoreIndex.from_documents(
    raw_docs,
    storage_context=storage_context,
    transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)]
);

# %%
# Upsert (embeds + writes into your vector DB)
# raw_docs = load_and_pretranslate_docs(S3_BUCKET, S3_PREFIX)
nodes = SentenceSplitter(chunk_size=1024, chunk_overlap=20).get_nodes_from_documents(raw_docs)
index.insert_nodes(nodes)

# # replace by node_id if matching
# index.update_nodes(nodes)         
# # delete by node_id if matching
# index.delete_nodes([node.node_id])

# %%
# Get index
# index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# 4) add a reranking step in the RAG pipeline
# RankLLMRerank
reranker = LLMRerank(
    choice_batch_size=5, top_n=3, llm=llm
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)


# response = query_engine.query("What is CASI ?")
# print(response)

# %%
from typing import List, Literal
from pydantic import BaseModel, Field

class TopError(BaseModel):
    label: str
    count: int
    risk: Literal["Low", "Med", "High"]

class SessionSummary(BaseModel):
    total_errors: int
    top_errors: List[TopError]

class TimelineHighlight(BaseModel):
    label: str
    start: str  # "mm:ss"
    end: str    # "mm:ss"
    confidence: float  # 0–1

class CoachingPlanItem(BaseModel):
    label: str
    priority: int
    cues: List[str]
    drills: List[str]
    practice_terrain: Literal["flats", "green", "blue"]
    focus_timecodes: List[str]  # ["mm:ss-mm:ss", ...]

class SkiingCoachOutput(BaseModel):
    session_summary: SessionSummary
    coaching_note: str
    timeline_highlights: List[TimelineHighlight]
    coaching_plan: List[CoachingPlanItem]
    safety_notes: List[str]


# %%
arxiv_tool = ArxivToolSpec()
search_tool = DuckDuckGoSearchToolSpec()
api_tools = arxiv_tool.to_tool_list() + search_tool.to_tool_list()
skiing_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="skiing_guide_tool",
        description="""
        Retrieval Q&A over a skiing technique knowledge base. Use this to look up definitions,
        biomechanics, drills, and equipment tuning for specific errors (e.g., back-seat stance,
        late edge engagement, A-frame, rotation). Input should be concise (e.g., "drills for
        late edge engagement", "edge angle cues for carving") and the tool returns short,
        actionable guidance to be woven into coaching output.""",
    ),
)
system_prompt = """You are an AI Skiing Coach. The user input is a timeline of detected skiing ERRORS with time slots
(e.g., list of events: {{label, start_time, end_time, confidence}}). Your job is to:

1) Parse and summarize the session:
   - Count total errors; cluster by label; list top 2–3 most frequent/impactful errors.
   - Identify any CRITICAL safety issues (e.g., back-seat at high speed, loss of edge on steep, runaway skis).

2) Diagnose causes & risk:
   - For each top error, explain likely root causes (balance/fore-aft, edging, rotation, pressure timing).
   - Note risk level (Low/Med/High) and when it spikes (specific timecodes).

3) Give targeted coaching tied to timecodes:
   - Provide 1–3 concise cues and 2–3 specific drills per top error.
   - Reference exact time ranges where the error occurs most (e.g., "00:41–00:55, 01:12–01:28").
   - Include terrain/speed suggestions for practice (green/blue, flats, gentle pitch).

4) Equipment & conditions (if relevant):
   - Mention simple checks (boot cuff alignment, forward lean, tune/sharpness) only when they plausibly relate.

5) Output format:
   a JSON object on a separate line with this schema:

   {{
     "session_summary": {{
       "total_errors": <int>,
       "top_errors": [{{"label": <str>, "count": <int>, "risk": "Low|Med|High"}}]
     }},
     "coaching_note": <str>,
     "timeline_highlights": [{{"label": <str>, "start": "<mm:ss>", "end": "<mm:ss>", "confidence": <0-1>}}],
     "coaching_plan": [
       {{
         "label": <str>,
         "priority": 1,
         "cues": ["...","..."],
         "drills": ["...","..."],
         "practice_terrain": "flats|green|blue",
         "focus_timecodes": ["mm:ss-mm:ss", "..."]
       }}
     ],
     "safety_notes": ["...", "..."]
   }}

Rules:
- Prefer concrete, short cues ("hips to hands", "shins to tongue", "tip, roll, pressure").
- Tie every recommendation to observed timecodes when possible.
- Use the skiing_guide_tool to recall drills/definitions when needed.
- If an error label is unknown, infer from context and state the assumption.
- If the input lacks time fields, still provide cues/drills and mark timecodes as [].
- You MUST respond with only a JSON object that matches the provided schema.
- If a field is unknown, still include it with a reasonable default (0, "", []), but never omit required keys.
"""

all_tools = api_tools + [skiing_tool]
agent = FunctionAgent(
    tools=all_tools,
    llm=llm,
    system_prompt=system_prompt,
    output_cls=SkiingCoachOutput,
    allow_parallel_tool_calls = True  # Uncomment this line to allow multiple tool invocations
)

# %%
text_input = "edging, time: 1:00-2:00"

async def main():
    response = await agent.run(user_msg=text_input)

    data = response.structured_response

    # 2) As Pydantic model:
    data_model = response.get_pydantic_model(SkiingCoachOutput)

    print(response, data, data_model)

if __name__ == "__main__":
    asyncio.run(main())


