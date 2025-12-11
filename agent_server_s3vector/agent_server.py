# agent_server.py

import os
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.vector_stores.s3 import S3VectorStore  # ✅ S3 Vectors
import boto3

# only for local dev; in prod, set env vars directly
# from dotenv import load_dotenv
# load_dotenv()


# ---------- Config from env ----------

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

# S3 Vectors config (must already exist)
VECTOR_BUCKET_NAME = os.environ.get("VECTOR_BUCKET_NAME", "skiing-rag-vectors")
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "skiing-rag-index")

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


# ---------- AWS session (same style as ingest) ----------

session = boto3.Session(region_name=AWS_REGION)


# ---------- LLM + embedding (must match ingest) ----------

llm = OpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0,
)

embed_model = OpenAIEmbedding(
    model=EMBEDDING_MODEL,
    dimensions=EMBEDDING_DIM,
    api_key=OPENAI_API_KEY,
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512  # can differ from ingest chunk_size; this is for new chunks only


# ---------- Vector store & query engine (S3 Vectors) ----------

# Your ingest wrote:
#   data: {"float32": embedding}
#   metadata: {"text_preview": "...", ...}
# So here we:
# - point to the same bucket/index
# - say data_type="float32"
# - set text_field="text_preview" so nodes.text is populated from that metadata key

vector_store = S3VectorStore(
    index_name_or_arn=VECTOR_INDEX_NAME,
    bucket_name_or_arn=VECTOR_BUCKET_NAME,
    data_type="float32",
    text_field="text_preview",   # 🔑 match metadata key from ingest
    sync_session=session,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    # embed_model is already in Settings, so we can rely on that
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
)


# ---------- Tools & agent ----------

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


skiing_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="skiing_guide_tool",
        description=(
            "Retrieval Q&A over a skiing technique knowledge base. "
            "Use this to look up definitions, biomechanics, drills, "
            "and equipment tuning for specific errors."
        ),
    ),
)

system_prompt = """
You are an AI Skiing Coach. The user input is a timeline of detected skiing ERRORS with time slots
(e.g., list of events: {label, start_time, end_time, confidence}). Your job is to:

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
   a JSON object with this schema:

   {
     "session_summary": {
       "total_errors": <int>,
       "top_errors": [{"label": <str>, "count": <int>, "risk": "Low|Med|High"}]
     },
     "coaching_note": <str>,
     "timeline_highlights": [{"label": <str>, "start": "<mm:ss>", "end": "<mm:ss>", "confidence": <0-1>}],
     "coaching_plan": [
       {
         "label": <str>,
         "priority": 1,
         "cues": ["...","..."],
         "drills": ["...","..."],
         "practice_terrain": "flats|green|blue",
         "focus_timecodes": ["mm:ss-mm:ss", "..."]
       }
     ],
     "safety_notes": ["...", "..."]
   }

Rules:
- Respond ONLY with JSON, no extra text.
- Prefer concrete, short cues ("hips to hands", "shins to tongue", "tip, roll, pressure").
- Tie every recommendation to observed timecodes when possible.
- Use the skiing_guide_tool when you need drills/definitions.
- If an error label is unknown, infer from context and state the assumption.
- If the input lacks time fields, still provide cues/drills and use [] for timecodes.
"""

agent = FunctionAgent(
    tools=[skiing_tool],
    llm=llm,
    system_prompt=system_prompt,
    output_cls=SkiingCoachOutput,
    allow_parallel_tool_calls=True,
)


# ---------- FastAPI app ----------

app = FastAPI()

class AnalyzeRequest(BaseModel):
    events: Any  # you can tighten this schema later (List[Event], etc.)


@app.post("/analyze_session")
async def analyze_session(payload: AnalyzeRequest) -> Dict[str, Any]:
    """
    Input: { "events": [...] }
    Output: JSON object as defined in system_prompt.
    """
    try:
        user_input = {
            "events": payload.events
        }
        prompt = f"Skiing error timeline JSON: {user_input}"
        resp = await agent.run(prompt)

        data_model = resp.get_pydantic_model(SkiingCoachOutput)
        text = data_model.json()

        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail={"raw": text})

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


