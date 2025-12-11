# AI Skiing Coach – API Server

This service provides a **structured AI skiing coaching analysis** based on a timeline of detected skiing errors.
It uses:

* **FastAPI** for the HTTP interface
* **OpenAI (LLM + embeddings)**
* **LlamaIndex** with an **S3VectorStore** for retrieval
* A custom **FunctionAgent** producing fully structured coaching output

---

## 🚀 Features

* Accepts a list of skiing error events with timestamps
* Performs retrieval-augmented reasoning using an S3-backed vector index
* Produces a structured JSON coaching report including:

  * Session summary
  * Top errors
  * Coaching notes
  * Timeline highlights
  * Prioritized coaching plan
  * Safety considerations
* Guaranteed output schema via `SkiingCoachOutput` Pydantic model

---

# 🔧 1. Running the Server

## Install dependencies

```bash
pip install fastapi uvicorn boto3 llama-index openai
```

*(Your project may include additional dependencies such as python-dotenv, pydantic v2, etc.)*

## Start the API server

```bash
uvicorn agent_server:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```
POST http://localhost:8000/analyze_session
```

---

# 🌱 2. Environment Variables

The following environment variables **must** be set:

| Variable             | Purpose                        | Default                  |
| -------------------- | ------------------------------ | ------------------------ |
| `AWS_REGION`         | AWS region                     | `us-east-2`              |
| `VECTOR_BUCKET_NAME` | S3 bucket holding vector index | `skiing-rag-vectors`     |
| `VECTOR_INDEX_NAME`  | S3 index name                  | `skiing-rag-index`       |
| `EMBEDDING_DIM`      | Embedding dimensionality       | `1024`                   |
| `OPENAI_API_KEY`     | **Required** OpenAI API key    | *(none)*                 |
| `LLM_MODEL`          | LLM used by the agent          | `gpt-4o-mini`            |
| `EMBEDDING_MODEL`    | Embedding model                | `text-embedding-3-small` |

These must match the settings used during your **ingestion pipeline**.

---

# 📡 3. API Endpoint

## **POST** `/analyze_session`

Analyzes the skiing session using the LLM + vector retrieval and returns a structured coaching plan.

---

# 📝 4. Request Format

Top-level request object:

```json
{
  "events": <any JSON value, typically an array of error events>
}
```

Although `events` is typed as `Any`, the recommended format is an array of event objects:

```json
{
  "events": [
    {
      "label": "back seat",
      "start_time": "00:12",
      "end_time": "00:18",
      "confidence": 0.92
    },
    {
      "label": "late edge set",
      "start_time": "00:41",
      "end_time": "00:55",
      "confidence": 0.88
    }
  ]
}
```

### Recommended Event Fields

| Field        | Type        | Meaning                 |
| ------------ | ----------- | ----------------------- |
| `label`      | string      | Error name              |
| `start_time` | `"mm:ss"`   | Start of error in video |
| `end_time`   | `"mm:ss"`   | End of error            |
| `confidence` | float (0–1) | Detection confidence    |

Additional fields are allowed (e.g., slope, speed).

---

# 📤 5. Response Format

The response always conforms to this schema:

```json
{
  "session_summary": {
    "total_errors": <int>,
    "top_errors": [
      { "label": <str>, "count": <int>, "risk": "Low|Med|High" }
    ]
  },

  "coaching_note": <string>,

  "timeline_highlights": [
    {
      "label": <str>,
      "start": "<mm:ss>",
      "end": "<mm:ss>",
      "confidence": <float>
    }
  ],

  "coaching_plan": [
    {
      "label": <str>,
      "priority": <int>,
      "cues": [<str>, ...],
      "drills": [<str>, ...],
      "practice_terrain": "flats|green|blue",
      "focus_timecodes": ["mm:ss-mm:ss", ...]
    }
  ],

  "safety_notes": [<string>, ...]
}
```

---

# 🧩 6. Field Definitions

### `session_summary`

| Field          | Type  | Description                       |
| -------------- | ----- | --------------------------------- |
| `total_errors` | int   | Total number of detected errors   |
| `top_errors`   | array | Most frequent or impactful errors |

### `top_errors[]`

| Field   | Type   | Description                |
| ------- | ------ | -------------------------- |
| `label` | string | Error category             |
| `count` | int    | Occurrences                |
| `risk`  | enum   | `"Low"`, `"Med"`, `"High"` |

---

### `timeline_highlights[]`

| Field        | Type      |
| ------------ | --------- |
| `label`      | string    |
| `start`      | `"mm:ss"` |
| `end`        | `"mm:ss"` |
| `confidence` | float     |

---

### `coaching_plan[]`

| Field              | Type                            |
| ------------------ | ------------------------------- |
| `label`            | string                          |
| `priority`         | integer (1 = highest)           |
| `cues`             | array of strings                |
| `drills`           | array of strings                |
| `practice_terrain` | `"flats"`, `"green"`, `"blue"`  |
| `focus_timecodes`  | array of `"mm:ss-mm:ss"` ranges |

---

# 📘 7. Example Request

```json
{
  "events": [
    {
      "label": "back seat",
      "start_time": "00:12",
      "end_time": "00:18",
      "confidence": 0.92
    },
    {
      "label": "late edge set",
      "start_time": "00:41",
      "end_time": "00:55",
      "confidence": 0.88
    }
  ]
}
```

---

# 📙 8. Example Response (abridged)

```json
{
  "session_summary": {
    "total_errors": 3,
    "top_errors": [
      { "label": "back seat", "count": 2, "risk": "High" },
      { "label": "late edge set", "count": 1, "risk": "Med" }
    ]
  },
  "coaching_note": "You fall into the back seat on medium-speed turns...",
  "timeline_highlights": [
    {
      "label": "pronounced back seat",
      "start": "00:41",
      "end": "00:55",
      "confidence": 0.9
    }
  ],
  "coaching_plan": [
    {
      "label": "fix back-seat stance",
      "priority": 1,
      "cues": ["shins to tongue", "hips over feet"],
      "drills": ["static ankle flex drill", "green terrain slow turns"],
      "practice_terrain": "green",
      "focus_timecodes": ["00:41-00:55"]
    }
  ],
  "safety_notes": [
    "Avoid back-seat skiing at speed...",
    "Check brake function on icy terrain."
  ]
}
```

---

# ⚠️ 9. Error Handling

If the model returns invalid JSON or fails schema validation, the server returns:

```json
{
  "detail": {
    "raw": "<raw model output>"
  }
}
```

or

```json
{
  "detail": "Internal error message"
}
```

Status code: **500**

---

# 🧪 10. Testing Locally

You can test using cURL:

```bash
curl -X POST http://localhost:8000/analyze_session \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

Or with a sample body inline:

```bash
curl -X POST http://localhost:8000/analyze_session \
  -H "Content-Type: application/json" \
  -d '{"events":[{"label":"back seat","start_time":"00:12","end_time":"00:18","confidence":0.92}]}'
```

---

# 🎿 11. Internal Architecture (Optional)

* **Agent**: `FunctionAgent` with:

  * system prompt defining skiing-specific logic & output format
  * `skiing_guide_tool` (vector retrieval tool)
* **Retrieval**: `VectorStoreIndex` over **S3VectorStore**
* **LLM**: OpenAI model
* **Schema Enforcement**: `output_cls=SkiingCoachOutput`

---
