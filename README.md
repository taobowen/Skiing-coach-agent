# 🏂🏻 **Guidex – AI Skiing Coach**

**AI-powered skiing technique analysis and personalized coaching system**

Guidex is an end-to-end AI skiing coach that analyzes user-uploaded skiing videos, detects technique errors, and generates personalized, timestamped coaching feedback. It blends pose estimation, ML classification, retrieval-augmented generation (RAG), and LLM coaching to deliver a professional, data-driven training experience.

## **Key Features**

### **Automatic Video Analysis**

* Upload a skiing or snowboarding video
* System extracts keyframes and runs **AlphaPose** for pose estimation
* Computes time-series features for ML inference

### **Technique Error Detection**

* Transformer-based classifier trained on **Amazon SageMaker**
* Identifies common skiing errors such as:

  * Back-seat stance
  * Late edge engagement
  * Upper body rotation
  * A-frame
* Groups errors into labeled time segments with confidence scores

### **AI Coaching Agent**

* Error labels & time ranges are sent to an LLM-powered coaching agent
* Generates:

  * Actionable coaching cues
  * Targeted drills
  * Terrain recommendations
  * Safety notes
* Produces structured, timestamp-aware coaching reports

### 🔍 **RAG Technique Knowledge Engine**

* Skiing-technique documents stored in **Amazon S3**
* Automatic translation for multilingual sources
* Embedded using **Bedrock Titan Embeddings**
* Indexed in **Amazon OpenSearch Serverless**
* Reranked with **Bedrock Rerank** for high relevance
* Used by the coaching agent to return ski-specific, validated technique guidance

### **Event-Driven Pipeline**

* New video uploads trigger the pipeline via **S3 → EventBridge**
* Fully automated ingestion, inference, and coaching generation
* Logs and metadata stored in MySQL

### **Mobile App (React Native)**

* iOS app for:

  * Video capture & upload
  * Session history
  * Coach/student modes
  * Viewing AI-generated feedback

---

## **Architecture Overview**

```
User Video Upload (React Native)
            │
            ▼
        Amazon S3
            │ (S3 → EventBridge Trigger)
            ▼
   Extraction & Pose Estimation (AlphaPose)
            │
            ▼
 Time-Series Feature Builder (Python)
            │
            ▼
Transformer Classifier (SageMaker)
            │
            ▼
  Error Labels + Timeline Segments
            │
            ▼
      LLM Coaching Agent
(OpenAI + LlamaIndex + Tools)
            │
            ▼
       RAG Knowledge Engine
(S3 → Translate → Titan Embed → OpenSearch)
            │
            ▼
     Personalized Coaching Report
```

---

## **Tech Stack**

### **Frontend**

* React Native (iOS)
* Expo
* AWS Amplify (optional)

### **Backend / ML**

* Python
* AlphaPose
* Amazon SageMaker (Training + Inference)
* MySQL (session metadata)

### **AI & RAG**

* Amazon Bedrock (LLM, Embeddings, Rerank)
* LlamaIndex
* Amazon OpenSearch Serverless
* S3-based corpus ingestion pipeline

### **Infrastructure**

* Amazon S3
* EventBridge triggers
* Lambda / ECS tasks
* IAM, CloudWatch
* Docker

## **How It Works (Quick Start)**

### **1. Upload a skiing video**

Use the iOS app or send a file directly to the S3 bucket.

### **2. EventBridge triggers processing**

Automatically starts:

* Pose estimation
* Feature extraction
* Classifier inference

### **3. Error labels generated**

Classifier outputs structured errors with time ranges.

### **4. Coaching agent produces insights**

LLM + RAG generate:

* Cues
* Drills
* Terrain suggestions
* Safety notes

### **5. Feedback is returned**

Displayed in the mobile app or API response.

---

## **Example Output**

```json
{
  "session_summary": {
    "total_errors": 14,
    "top_errors": [
      {"label": "Back-seat stance", "count": 6, "risk": "High"},
      {"label": "Late edge engagement", "count": 5, "risk": "Medium"}
    ]
  },
  "coaching_plan": [
    {
      "label": "Back-seat stance",
      "cues": ["Shins to tongue", "Hands forward"],
      "drills": ["Javelin turns", "Side-slip balance"],
      "practice_terrain": "Green",
      "focus_timecodes": ["01:10-01:24", "02:03-02:16"]
    }
  ],
  "safety_notes": ["Reduced control at higher speed", "Focus on balance before steeper terrain"]
}
```

---

## **Contributing**

Pull requests and feature suggestions are welcome.
Feel free to open issues for:

* Bug reports
* Model improvements
* Feature requests
* New technique documents for RAG
