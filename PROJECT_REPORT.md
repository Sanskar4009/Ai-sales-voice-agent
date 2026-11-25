# One-Page Project Report: Fortivex Autonomous AI Sales Voice Agent ("Forvi")

## 1. Executive Summary & Business Problem
In B2B sales, contacting an inbound prospect more than 5 minutes after inquiry reduces qualification rates by over 80%. Relying solely on human Sales Development Representatives (SDRs) introduces high payroll costs, limited operating hours, and inconsistent CRM logging. **Forvi** solves this as an autonomous, real-time AI Sales Voice Agent that conducts natural spoken discovery calls, grounds answers in company documentation via RAG, syncs qualified leads and demo bookings into the CRM in real time, and generates structured post-call RevOps intelligence.

## 2. Technical Architecture & Hybrid ML Stack
Forvi uses a low-latency, full-duplex streaming pipeline built for sub-800ms conversational turnaround:
* **Transport & API Layer:** FastAPI + Bi-directional WebSockets (/ws/voice/{session_id}) streaming browser MediaRecorder WebM audio frames and real-time UI state events.
* **Speech-to-Text (ASR):** Groq-hosted whisper-large-v3-turbo for ultra-fast English speech transcription.
* **Grounded Knowledge Retrieval (RAG):** Local ll-MiniLM-L6-v2 (PyTorch) dense embeddings indexed in **FAISS (IndexFlatIP)** over data/product_kb.md, eliminating pricing and feature hallucinations.
* **Reasoning & Autonomous Tool Calling:** Groq LPU LLM managing multi-turn conversation state and invoking upsert_lead and ook_appointment tools against a transactional SQLAlchemy database.
* **Neural Text-to-Speech (TTS):** Deepgram Aura (ura-asteria-en) with automatic browser Web Speech API fallback, paired with a custom Carbon Teal & Mint Foam interactive dashboard.

## 3. End-to-End Business Workflow
1. **RAG-Grounded Discovery:** Answers product, security, and pricing questions in concise 2–3 sentence voice turns.
2. **Real-Time Lead Capture:** Extracts prospect name, company, email, and budget mid-conversation and upserts the CRM record.
3. **Atomic Calendar Booking:** Checks open demo slots and books appointments transactionally without double-booking conflicts.
4. **Post-Call JSON Intelligence:** Evaluates the full transcript on call completion to output Sentiment (Positive/Neutral/Negative), Lead Tier (Hot/Warm/Cold), Summary Bullets, Objections, and Next Best Action.

## 4. Scalability (100,000 Users), Security & Unit Economics
* **100k Scale Architecture:** Designed for 3,000–5,000 peak concurrent voice sessions using stateless FastAPI pods behind an L7 Load Balancer with Kubernetes HPA, **Redis Cluster** (session memory & Pub/Sub), **Celery + Kafka/RabbitMQ** async post-call workers, **Managed PostgreSQL + PgBouncer** (FOR UPDATE SKIP LOCKED), and **Qdrant/Pinecone** with tenant-isolated namespaces.
* **Security & Compliance:** WSS/TLS 1.3 encryption in transit, AES-256 at rest, PII redaction in logs, RBAC, and GDPR/DPDP alignment.
* **Commercial Feasibility:** End-to-end voice inference costs **~.025–.04/min (₹2.1–₹3.4/min)**—delivering **>85% operational cost savings** over traditional SDR workflows while scaling 24/7.
