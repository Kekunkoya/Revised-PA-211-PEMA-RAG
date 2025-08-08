# Revised-PA-211-PEMA-RAG
Revised-PA-211-PEMA-RAG
Revised-PA-211-PEMA-RAG is a Retrieval-Augmented Generation (RAG) system built to support PA 211 and PEMA in delivering fast, accurate, and context-aware disaster response information.
This Streamlit-based demo answers questions from PA emergency-preparedness PDFs using multiple retrieval strategies and compares results from OpenAI, Google Gemini, and a merged “Both” mode.

🔹 Core Features
Multi-Model Support — Compare OpenAI, Gemini, or a merged combined answer.
Five Retrieval Pipelines:
Standard — Fast, precise chunk retrieval.
Contextual — Expands chunks with neighbors for broader recall.
Query Transformation — Rewrites, steps back, and decomposes questions for better retrieval.
Adaptive — Classifies query type and chooses the optimal retrieval strategy.
Combined — Merges Standard + Contextual (+ optional transforms).
Vector Stores — Separate prebuilt stores for OpenAI and Gemini embeddings for speed.
Optional QA Memory — Small hint index from PA211_expanded_dataset.json to enhance context.
Similarity Scoring — Reports cosine similarity and answer-to-context alignment for each output.

-----
🔹 Data Sources
The system indexes multiple authoritative PA 211 / PEMA documents, such as:
211 Responds to Urgent Needs.pdf
PA 211 Disaster Community Resources.pdf
PEMA.pdf
Ready.gov Disaster Preparedness Guide for Older Adults.pdf
Substantial Damages Toolkit.pdf

------
🔹 Workflow
Ingest PDFs → extract text → chunk (900 chars, 250 overlap).
Build vector stores for OpenAI & Gemini embeddings.
Retrieve top-K results per query using selected pipeline(s).
Generate answers via OpenAI, Gemini, or Both mode.
Display metrics & retrieved context for transparency.


-----
🔹 Why It Matters
During disasters, speed and accuracy save lives. This RAG platform enables PA 211 and PEMA to:
Provide verified, consistent answers from trusted documents.
Compare AI outputs for quality assurance.
Adapt retrieval strategies based on the nature of the query.

