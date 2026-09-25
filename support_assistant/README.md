# Module 3 — Support Assistant

## Offline graded mode
Leave `MOCK_LLM` unset or set `MOCK_LLM=1`. No LLM API call is required.

## Install
From repository root:

```bash
pip install -r requirements.txt
```

## Run
```bash
cd support_assistant
uvicorn main:app --reload --port 7860
```

Example policy query:
```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"What is the delivery fee below INR 149?"}'
```

Example general query:
```bash
curl -X POST http://127.0.0.1:7860/ask   -H "Content-Type: application/json"   -d '{"query":"What is the capital of India?"}'
```

## Architecture
**Ingestion → Embedding → Retrieval → Generation**

1. `docs/doc_01.txt` ... `doc_08.txt` are loaded by `ingest()`.
2. `SentenceTransformer(all-MiniLM-L6-v2)` embeds each complete document.
3. ChromaDB stores the vectors in the `zepto_policies` collection.
4. `classify_intent` routes policy questions to `retrieve_and_answer`.
5. `retrieve_and_answer` embeds the query and retrieves the top 3 chunks by cosine similarity.
6. In default mock mode, the answer is deterministically built from the top retrieved chunk.
7. `direct_answer` handles general questions with the required fixed mock response.

The `MOCK_LLM` toggle affects generation/classification integration, not the ChromaDB
retrieval itself. The required baseline is deterministic and offline.
