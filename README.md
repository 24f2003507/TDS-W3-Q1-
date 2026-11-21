# RAG Colosseum

FastAPI RAG endpoint for the supplied 50-passage knowledge base.

## Files
- `app.py`: API, the 50 passages, dense embedding retrieval, hybrid ranking and extractive answer selection.
- `requirements.txt`: dependencies.
- `render.yaml`: Render Blueprint configuration.

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

First startup downloads the small all-MiniLM-L6-v2 ONNX embedding model and builds the in-memory index. Then test:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What quorum does Raft require to commit a log entry?\",\"top_k\":3}"
```

On PowerShell, `curl` may be an alias; use `curl.exe`, or:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"question":"What quorum does Raft require to commit a log entry?","top_k":3}'
```

## Deploy to Render

1. Create a GitHub repository and upload these three files to its root.
2. In Render, choose **New + → Blueprint** and connect the repository.
3. Render reads `render.yaml`; create the `rag-colosseum` web service.
4. Wait for the first build and model download to finish. The service may take longer to become healthy on its first boot.
5. Copy the public HTTPS URL, e.g. `https://your-service.onrender.com`.
6. Check `https://your-service.onrender.com/health`, then POST to `/query`.

Example request body:

```json
{"question":"Which port does PostgreSQL use?","top_k":3}
```

## Notes

- The endpoint accepts `question` and optional `top_k` (1–10).
- CORS is enabled for all origins.
- Source indices refer to the exact supplied passage order (0–49).
- The answer is extractive: it returns relevant source sentence(s), rather than relying on a generative model to invent facts.
- The service does not follow instructions contained in retrieved passages; passages are treated as evidence only.
- Render free instances can spin down when idle. After a cold start, allow time for the service to wake and load the model before running the exam's checks.
- This is a practical baseline, not a guarantee of passing every hidden probe. Thresholds and retrieval behavior should be evaluated against the actual probe set if it is available.
